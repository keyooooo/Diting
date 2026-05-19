import logging
import json
import os
from typing import Any, Optional, Dict, List

import redis

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "diting")
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        self._client = None

    def _get_client(self):
        '''
        如果在 __init__ 里直接去连 Redis，万一网络有抖动或者 Redis 还没启动完毕，整个 Python 进程（比如 FastAPI 进程）就会直接崩溃报错。这种“什么时候用，什么时候才真正去初始化”的懒加载机制，能保证服务的健壮性。
        redis.Redis.from_url 在底层会自动帮你维护一个连接池（Connection Pool）。后续你高并发调用 _get_client() 时，它不是频繁建连断连，而是复用池子里的长连接，性能极高。
        '''
        if self._client is None:
            # 引入 socket_timeout(防止网络卡死死等) 和 health_check_interval(自动检测死连)
            self._client = redis.Redis.from_url(
                self.redis_url, 
                decode_responses=True,
                socket_timeout=2.0,
                health_check_interval=30
            )
        return self._client

    def _key(self, key: str) -> str:
        '''
        在企业开发中，一个 Redis 实例往往被多个微服务、或者同一个项目的测试/开发环境共用。通过 REDIS_KEY_PREFIX:key 这种前缀，有效防止了你的 RAG 分块数据把别人的用户登录 Token 给不小心覆盖掉。
        '''
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        """查询单条 JSON 数据"""
        try:
            value = self._get_client().get(self._key(key))
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Redis get_json fail! key: {key}, error: {e}")
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """写入单条 JSON 数据"""
        try:
            '''
            在 json.dumps 时，默认会将中文转成 \u4e2d\u6587 这样的 ASCII 编码。一个中文字符原本占 3 个字节，转成这种编码后暴增到 12 个字节！对于重度依赖文本存储的 RAG 系统来说，加上 ensure_ascii=False 能直接让 Redis 节省 50% 以上的内存空间。
            '''
            payload = json.dumps(value, ensure_ascii=False)
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        except Exception as e:
            logger.error(f"Redis set_json 失败, key: {key}, 错误: {str(e)}")

    def mset_json(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        批量写入 JSON 数据到 Redis（使用 Pipeline 降低网络 RTT 开销）
        :param mapping: 键值对字典，例如 {"key1": value1, "key2": value2}
        :param ttl: 过期时间（秒）
        """
        if not mapping:
            return

        try:
            client = self._get_client()
            
            # 1. 开启一个非事务型的管道 (transaction=False 性能更高)
            with client.pipeline(transaction=False) as pipe:
                for key, value in mapping.items():
                    # 2. 序列化数据
                    payload = json.dumps(value, ensure_ascii=False)
                    
                    # 3. 注意：这里是用 pipe.setex，把命令暂时缓存在本地管道里，不发生网络交互
                    real_key = self._key(key)
                    pipe.setex(real_key, ttl or self.default_ttl, payload)
                
                # 4. 核心大招：打包一次性发射到 Redis 服务器执行
                pipe.execute()
                
        except Exception as e:
            # 避坑指南：原代码里用 try...except Exception: return 把错误完全吞掉了。
            # 在批量操作时，建议至少打印一行日志，否则线上 Redis 挂了你根本无从知晓。
            logger.error(f"Redis mset_json 批量写入失败: {str(e)}")
        
    def mget_json(self, keys: List[str]) -> Dict[str, Any]:
        """
        原生的批量读取。
        传入一批 keys，一次性返回一个包含了所有命中缓存的字典。
        """
        if not keys:
            return {}
        
        result = {}
        try:
            client = self._get_client()
            # 1. 批量对齐前缀
            real_keys = [self._key(k) for k in keys]
            
            # 2. 调用 Redis 原生高性能 MGET 命令，单次网络交互拿回所有数据
            raw_values = client.mget(real_keys)
            
            # 3. 重新组装回 Dict
            for original_key, raw_val in zip(keys, raw_values):
                if raw_val:
                    try:
                        result[original_key] = json.loads(raw_val)
                    except json.JSONDecodeError:
                        continue # 容错，万一某条缓存损坏了不影响其他数据
            return result
        except Exception as e:
            logger.error(f"Redis mget_json 批量读取失败: {str(e)}")
            return {}
        
    def delete(self, key: str) -> None:
        """删除单条数据"""
        try:
            self._get_client().delete(self._key(key))
        except Exception as e:
            logger.error(f"Redis delete fail!, key: {key}, error: {str(e)}")

    def delete_pattern(self, pattern: str) -> None:
        '''无阻塞地按模糊匹配删除。'''
        try:
            client = self._get_client()
            full_pattern = self._key(pattern)
            # scan_iter 内部以 1000 条（count）为单位分批滚动抓取 keys，绝不锁死 Redis
            # keys = self._get_client().keys(full_pattern)
            batch_keys = []
            for key in client.scan_iter(match=full_pattern, count=1000):
                batch_keys.append(key)
                # 攒满 500 条就批量删一次，防止内存撑爆
                if len(batch_keys) >= 500:
                    client.delete(*batch_keys)
                    batch_keys = []
            # 别忘了清除尾数
            if batch_keys:
                client.delete(*batch_keys)
        except Exception as e:
            logger.error(f"Redis delete_pattern 模糊删除失败, pattern: {pattern}, 错误: {str(e)}")

cache = RedisCache()
