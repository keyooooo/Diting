"""父级分块文档存储（用于 Auto-merging Retriever）"""
from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy.dialects.postgresql import insert
import logging

from app.infrastructure.cache import cache
from app.core.database import SessionLocal
from app.models import ParentChunk

logger = logging.getLogger(__name__)

class ParentChunkStore:
    """基于 PostgreSQL + Redis 的父级分块存储。"""

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    def upsert_documents(self, docs: List[Dict[str, Any]], db_session=None) -> int:
        """
        高性能批量写入/更新（原生支持 PG ON CONFLICT 事务 + 缓存原子同步）
        """
        if not docs:
            return 0

        # 允许从外部 FastAPI Depends 注入 session，若无则自建兜底
        db = db_session or SessionLocal()
        current_time = datetime.now(timezone.utc)
        
        insert_payloads = []
        cache_mapping = {}

        # 1. 内存中完成数据结构对齐与清洗，绝不在循环中碰数据库
        for doc in docs:
            chunk_id = (doc.get("chunk_id") or "").strip()
            if not chunk_id:
                continue

            base_data = {
                "text": doc.get("text", ""),
                "filename": doc.get("filename", ""),
                "file_type": doc.get("file_type", ""),
                "file_path": doc.get("file_path", ""),
                "page_number": int(doc.get("page_number", 0) or 0),
                "parent_chunk_id": doc.get("parent_chunk_id", ""),
                "root_chunk_id": doc.get("root_chunk_id", ""),
                "chunk_level": int(doc.get("chunk_level", 0) or 0),
                "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
            }

            # 组装 DB 批量负载
            insert_payloads.append({
                "chunk_id": chunk_id,
                **base_data,
                "updated_at": current_time
            })

            # 组装 Redis 批量负载
            cache_mapping[self._cache_key(chunk_id)] = {"chunk_id": chunk_id, **base_data}

        if not insert_payloads:
            return 0

        try:
            # 2. 核心优化：利用 SQLAlchemy Core 跑原生 PG 批量 Upsert
            stmt = insert(ParentChunk).values(insert_payloads)
            
            # 定义当冲突发生时，需要被覆盖更新的列
            update_columns = {
                col.name: stmt.excluded[col.name] 
                for col in ParentChunk.__table__.columns 
                if col.name not in ["id", "chunk_id", "created_at"]
            }
            
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=["chunk_id"],  # 必须在模型中对 chunk_id 设立 UNIQUE 索引或主键
                set_=update_columns
            )
            db.execute(upsert_stmt)
            db.commit()

            # 3. 核心优化：利用 Redis 批量写机制减少网络 RTT
            # 假定你的 cache 工具类封装了 mset_json，如果没有，可以使用 redis.pipeline()
            if hasattr(cache, "mset_json"):
                cache.mset_json(cache_mapping)
            else:
                # 降级方案：利用纯粹的原生底层 pipeline 批量推送
                with cache.pipeline() as pipe:
                    for k, v in cache_mapping.items():
                        pipe.set_json(k, v)
                    pipe.execute()

            return len(insert_payloads)
        except Exception as e:
            db.rollback()
            logger.error(f"批量 Upsert 失败，事务已回滚。错误原因: {str(e)}")
            raise e
        finally:
            if not db_session:  # 如果是内部自建的，负责关闭
                db.close()

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """
        批量查询父块：Redis 缓存 → PostgreSQL 兜底 → 缓存回填 → 按原始顺序返回。
        全程最多三次网络往返（MGET + IN 查询 + Pipeline 回填），不在循环中访问网络。
        """
        if not chunk_ids:
            return []

        # Step 1: build cache_key <-> raw_id mapping (keys must match _cache_key format used by upsert)
        cache_key_to_id: Dict[str, str] = {}
        clean_ids: List[str] = []
        for cid in chunk_ids:
            raw_id = (cid or "").strip()
            if not raw_id:
                continue
            clean_ids.append(raw_id)
            cache_key_to_id[self._cache_key(raw_id)] = raw_id

        if not clean_ids:
            return []

        # Step 2: batch Redis read (single MGET)
        ordered_results: Dict[str, dict] = {}
        missing_ids: List[str] = []

        cached_map = cache.mget_json(list(cache_key_to_id.keys()))
        for cache_key, raw_id in cache_key_to_id.items():
            if cache_key in cached_map:
                ordered_results[raw_id] = cached_map[cache_key]
            else:
                missing_ids.append(raw_id)

        # Step 3: PostgreSQL fallback (single IN query)
        if missing_ids:
            db = SessionLocal()
            try:
                rows = (
                    db.query(ParentChunk)
                    .filter(ParentChunk.chunk_id.in_(missing_ids))
                    .all()
                )
                backfill_payload: Dict[str, dict] = {}
                for row in rows:
                    data = self._to_dict(row)
                    ordered_results[row.chunk_id] = data
                    backfill_payload[self._cache_key(row.chunk_id)] = data

                # Step 4: cache backfill (single Pipeline)
                if backfill_payload:
                    cache.mset_json(backfill_payload)
            finally:
                db.close()

        # 5. 按原始输入顺序返回，保证调用方依赖的顺序一致性
        return [ordered_results[cid] for cid in clean_ids if cid in ordered_results]

    def delete_by_filename(self, filename: str, db_session=None) -> int:
        """按文件名删除父级分块，返回删除条数。"""
        if not filename:
            return 0

        db = db_session or SessionLocal()
        try:
           # 1. 性能优化：在 SELECT 时只捞取 chunk_id 这一列，拒绝执行 SELECT * 降低内存与带宽
            rows = db.query(ParentChunk.chunk_id).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted_count = len(chunk_ids)

            if deleted_count > 0:
                # 2. 先淘汰 Redis 缓存，再删 DB（避免缓存脏读）
                for cid in chunk_ids:
                    cache.delete(self._cache_key(cid))

                # 3. 缓存斩断后，再重拳抹除 DB，保障高并发下的数据一致性
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()

            return deleted_count
        except Exception as e:
            db.rollback()
            logger.error(f"删除文件 {filename} 的父块数据失败: {str(e)}")
            raise e
        finally:
            if not db_session:
                db.close()
