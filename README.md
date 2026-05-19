# Diting — Agentic RAG 智能问答系统

基于 LangGraph 的 Agent 驱动问答后端，集成混合检索、自反思检索管道与流式对话，支持多知识库隔离与全链路实时可观测。

## 架构概览

```
用户上传文档                    用户提问
    │                              │
    ▼                              ▼
DocumentLoader                LangGraph Agent
 三级滑窗分块                  ReAct 工具调用
    │                              │
    ├── L1/L2 父块 ──► PG + Redis  │
    │                              ▼
    └── L3 叶子块       RAG Pipeline (4 节点状态图)
         │               检索 → 评分 → 重写 → 扩展检索
         ▼                              │
     EmbeddingService         ┌─────────┴─────────┐
     BGE-M3 + BM25            │                   │
         │                    ▼                   ▼
         ▼               Hybrid Search       Jina Rerank
      Milvus             Dense + Sparse      交叉编码器精排
   HNSW + SPARSE         RRF 融合排序        Auto-Merging
   _INVERTED_INDEX           │               L3→L2→L1
                              ▼
                         LLM 生成答案
                        Streaming SSE
```

## 核心特性

- **Agent 驱动对话**：LangGraph 状态图 + LangChain `create_agent` 实现 ReAct 工具调用，支持同步/流式双模式
- **双向量混合检索**：Dense（BGE-M3, 1024维）+ Sparse（自实现 BM25），Milvus Hybrid Search + RRF 融合，兼顾语义与关键词匹配
- **自反思检索管道**：4 节点 LangGraph 状态图，LLM 结构化评分判定召回质量，不达标时自动触发 Step-back / HyDE 查询重写并扩展检索
- **三层分块与 Auto-Merging**：L1/L2/L3 递进分块，Leaf-Only 向量化存储，检索时子块满足阈值自动合并为父块
- **实时过程可观测**：异步事件桥接机制穿透同步工具执行，前端实时展示 Searching → Grading → Rewriting 全链路
- **BM25 统计持久化**：词表、df 及平均文档长度按知识库分文件增量持久化，入库/删除时自动同步统计
- **多知识库隔离**：Milvus Collection、EmbeddingService、BM25 状态均按知识库独立管理
- **多层容错降级**：Hybrid → Dense 自动降级，Rerank 失败保留 RRF 排序，Grader 不可用时安全偏向重写，客户端断开自动取消 Agent 任务

## 技术栈

| 层次       | 技术                                            |
| ---------- | ----------------------------------------------- |
| 框架       | Python 3.12+, FastAPI, Uvicorn                  |
| Agent/编排 | LangChain, LangGraph                            |
| 向量数据库 | Milvus 2.5 (HNSW + SPARSE_INVERTED_INDEX)       |
| 关系数据库 | PostgreSQL 15 (SQLAlchemy 2.0 + SQLModel)       |
| 缓存       | Redis 7                                         |
| 嵌入模型   | BAAI/bge-m3 (HuggingFace sentence-transformers) |
| 稀疏向量   | 自实现 BM25（增量统计 + 持久化）                |
| 精排       | Jina Rerank（交叉编码器）                       |
| 文档解析   | pypdf, docx2txt, openpyxl, Unstructured         |
| 鉴权       | JWT (HS256) + PBKDF2-SHA256                     |
| 部署       | Docker Compose, uv                              |

## 快速开始

### 前置要求

- Python 3.12+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv)（Python 包管理器）

### 1. 启动基础设施

```bash
docker compose up -d
```

启动 PostgreSQL、Redis、Milvus Standalone、etcd、MinIO。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入必填项：

```env
# LLM 模型（OpenAI 兼容 API）
ARK_API_KEY=your-api-key(这里是火山方舟的key形式，请根据你自己的模型进行更换！！！)
MODEL=your-model-id
BASE_URL=https://your-api-endpoint/v1

# 基础设施（容器部署保持默认即可）
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/langchain_app
REDIS_URL=redis://localhost:6379/0
MILVUS_HOST=localhost
MILVUS_PORT=19530

# JWT
JWT_SECRET_KEY=your-random-secret-at-least-32-chars

# 可选
# RERANK_MODEL=your-rerank-model
# RERANK_BINDING_HOST=https://your-rerank-api
# RERANK_API_KEY=your-rerank-api-key
# AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
# AMAP_API_KEY=your-amap-key
```

### 3. 安装依赖

```bash
cd backend
uv sync
```

### 4. 启动后端

```bash
uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 访问

- 前端界面：http://localhost:8000
- Milvus 管理面板（Attu）：http://localhost:8080
- API 文档（Swagger）：http://localhost:8000/docs

## 项目结构

```
Diting/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 应用入口
│   │   ├── models.py                # SQLModel 数据模型
│   │   ├── schemas.py               # Pydantic 请求/响应模型
│   │   ├── core/
│   │   │   ├── database.py          # 数据库引擎与会话管理
│   │   │   └── security.py          # JWT 鉴权、密码哈希、限流
│   │   ├── api/
│   │   │   ├── main.py              # 路由聚合
│   │   │   ├── deps.py              # 共享依赖（kb_id → collection_name）
│   │   │   └── routes/
│   │   │       ├── auth.py          # 注册/登录
│   │   │       ├── chat.py          # 同步 & 流式对话
│   │   │       ├── sessions.py      # 会话管理
│   │   │       ├── documents.py     # 文档上传/删除（异步任务 + 进度轮询）
│   │   │       └── knowledge_bases.py  # 知识库 CRUD
│   │   ├── agent/
│   │   │   ├── agent.py             # Agent 创建、同步/流式对话、摘要
│   │   │   ├── tools.py             # 工具定义、全局状态机、跨线程 RAG 推送
│   │   │   └── storage.py           # 对话持久化（PG + Redis 双层）
│   │   ├── rag/
│   │   │   ├── document_loader.py   # 多格式加载 + 三级递归分块
│   │   │   ├── embedding.py         # Dense (BGE-M3) + Sparse (BM25) 向量化
│   │   │   ├── milvus_client.py     # Milvus 连接管理、混合检索、降级
│   │   │   ├── milvus_writer.py     # 批量向量化写入
│   │   │   ├── retrieval.py         # 检索编排：Hybrid → Rerank → Auto-Merge
│   │   │   ├── pipeline.py          # LangGraph RAG 状态图（4 节点）
│   │   │   └── parent_chunk_store.py # 父块 PG + Redis 双层存储
│   │   └── infrastructure/
│   │       ├── cache.py             # Redis 缓存封装
│   │       └── upload_jobs.py       # 上传/删除任务进度管理
│   ├── frontend/
│   │   ├── index.html               # Vue 3 单页应用
│   │   ├── script.js                # 前端逻辑（流式对话、进度轮询）
│   │   └── style.css                # 样式
│   └── data/                         # BM25 状态文件（运行时生成）
├── docker-compose.yml                # 基础设施编排
├── pyproject.toml                    # 项目元数据与依赖
└── .env.example                      # 环境变量模板
```

## API 概览

| 端点                                 | 方法     | 说明                      |
| ------------------------------------ | -------- | ------------------------- |
| `/auth/register`                     | POST     | 用户注册                  |
| `/auth/login`                        | POST     | 用户登录                  |
| `/auth/me`                           | GET      | 获取当前用户信息          |
| `/chat`                              | POST     | 同步对话                  |
| `/chat/stream`                       | POST     | 流式对话（SSE）           |
| `/sessions`                          | GET      | 获取会话列表              |
| `/sessions/{id}`                     | DELETE   | 删除会话                  |
| `/documents`                         | GET      | 文档列表（管理员）        |
| `/documents/upload`                  | POST     | 同步上传文档              |
| `/documents/upload/async`            | POST     | 异步上传文档（含进度）    |
| `/documents/upload/jobs/{id}`        | GET      | 查询上传任务进度          |
| `/documents/delete/async/{filename}` | DELETE   | 异步删除文档              |
| `/documents/delete/jobs/{id}`        | GET      | 查询删除任务进度          |
| `/knowledge-bases`                   | GET/POST | 知识库列表/创建（管理员） |
| `/knowledge-bases/{id}`              | DELETE   | 删除知识库（管理员）      |

## 检索管道详解

```
用户提问
    │
    ▼
① retrieve_initial      Hybrid Search (Dense + Sparse, RRF k=60)
    │                   → candidate_k = top_k × 3 粗召回
    │                   → Jina Rerank 精排
    │                   → Auto-Merging (L3→L2→L1)
    │
    ▼
② grade_documents       LLM 结构化评分 (binary yes/no)
    │
    ├── yes → 生成答案
    │
    └── no  → ③ rewrite_question
                │       LLM Router 选择策略
                │       ├── step_back: 抽象退步问题
                │       ├── hyde: 生成假设文档嵌入
                │       └── complex: 两者并行
                │
                ▼
             ④ retrieve_expanded
                多路检索 → 去重 → 统一重排 → 生成答案
```

## License

MIT
