"""文档向量化并写入 Milvus - 支持密集+稀疏向量"""
from app.rag.embedding import get_embedding_service
from app.rag.milvus_client import MilvusManager


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, milvus_manager: MilvusManager = None):
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 50, progress_callback=None, *, collection_name: str):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量），collection_name 按用户隔离。
        :param documents: 文档列表
        :param batch_size: 批次大小
        写入 50 条的向量维度是 50*1024*4字节 = 200KB，加上稀疏向量和元数据，单批内存约 500KB-1MB，很安全
        BGE-M3 一次编码 50 条比逐条编码快 5-10 倍，但超过 100 条后吞吐不再明显提升
        :param progress_callback: 给前端计算进度
        :param collection_name: 集合名称，按用户隔离
        """
        if not documents:
            return

        self.milvus_manager.init_collection(collection_name)
        '''
        下面这一步是关键时序——必须在 get_all_embeddings 之前。原因在学 embedding 时说过：BM25 的 IDF 需要知道新文档存在，否则新词的 IDF 是错的。
        注意这里是一次性传入全部，不是分批。因为 BM25 统计是整体更新的（_total_docs 累加、_doc_freq 累加），不需要和向量计算同步分批。
        '''
        all_texts = [doc["text"] for doc in documents]
        embedding_service = get_embedding_service(collection_name)
        embedding_service.increment_add_documents(all_texts)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]

            # 同时生成密集向量和稀疏向量
            dense_embeddings, sparse_embeddings = embedding_service.get_all_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            self.milvus_manager.insert(collection_name, insert_data)

            # 每个批次写入后更新进度，前端据此展示“向量化入库 xx%”。
            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)
