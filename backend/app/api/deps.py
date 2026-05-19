def _collection_name(kb_id: int | None) -> str:
    """知识库 ID → Milvus collection 名称。无知识库时使用默认。"""
    return f"kb_{kb_id}" if kb_id else "default"
