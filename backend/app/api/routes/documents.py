import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import _collection_name
from app.core.security import get_current_user, require_admin
from app.rag.document_loader import DocumentLoader
from app.rag.embedding import get_embedding_service
from app.rag.milvus_client import MilvusManager
from app.rag.milvus_writer import MilvusWriter
from app.models import User
from app.rag.parent_chunk_store import ParentChunkStore
from app.schemas import (
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
)
from app.infrastructure.upload_jobs import DELETE_STEPS, delete_job_manager, upload_job_manager

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # backend/
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(milvus_manager=milvus_manager)

router = APIRouter()


def _remove_bm25_stats_for_filename(filename: str, *, collection_name: str) -> None:
    """删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。"""
    rows = milvus_manager.query_all(
        collection_name,
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    texts = [r.get("text") or "" for r in rows]
    emb_svc = get_embedding_service(collection_name)
    emb_svc.increment_remove_documents(texts)


def _is_supported_document(filename: str) -> bool:
    '''门卫函数:支持上传的文件格式'''
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith(".txt")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
        or file_lower.endswith(".md")
    )


async def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    """按块写入上传文件，避免大文件一次性读入内存。"""
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _process_upload_job(job_id: str, file_path: str, filename: str, *, collection_name: str) -> None:
    """上传编排器，后台执行耗时的解析、分块、向量化入库，并持续更新任务进度。"""
    failed_step = "cleanup"
    try:
        upload_job_manager.complete_step(job_id, "upload", "文件已保存到服务器")

        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
        milvus_manager.init_collection(collection_name)
        delete_expr = f'filename == "{filename}"'
        try:
            _remove_bm25_stats_for_filename(filename, collection_name=collection_name)
        except Exception:
            pass
        try:
            milvus_manager.delete(collection_name, delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass
        upload_job_manager.complete_step(job_id, "cleanup", "旧版本清理完成")

        failed_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "正在解析文档并执行三级分块")
        new_docs = loader.load_document(file_path, filename)
        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        upload_job_manager.complete_step(
            job_id,
            "parse",
            f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个",
        )

        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "正在写入父级分块")
        parent_chunk_store.upsert_documents(parent_docs)
        upload_job_manager.complete_step(job_id, "parent_store", f"父级分块已入库：{len(parent_docs)} 个")

        failed_step = "vector_store"
        total_leaf = len(leaf_docs)
        upload_job_manager.update_step(
            job_id,
            "vector_store",
            0,
            "running",
            f"正在向量化入库：0 / {total_leaf}",
            total_chunks=total_leaf,
            processed_chunks=0,
        )

        def _on_vector_progress(processed: int, total: int) -> None:
            percent = round(processed * 100 / total) if total else 100
            upload_job_manager.update_step(
                job_id,
                "vector_store",
                percent,
                "running",
                f"正在向量化入库：{processed} / {total}",
                total_chunks=total,
                processed_chunks=processed,
            )

        milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress, collection_name=collection_name)
        upload_job_manager.complete_step(job_id, "vector_store", f"向量化入库完成：{total_leaf} 个叶子分块")
        upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
    except Exception as e:
        upload_job_manager.fail_job(job_id, failed_step, str(e))


def _process_delete_job(job_id: str, filename: str, *, collection_name: str) -> None:
    """后台执行文档删除，并把每个删除阶段同步给前端行内进度卡片。"""
    failed_step = "prepare"
    try:
        failed_step = "prepare"
        delete_job_manager.update_step(job_id, "prepare", 20, "running", "正在初始化 Milvus 集合")
        milvus_manager.init_collection(collection_name)
        delete_expr = f'filename == "{filename}"'
        delete_job_manager.complete_step(job_id, "prepare", "删除任务已创建")

        failed_step = "bm25"
        delete_job_manager.update_step(job_id, "bm25", 20, "running", "正在同步 BM25 统计")
        _remove_bm25_stats_for_filename(filename, collection_name=collection_name)
        delete_job_manager.complete_step(job_id, "bm25", "BM25 统计已同步")

        failed_step = "milvus"
        delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在删除 Milvus 向量数据")
        result = milvus_manager.delete(collection_name, delete_expr)
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        delete_job_manager.complete_step(job_id, "milvus", f"向量数据已删除：{deleted_count} 条")

        failed_step = "parent_store"
        delete_job_manager.update_step(job_id, "parent_store", 30, "running", "正在删除 Redis/PostgreSQL父级分块")
        parent_chunk_store.delete_by_filename(filename)
        delete_job_manager.complete_step(job_id, "parent_store", "父级分块已删除")

        # 完成摘要会由前端保留 3 秒，再自动从文档列表移除。
        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条")
    except Exception as e:
        delete_job_manager.fail_job(job_id, failed_step, str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(kb_id: int | None = None, _: User = Depends(require_admin)):
    """获取已上传的文档列表（管理员）从 Milvus 查。因为文档列表的本质是"哪些文件有向量 chunk"，Milvus 是权威源"""
    try:
        collection_name = _collection_name(kb_id)
        milvus_manager.init_collection(collection_name)

        results = milvus_manager.query(
            collection_name,
            output_fields=["filename", "file_type"],
            limit=10000,
        )

        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    kb_id: int | None = Form(None),
    _: User = Depends(require_admin),
):
    """轻量版异步上传：文件落盘后立即返回 job_id，后台继续解析和向量化。也就是add_task(_process_upload_job)  ← 把重活丢给后台线程"""
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not _is_supported_document(filename):
        raise HTTPException(status_code=400, detail="仅支持 PDF、TXT、MarkDown、Word 和 Excel 文档")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job = upload_job_manager.create_job(filename)
    file_path = UPLOAD_DIR / filename

    try:
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", "正在保存文件到服务器")
        await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(job["job_id"], "upload", "文件已上传，等待后台处理")
    except Exception as e:
        upload_job_manager.fail_job(job["job_id"], "upload", f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    background_tasks.add_task(_process_upload_job, job["job_id"], str(file_path), filename, collection_name=_collection_name(kb_id))
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
    )


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    kb_id: int | None = None,
    current_user: User = Depends(require_admin),
):
    """轻量版异步删除：立即返回 job_id，实际删除在后台执行。"""
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    background_tasks.add_task(_process_delete_job, job["job_id"], filename, collection_name=_collection_name(kb_id))
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), kb_id: int | None = Form(None), _: User = Depends(require_admin)):
    """上传文档并进行 embedding（管理员）, 小文件用同步一步到位，与_process_upload_job 逻辑完全一致，只是 全部在主请求线程里跑完。"""
    try:
        collection_name = _collection_name(kb_id)
        filename = file.filename or ""
        file_lower = filename.lower()
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith(".txt")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
            or file_lower.endswith(".md")
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF、TXT、MarkDown、Word 和 Excel 文档")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        milvus_manager.init_collection(collection_name)

        delete_expr = f'filename == "{filename}"'
        try:
            _remove_bm25_stats_for_filename(filename, collection_name=collection_name)
        except Exception:
            pass
        try:
            milvus_manager.delete(collection_name, delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs, collection_name=collection_name)

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, kb_id: int | None = None, _: User = Depends(require_admin)):
    """删除文档在 Milvus 中的向量（保留本地文件，管理员）"""
    try:
        collection_name = _collection_name(kb_id)
        milvus_manager.init_collection(collection_name)

        delete_expr = f'filename == "{filename}"'
        _remove_bm25_stats_for_filename(filename, collection_name=collection_name)
        result = milvus_manager.delete(collection_name, delete_expr)
        parent_chunk_store.delete_by_filename(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
