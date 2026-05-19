"""文档加载和分片服务"""
from pathlib import Path
from typing import Any, Dict, List
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)

class DocumentLoader:

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # 1. 定义各层级的参数配置
        # 这里你可以自由调整比例，甚至从 config 文件读取
        self.configs = {
            1: {"size": max(1200, chunk_size * 2), "overlap": max(240, chunk_overlap * 2)},
            2: {"size": max(600, chunk_size),      "overlap": max(120, chunk_overlap)},
            3: {"size": max(300, chunk_size // 2), "overlap": max(60, chunk_overlap // 2)},
        }
        # 2. 动态生成并存储 Splitter
        # 使用字典推导式：{1: Splitter1, 2: Splitter2, 3: Splitter3}
        self.splitters = {
            level: self._create_splitter(cfg["size"], cfg["overlap"])
            for level, cfg in self.configs.items()
        }

    def _create_splitter(self, size: int, overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
        )

    
    # 决策层：将映射关系定义为类常量，方便扩展
    LOADER_MAPPING = {
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".pdf": (PyPDFLoader, {}),
        ".docx": (Docx2txtLoader, {}),
        ".doc": (Docx2txtLoader, {}),
        ".xls": (UnstructuredExcelLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
    }
    # 实现层
    def _get_loader(self, file_path: Path):
        """逻辑剥离：根据后缀名决定使用哪个加载器"""
        suffix = file_path.suffix.lower()
        if suffix not in self.LOADER_MAPPING:
            return None
        loader_class, kwargs = self.LOADER_MAPPING[suffix]
        return loader_class(str(file_path), **kwargs)
    
    def load_document(self, file_path: str | Path, filename: str | None = None) -> List:
        """
        职责：加载单个文件 → 提取 metadata → 三层递归分块 → 返回带父子关系的 chunk 列表。
        """
        path = Path(file_path)
        loader = self._get_loader(path)
        logger = logging.getLogger(__name__)
        if not loader:
            logger.warning(f"不支持的文件格式：{path.name}")
            return []

        try:
            raw_docs = loader.load()
            logger.info(f"成功加载：{path.name}，生成 {len(raw_docs)} 个原始 Document 对象")
        except Exception as e:
            logger.error(f"加载失败：{path.name}，错误：{str(e)}")
            return []

        file_name = filename or path.name
        file_type = path.suffix.lower()
        file_path_str = str(path)

        all_chunks = []
        for raw_doc in raw_docs:
            page_content = raw_doc.page_content or ""
            metadata = raw_doc.metadata or {}

            # PyPDFLoader 的页码在 "page" 字段（0-indexed），其他 loader 默认 0
            page_number = metadata.get("page", 0)
            if not isinstance(page_number, int):
                page_number = 0

            base_doc = {
                "filename": file_name,
                "file_type": file_type,
                "file_path": file_path_str,
                "page_number": page_number,
            }

            chunks = self._split_page_to_three_levels(
                text=page_content,
                base_doc=base_doc,
                page_global_chunk_idx=len(all_chunks),
            )
            all_chunks.extend(chunks)

        logger.info(f"文件 {file_name} 分块完成，共生成 {len(all_chunks)} 个层级块")
        return all_chunks
        
    def load_documents_from_folder(self, folder_path: str) -> List:
        """
        职责：只负责『遍历文件夹』。
        它不关心怎么加载 PDF 或 MD，它只负责把文件路径交给执行者。
        """
        all_docs = []
        folder = Path(folder_path)
        logger = logging.getLogger(__name__)
        if not folder.is_dir():
            logger.error(f"路径不存在或不是文件夹: {folder_path}")
            return []

        # 使用 pathlib 的 iterdir 替代 os.listdir，更现代
        for file_path in folder.iterdir():
            if file_path.is_file():
                # 调用单一执行函数
                docs = self.load_document(file_path)
                all_docs.extend(docs)
                
        return all_docs

    def _build_chunk_id(self, filename: str, page_number: int, level: int, idx: int) -> str:
            """辅助方法：构建唯一的 Chunk ID"""
            return f"{filename}::p{page_number}::l{level}::n{idx}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict[str, Any],
        page_global_chunk_idx: int = 0
    ) -> List[Dict]:
        """
        入口函数：处理单页文本的层级切分。
        职责：初始化状态上下文，并触发递归流水线。
        """
        if not text:
            return []

        # 状态容器 (State Context)
        # 用字典包裹计数器，利用 Python 字典按引用传递的特性，
        # 在递归栈中完美实现状态的全局自增，避免了 global 或 nonlocal 关键字污染。
        state = {
            "level_counters": {level: 0 for level in self.configs.keys()},
            "global_idx": page_global_chunk_idx
        }

        # 启动递归，初始 parent_id 和 root_id 均为空
        return self._recursive_split(
            text=text,
            base_doc=base_doc,
            current_level=1,
            parent_id="",
            root_id="",
            state=state
        )

    def _recursive_split(
        self,
        text: str,
        base_doc: Dict[str, Any],
        current_level: int,
        parent_id: str,
        root_id: str,
        state: Dict[str, Any]
    ) -> List[Dict]:
        """
        核心调度层：利用递归实现“剥洋葱”式的 N 层父子块切分。
        """
        # 1. 递归终止条件：当前层级不在配置中（比如超过3层），或文本为空
        if current_level not in self.splitters or not text:
            return []

        all_chunks: List[Dict] = []
        filename = base_doc.get("filename", "unknown")
        page_number = int(base_doc.get("page_number") or 0)

        # 2. 从字典中取出当前层级的执行器并切分
        splitter = self.splitters[current_level]
        docs = splitter.create_documents([text], [base_doc])

        # 3. 遍历当前层切出的所有子块
        for doc in docs:
            chunk_text = (doc.page_content or "").strip()
            if not chunk_text:
                continue

            # 获取当前层级的序列号
            current_counter = state["level_counters"][current_level]
            # 递增当前层级的序列号
            state["level_counters"][current_level] += 1

            # 4. 生成当前块的唯一 ID
            current_id = self._build_chunk_id(filename, page_number, current_level, current_counter)
            
            # 确立根节点：如果当前没有 root_id（即位于第 1 层），自己就是老祖宗
            actual_root_id = root_id if root_id else current_id

            # 5. 拼装当前块的元数据
            chunk_dict = {
                **base_doc,
                "text": chunk_text,
                "chunk_id": current_id,
                "parent_chunk_id": parent_id,
                "root_chunk_id": actual_root_id,
                "chunk_level": current_level,
                "chunk_idx": state["global_idx"],
            }
            all_chunks.append(chunk_dict)
            
            # 递增全局 Chunk 索引
            state["global_idx"] += 1

            # 6. 核心魔法：带着当前块的上下文，向下一层级发起递归
            #    把自己的 current_id 作为儿子的 parent_id 传下去
            child_chunks = self._recursive_split(
                text=chunk_text,
                base_doc=base_doc,
                current_level=current_level + 1,
                parent_id=current_id,      
                root_id=actual_root_id,    
                state=state
            )
            # 将底层返回的所有子孙块追加到当前列表中
            all_chunks.extend(child_chunks)

        return all_chunks