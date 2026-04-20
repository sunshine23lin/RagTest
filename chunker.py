# 混合数据切分器
# 基于语义+结构的智能切片策略

import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_CONFIG


class HybridChunker:
    """混合数据切分器：基于语义+结构"""

    def __init__(self):
        self.chunk_size = CHUNK_CONFIG["chunk_size"]
        self.chunk_overlap = CHUNK_CONFIG["chunk_overlap"]
        self.min_chunk_size = CHUNK_CONFIG["min_chunk_size"]
        self.separators = CHUNK_CONFIG["separators"]

        # 初始化递归字符切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            keep_separator=True,
            add_start_index=True,
        )

    def chunk_documents(self, documents):
        """对文档列表进行混合切分"""
        chunked_docs = []

        for doc in documents:
            doc_type = doc.metadata.get("type", "text")

            if doc_type == "table":
                # 表格单独作为一个 chunk，不切分
                chunked_docs.append(doc)
            else:
                # 文本进行递归字符切分
                # 但先检查是否包含表格标记（Markdown 表格）
                text = doc.page_content
                if '|' in text and '---' in text:
                    # 包含表格标记，按表格分割
                    parts = self._split_text_with_tables(text)
                    for part in parts:
                        if part.strip():
                            if '|' in part and '---' in part:
                                # 表格部分，不切分
                                chunked_docs.append(Document(
                                    page_content=part,
                                    metadata=doc.metadata.copy()
                                ))
                            else:
                                # 普通文本，进行切分
                                temp_doc = Document(
                                    page_content=part,
                                    metadata=doc.metadata.copy()
                                )
                                chunks = self.text_splitter.split_documents([temp_doc])
                                chunked_docs.extend(chunks)
                else:
                    # 普通文本，进行递归字符切分
                    chunks = self.text_splitter.split_documents([doc])
                    chunked_docs.extend(chunks)

        # 过滤过小的 chunk
        chunked_docs = [
            doc for doc in chunked_docs
            if len(doc.page_content.strip()) >= self.min_chunk_size
        ]

        return chunked_docs

    def _split_text_with_tables(self, text):
        """将包含表格的文本分割为表格和非表格部分"""
        lines = text.split('\n')
        parts = []
        current_part = []
        in_table = False
        
        for line in lines:
            is_table_line = '|' in line and ('---' in line or (current_part and '|' in current_part[-1]))
            
            if is_table_line:
                if not in_table and current_part:
                    parts.append('\n'.join(current_part))
                    current_part = []
                in_table = True
                current_part.append(line)
            else:
                if in_table and current_part:
                    parts.append('\n'.join(current_part))
                    current_part = []
                in_table = False
                current_part.append(line)
        
        if current_part:
            parts.append('\n'.join(current_part))
        
        return parts

    def chunk_text_with_structure(self, text, metadata=None):
        """对纯文本进行结构化切分（识别标题层级）"""
        if metadata is None:
            metadata = {}

        # 识别标题行（如 "1. 工程概况"、"## 项目基本情况"）
        lines = text.split("\n")
        sections = []
        current_section = {"title": "", "content": []}

        for line in lines:
            # 匹配标题模式：数字开头 或 # 开头
            title_match = re.match(r'^(#{1,6}\s*|\d+[\.、]\s*)', line.strip())
            if title_match and line.strip():
                # 保存上一个 section
                if current_section["content"]:
                    sections.append(current_section)
                # 开始新 section
                current_section = {
                    "title": line.strip(),
                    "content": []
                }
            else:
                current_section["content"].append(line)

        # 保存最后一个 section
        if current_section["content"]:
            sections.append(current_section)

        # 对每个 section 进行切分
        chunked_docs = []
        for section in sections:
            section_text = "\n".join(section["content"]).strip()
            if not section_text:
                continue

            # 构建元数据
            section_metadata = metadata.copy()
            section_metadata["section_title"] = section["title"]
            section_metadata["type"] = "text"

            # 如果 section 内容小于 chunk_size，直接作为一个 chunk
            if len(section_text) <= self.chunk_size:
                chunked_docs.append(Document(
                    page_content=section_text,
                    metadata=section_metadata
                ))
            else:
                # 否则进行递归切分
                doc = Document(
                    page_content=section_text,
                    metadata=section_metadata
                )
                chunks = self.text_splitter.split_documents([doc])
                chunked_docs.extend(chunks)

        return chunked_docs

    def chunk_with_overlap_context(self, documents, context_lines=3):
        """带上下文重叠的切分（保持语义连贯性）"""
        chunked_docs = []

        for doc in documents:
            doc_type = doc.metadata.get("type", "text")

            if doc_type == "table":
                chunked_docs.append(doc)
                continue

            lines = doc.page_content.split("\n")
            chunks = []
            current_chunk = []
            current_length = 0

            for i, line in enumerate(lines):
                current_chunk.append(line)
                current_length += len(line)

                if current_length >= self.chunk_size:
                    # 添加上下文行
                    start_idx = max(0, i - context_lines)
                    context = "\n".join(lines[start_idx:i + 1])

                    chunked_docs.append(Document(
                        page_content=context,
                        metadata=doc.metadata.copy()
                    ))

                    # 保留重叠部分
                    overlap_lines = current_chunk[-(context_lines + 1):]
                    current_chunk = overlap_lines
                    current_length = sum(len(l) for l in overlap_lines)

            # 处理最后一个 chunk
            if current_chunk:
                chunked_docs.append(Document(
                    page_content="\n".join(current_chunk),
                    metadata=doc.metadata.copy()
                ))

        return chunked_docs
