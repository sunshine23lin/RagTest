# -*- coding: utf-8 -*-
"""重新导入文档到 Milvus（使用新的 HNSW 索引）"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from agent_main import RagAgentPipeline

doc_path = "D:\\省设计院\\智能体总工程师考试题目-考试文档.pdf"

print(f"正在导入文档: {doc_path}")
print("使用 HNSW 索引...")

pipeline = RagAgentPipeline(doc_path=doc_path, use_memory_search=True)

# 加载并处理文档
print("\n1. 加载文档并切片...")
chunked_docs = pipeline.load_and_process_document(doc_path)
print(f"   共生成 {len(chunked_docs)} 个切片")

# 构建索引
print("\n2. 构建索引（向量索引 + BM25 索引）...")
pipeline.build_indexes(chunked_docs)
print("   索引构建完成")

# 加载到内存
print("\n3. 加载数据到内存...")
pipeline.load_to_memory()
print("   加载完成")

print("\n文档导入完成！现在可以开始提问。")
