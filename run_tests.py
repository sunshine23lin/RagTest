# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from agent_main import RagAgentPipeline

doc_path = "D:\\省设计院\\智能体总工程师考试题目-考试文档.pdf"

# 使用 Milvus 查询（不再使用内存搜索）
pipeline = RagAgentPipeline(doc_path=doc_path, use_memory_search=False)

# 删除旧集合（旧向量维度 1024，需要重建为 2048）
print("正在删除旧向量索引...")
pipeline.vector_store.delete_collection()

# 重新构建索引
print("正在重新构建索引...")
chunked_docs = pipeline.load_and_process_document(doc_path)
pipeline.build_indexes(chunked_docs)

pipeline.init_agent()

test_cases = [
    "本工程项目名称是什么？",
    "本项目风机的单机容量有哪些？",
    "本项目的总装机容量是多少？",
    "本项目海域的常浪向是什么？",
    "本工程66kV配电装置的接线方式是什么？",
    "2019年5月、90m高度处的风速是多少？",
    "请检查本文中有哪些上下文不一致或口径不一致的地方，并分项说明原因。"
]

for i, question in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"问题{i}: {question}")
    print(f"{'='*60}")
    
    result = pipeline.agent.ask(question)
    print(f"回答: {result['answer']}")
    print()
