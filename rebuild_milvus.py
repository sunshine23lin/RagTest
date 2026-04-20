# -*- coding: utf-8 -*-
"""重建 Milvus 集合脚本 - 删除旧集合并重新导入文档"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pymilvus import utility, connections
from config import MILVUS_CONFIG

# 连接 Milvus
connections.connect(
    alias="default",
    host=MILVUS_CONFIG["host"],
    port=MILVUS_CONFIG["port"]
)

# 列出所有集合
print("当前所有集合:")
collections = utility.list_collections()
for c in collections:
    print(f"  - {c}")

# 删除旧的 rag_ 开头的集合
rag_collections = [c for c in collections if c.startswith("rag_")]
if rag_collections:
    print(f"\n找到 {len(rag_collections)} 个 RAG 集合，准备删除...")
    for c in rag_collections:
        print(f"  删除集合: {c}")
        utility.drop_collection(c)
    print("删除完成!")
else:
    print("\n没有找到需要删除的 RAG 集合")

# 确认删除结果
print("\n删除后的集合列表:")
collections = utility.list_collections()
for c in collections:
    print(f"  - {c}")

print("\nMilvus 集合重建完成，请重新运行文档导入流程。")
