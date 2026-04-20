# 主程序入口
# 支持文档加载、切片、向量化和检索

import argparse
import json
from document_loader import DocumentLoader
from chunker import HybridChunker
from embedder import DashscopeEmbedder
from vector_store import MilvusVectorStore


class RagPipeline:
    """RAG 管道：文档加载 -> 切片 -> 向量化 -> 存储 -> 检索"""

    def __init__(self, collection_name=None):
        self.loader = DocumentLoader()
        self.chunker = HybridChunker()
        self.embedder = DashscopeEmbedder()
        self.vector_store = MilvusVectorStore(collection_name=collection_name)

    def load_and_process_document(self, doc_path):
        """加载文档并进行切片处理"""
        print(f"\n=== 加载文档: {doc_path} ===")

        # 加载文档
        documents = self.loader.load_document(doc_path)
        print(f"文档加载完成，共 {len(documents)} 个文档块")

        # 混合切片
        print("正在进行混合数据切分...")
        chunked_docs = self.chunker.chunk_documents(documents)
        print(f"切片完成，共 {len(chunked_docs)} 个 chunk")

        return chunked_docs

    def build_vector_index(self, chunked_docs):
        """构建向量索引"""
        print("\n=== 构建向量索引 ===")

        # 提取文本
        texts = [doc.page_content for doc in chunked_docs]
        metadatas = [doc.metadata for doc in chunked_docs]

        # 批量嵌入
        print("正在生成向量嵌入...")
        embeddings = self.embedder.embed_documents(texts)
        print(f"向量嵌入完成，共 {len(embeddings)} 个向量")

        # 创建集合并插入数据
        self.vector_store.create_collection()
        self.vector_store.insert(embeddings, texts, metadatas)

        # 打印统计信息
        stats = self.vector_store.get_collection_stats()
        print(f"\n集合统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    def search(self, query, top_k=5):
        """执行向量检索"""
        print(f"\n=== 检索查询: {query} ===")

        # 嵌入查询
        query_embedding = self.embedder.embed_query(query)

        # 向量搜索
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # 打印结果
        print(f"\n找到 {len(results)} 条相关结果:\n")
        for i, result in enumerate(results, 1):
            print(f"--- 结果 {i} (相似度: {result['distance']:.4f}) ---")
            print(f"文本: {result['text'][:200]}...")
            print(f"元数据: {json.dumps(result['metadata'], ensure_ascii=False)}")
            print()

        return results

    def clear_index(self):
        """清空向量索引"""
        self.vector_store.delete_collection()
        print("向量索引已清空")


def main():
    parser = argparse.ArgumentParser(description="RAG 文档智能分析系统")
    parser.add_argument("--doc_path", type=str, help="文档路径")
    parser.add_argument("--query", type=str, help="检索查询")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--collection", type=str, default=None, help="集合名称")
    parser.add_argument("--clear", action="store_true", help="清空向量索引")

    args = parser.parse_args()

    pipeline = RagPipeline(collection_name=args.collection)

    if args.clear:
        pipeline.clear_index()
        return

    if args.doc_path:
        # 加载文档并构建索引
        chunked_docs = pipeline.load_and_process_document(args.doc_path)
        pipeline.build_vector_index(chunked_docs)

    if args.query:
        # 执行检索
        pipeline.search(args.query, top_k=args.top_k)
    elif not args.doc_path:
        # 交互式查询
        print("\n文档索引已准备就绪，请输入查询（输入 'exit' 退出）:")
        while True:
            query = input("\n查询: ").strip()
            if query.lower() == 'exit':
                print("程序退出。")
                break
            if query:
                pipeline.search(query, top_k=args.top_k)


if __name__ == "__main__":
    main()
