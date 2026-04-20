# 智能体主入口
# 支持文档加载、索引构建、批量问题回答

import argparse
import json
import os
from document_loader import DocumentLoader
from chunker import HybridChunker
from embedder import DashscopeEmbedder
from vector_store import MilvusVectorStore
from bm25_index import BM25Index
from hybrid_retriever import RRFHybridRetriever
from reranker import DashscopeReranker
from agent_workflow import DocumentAnalysisAgent


class RagAgentPipeline:
    """RAG 智能体管道"""

    def __init__(self, doc_path=None, use_memory_search=False):
        self.loader = DocumentLoader()
        self.chunker = HybridChunker()
        self.embedder = DashscopeEmbedder()
        
        # 根据文档路径生成集合名称
        self.doc_path = doc_path
        collection_name = self._get_collection_name(doc_path)
        
        self.vector_store = MilvusVectorStore(collection_name=collection_name)
        self.bm25_index = BM25Index()
        self.hybrid_retriever = RRFHybridRetriever()
        self.reranker = DashscopeReranker()
        self.agent = None
        self.use_memory_search = use_memory_search

    def _get_collection_name(self, doc_path):
        """根据文档路径生成集合名称（只包含字母、数字和下划线）"""
        if doc_path:
            import hashlib
            # 使用文档路径的MD5作为集合名称，确保只包含字母和数字
            doc_hash = hashlib.md5(doc_path.encode('utf-8')).hexdigest()
            return f"rag_{doc_hash}"
        return "rag_default"

    def is_collection_exists(self):
        """检查集合是否存在"""
        from pymilvus import utility
        return utility.has_collection(self.vector_store.collection_name)

    def load_and_process_document(self, doc_path):
        """加载文档并进行切片处理"""
        self.doc_path = doc_path

        documents = self.loader.load_document(doc_path)

        chunked_docs = self.chunker.chunk_documents(documents)

        return chunked_docs

    def build_indexes(self, chunked_docs):
        """构建所有索引（向量索引 + BM25 索引）"""
        texts = [doc.page_content for doc in chunked_docs]
        metadatas = [doc.metadata for doc in chunked_docs]

        embeddings = self.embedder.embed_documents(texts)

        self.vector_store.create_collection()
        self.vector_store.insert(embeddings, texts, metadatas)

        self.bm25_index.build_from_milvus(self.vector_store)

    def load_to_memory(self):
        """加载数据到内存"""
        if self.use_memory_search:
            self.vector_store.load_to_memory()
            self.bm25_index.build_from_memory(
                self.vector_store.in_memory_texts,
                self.vector_store.in_memory_metadatas
            )

    def init_agent(self):
        """初始化智能体"""
        if not self.bm25_index.bm25:
            if self.use_memory_search:
                self.bm25_index.build_from_memory(
                    self.vector_store.in_memory_texts,
                    self.vector_store.in_memory_metadatas
                )
            else:
                self.bm25_index.build_from_milvus(self.vector_store)
        
        self.agent = DocumentAnalysisAgent(
            bm25_index=self.bm25_index,
            vector_store=self.vector_store,
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            embedder=self.embedder,
            use_memory_search=self.use_memory_search
        )

    def ask(self, question: str) -> dict:
        """提问"""
        if not self.agent:
            self.init_agent()
        return self.agent.ask(question)

    def ask_batch(self, questions: list) -> list:
        """批量提问"""
        if not self.agent:
            self.init_agent()
        return self.agent.ask_batch(questions)


def print_results(results):
    """打印结果"""
    print("\n" + "="*80)
    print("回答结果汇总")
    print("="*80)

    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"问题 {i}: {result['question']}")
        print(f"{'─'*80}")
        print(f"问题类型: {result.get('question_type', 'unknown')}")
        print(f"\n回答:\n{result['answer']}")
        print(f"\n验证信息: {result.get('confidence', 'N/A')}")
        print(f"参考文档片段数: {result.get('context_sources', 0)}")


def main():
    parser = argparse.ArgumentParser(description="RAG 文档智能分析系统")
    parser.add_argument("--doc_path", type=str, help="文档路径")
    parser.add_argument("--query", type=str, help="单个查询问题")
    parser.add_argument("--batch", action="store_true", help="批量问题模式")
    parser.add_argument("--questions_file", type=str, help="问题文件路径（JSON 格式）")
    parser.add_argument("--clear", action="store_true", help="清空当前文档的向量索引")
    parser.add_argument("--rebuild", action="store_true", help="重新构建当前文档的索引")
    parser.add_argument("--memory", action="store_true", default=True, help="使用内存搜索模式（默认启用）")
    parser.add_argument("--no-memory", action="store_true", help="禁用内存搜索模式，使用Milvus查询")

    args = parser.parse_args()

    use_memory = not args.no_memory
    pipeline = RagAgentPipeline(doc_path=args.doc_path, use_memory_search=use_memory)

    if args.clear:
        pipeline.vector_store.delete_collection()
        return

    if args.doc_path:
        if args.rebuild or not pipeline.is_collection_exists():
            chunked_docs = pipeline.load_and_process_document(args.doc_path)
            pipeline.build_indexes(chunked_docs)
        
        if args.memory:
            pipeline.load_to_memory()

    pipeline.init_agent()

    if args.query:
        result = pipeline.ask(args.query)
        print(f"回答: {result['answer']}")

    elif args.batch or args.questions_file:
        if args.questions_file:
            with open(args.questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        else:
            questions = [
                "本工程项目名称是什么？",
                "本项目风机的单机容量有哪些？",
                "本项目的总装机容量是多少？",
                "本项目海域的常浪向是什么？",
                "本工程66kV配电装置的接线方式是什么？",
                "2019年5月、90m高度处的风速是多少？",
                "请检查本文中有哪些上下文不一致或口径不一致的地方，并分项说明原因。"
            ]

        results = pipeline.ask_batch(questions)
        print_results(results)

        results_file = "agent_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    else:
        print("RAG 文档智能分析系统已就绪")
        print("输入问题开始对话，输入 'exit' 退出")
        
        while True:
            query = input("\n问题: ").strip()
            if query.lower() == 'exit':
                print("程序退出。")
                break
            if query:
                result = pipeline.ask(query)
                print(f"回答: {result['answer']}")


if __name__ == "__main__":
    main()
