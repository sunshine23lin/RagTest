# BM25 关键词索引
# 支持文档的关键词检索

import os
import sys
import json
import jieba
from rank_bm25 import BM25Okapi
from config import BM25_CONFIG

# 抑制jieba的打印信息
jieba.setLogLevel(40)  # 只显示ERROR级别及以上


class BM25Index:
    """BM25 关键词索引"""

    def __init__(self, index_file=None):
        self.index_file = index_file or BM25_CONFIG["index_file"]
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.metadatas = []

    def build_from_milvus(self, vector_store):
        """从 Milvus 向量数据库构建 BM25 索引
        
        Args:
            vector_store: MilvusVectorStore 实例
        """
        # 从 Milvus 获取所有文本和元数据
        texts, metadatas = vector_store.get_all_texts_and_metadatas()
        
        self.documents = texts
        self.metadatas = metadatas
        self.doc_ids = list(range(len(texts)))
        
        # 中文分词
        tokenized_docs = [list(jieba.cut(doc)) for doc in self.documents]
        
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(tokenized_docs)
        
        return self.bm25

    def build_from_memory(self, texts, metadatas):
        """从内存数据构建 BM25 索引
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        self.documents = texts
        self.metadatas = metadatas
        self.doc_ids = list(range(len(texts)))
        
        # 中文分词
        tokenized_docs = [list(jieba.cut(doc)) for doc in self.documents]
        
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(tokenized_docs)
        
        return self.bm25

    def build_index(self, documents):
        """构建 BM25 索引（兼容旧接口）
        
        Args:
            documents: list of Document objects
        """
        # 提取文本、ID 和元数据
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        
        for doc in documents:
            self.documents.append(doc.page_content)
            self.doc_ids.append(id(doc))
            self.metadatas.append(doc.metadata)
        
        # 中文分词
        tokenized_docs = [list(jieba.cut(doc)) for doc in self.documents]
        
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(tokenized_docs)
        
        return self.bm25

    def search(self, query, top_k=None):
        """关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            list of (doc_index, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 索引未初始化")
        
        top_k = top_k or BM25_CONFIG["top_k"]
        
        # 查询分词
        tokenized_query = list(jieba.cut(query))
        
        # 获取分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 排序
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # 返回 Top-K
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                results.append({
                    "doc_index": idx,
                    "doc_id": self.doc_ids[idx],
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                    "score": float(scores[idx])
                })
        
        return results
