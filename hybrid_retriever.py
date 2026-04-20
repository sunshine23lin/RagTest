# RRF 混合检索器
# 融合 BM25 关键词检索和 Milvus 向量检索结果

import hashlib
from collections import defaultdict
from config import RRF_CONFIG


class RRFHybridRetriever:
    """RRF (Reciprocal Rank Fusion) 混合检索器"""

    def __init__(self, k=None, bm25_weight=None, vector_weight=None):
        self.k = k or RRF_CONFIG["k"]
        self.bm25_weight = bm25_weight or RRF_CONFIG["bm25_weight"]
        self.vector_weight = vector_weight or RRF_CONFIG["vector_weight"]

    def _get_doc_key(self, result):
        """生成文档唯一键，用于跨检索源去重"""
        # 优先使用元数据（page + type + source）作为唯一键
        metadata = result.get("metadata", {})
        if metadata and metadata.get("page") and metadata.get("type"):
            page = metadata.get("page", "")
            doc_type = metadata.get("type", "")
            source = metadata.get("source", "")
            return f"page{page}_{doc_type}_{source}"
        
        # 回退：使用文本内容的前100字符作为键
        text = result.get("text", "")
        if text:
            return "text_" + hashlib.md5(text[:100].encode('utf-8')).hexdigest()
        
        # 最后回退：使用 doc_id 或 id
        return result.get("doc_id") or result.get("id") or ""

    def rrf_fusion(self, bm25_results, vector_results):
        """RRF 融合两路检索结果
        
        Args:
            bm25_results: BM25 检索结果列表，每个元素包含 {"doc_id": ..., "text": ..., "score": ...}
            vector_results: 向量检索结果列表，每个元素包含 {"doc_id": ..., "text": ..., "distance": ...}
            
        Returns:
            融合后的结果列表，按 RRF 分数降序排列
        """
        # 存储每个文档的 RRF 分数
        rrf_scores = defaultdict(float)
        doc_map = {}

        # BM25 结果
        for rank, result in enumerate(bm25_results, 1):
            doc_key = self._get_doc_key(result)
            if doc_key:
                score = self.bm25_weight / (self.k + rank)
                rrf_scores[doc_key] += score
                if doc_key not in doc_map:
                    doc_map[doc_key] = result

        # 向量检索结果
        for rank, result in enumerate(vector_results, 1):
            doc_key = self._get_doc_key(result)
            if doc_key:
                score = self.vector_weight / (self.k + rank)
                rrf_scores[doc_key] += score
                if doc_key not in doc_map:
                    doc_map[doc_key] = result

        # 按 RRF 分数排序
        ranked_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 构建结果
        fused_results = []
        for doc_key in ranked_docs:
            result = doc_map[doc_key].copy()
            result["rrf_score"] = rrf_scores[doc_key]
            fused_results.append(result)

        return fused_results

    def search(self, bm25_search_fn, vector_search_fn, query, top_k=10):
        """执行混合检索
        
        Args:
            bm25_search_fn: BM25 搜索函数，接受 query 和 top_k 参数
            vector_search_fn: 向量搜索函数，接受 query 和 top_k 参数
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            融合后的 Top-K 结果
        """
        # 并行执行两路检索
        print(f"  - BM25 关键词检索...")
        bm25_results = bm25_search_fn(query, top_k=top_k)
        print(f"    找到 {len(bm25_results)} 条结果")

        print(f"  - 向量检索...")
        vector_results = vector_search_fn(query, top_k=top_k)
        print(f"    找到 {len(vector_results)} 条结果")

        # RRF 融合
        print(f"  - RRF 融合...")
        fused_results = self.rrf_fusion(bm25_results, vector_results)
        print(f"    融合后 {len(fused_results)} 条结果")

        # 返回 Top-K
        return fused_results[:top_k]
