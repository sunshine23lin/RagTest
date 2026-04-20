# 智谱 Reranker 重排器
# 使用 GLM-4-Plus 模型进行结果重排

from openai import OpenAI
from config import DASHSCOPE_CONFIG, RERANKER_CONFIG


class DashscopeReranker:
    """智谱 Reranker 重排器"""

    def __init__(self, model_name=None, api_key=None, base_url=None):
        self.model_name = model_name or DASHSCOPE_CONFIG["llm_model"]
        self.api_key = api_key or DASHSCOPE_CONFIG["api_key"]
        self.base_url = base_url or DASHSCOPE_CONFIG["base_url"]
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def rerank(self, query, documents, top_k=None):
        """对文档列表进行重排"""
        if not documents:
            return []

        top_k = top_k or RERANKER_CONFIG["final_top_k"]

        print(f"  - Reranker 重排 ({len(documents)} 个候选)...")

        scored_docs = []
        for doc in documents:
            text = doc.get("text", "")
            if text:
                score = self._compute_relevance(query, text)
                scored_docs.append({
                    **doc,
                    "rerank_score": score
                })

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        results = scored_docs[:top_k]
        print(f"    重排完成，返回 {len(results)} 条结果")

        return results

    def _compute_relevance(self, query, doc):
        """计算 query-doc 相关性分数"""
        try:
            prompt = f"""请评估以下查询和文档的相关性，只返回 0-1 之间的分数：

查询: {query}

文档: {doc[:500]}

相关性分数 (0-1):"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本相关性评估专家。请只返回 0-1 之间的数字分数。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                import re
                numbers = re.findall(r'[\d.]+', score_text)
                if numbers:
                    return max(0.0, min(1.0, float(numbers[0])))
                return 0.5

        except Exception as e:
            print(f"    Reranker 计算失败: {e}")
            return 0.5

    def batch_rerank(self, query, documents, batch_size=5):
        """批量重排（优化性能）"""
        if not documents:
            return []

        print(f"  - Reranker 批量重排 ({len(documents)} 个候选, 批次大小={batch_size})...")

        scored_docs = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            for doc in batch:
                text = doc.get("text", "")
                if text:
                    score = self._compute_relevance(query, text)
                    scored_docs.append({
                        **doc,
                        "rerank_score": score
                    })

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        print(f"    批量重排完成")

        return scored_docs
