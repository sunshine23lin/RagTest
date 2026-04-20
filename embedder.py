# 嵌入模型集成
# 支持多提供商：阿里百炼、智普等

from openai import OpenAI
from config import EMBEDDING_CONFIG


class Embedder:
    """嵌入模型封装（支持多提供商）"""

    def __init__(self, model_name=None, api_key=None, base_url=None):
        self.model_name = model_name or EMBEDDING_CONFIG["embedding_model"]
        self.api_key = api_key or EMBEDDING_CONFIG["api_key"]
        self.base_url = base_url or EMBEDDING_CONFIG["base_url"]
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def embed_documents(self, texts, batch_size=10):
        """批量嵌入文档（分批处理，每批最多 10 条）"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            batch_embeddings = [emb.embedding for emb in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_query(self, text):
        """嵌入单个查询"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        return response.data[0].embedding

    def get_dimension(self):
        """获取向量维度"""
        return EMBEDDING_CONFIG.get("embedding_dimension", 1024)


# 兼容旧代码的别名
DashscopeEmbedder = Embedder
