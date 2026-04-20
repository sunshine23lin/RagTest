# Milvus 向量数据库封装
# 支持本地 Milvus 实例

import json
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from config import MILVUS_CONFIG


class MilvusVectorStore:
    """Milvus 向量数据库操作封装"""

    def __init__(self, host=None, port=None, collection_name=None):
        self.host = host or MILVUS_CONFIG["host"]
        self.port = port or MILVUS_CONFIG["port"]
        self.collection_name = collection_name or MILVUS_CONFIG["collection_name"]
        self.dimension = MILVUS_CONFIG["dimension"]
        self.collection = None
        
        # 内存缓存
        self.in_memory_embeddings = None
        self.in_memory_texts = None
        self.in_memory_metadatas = None
        self.loaded_to_memory = False

        # 连接 Milvus
        self._connect()

    def _connect(self):
        """连接到 Milvus 服务器"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
        except Exception as e:
            print(f"连接 Milvus 失败: {e}")
            raise

    def create_collection(self, dimension=None):
        """创建集合"""
        dim = dimension or self.dimension

        # 检查集合是否已存在
        if utility.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已存在，直接加载")
            self.collection = Collection(self.collection_name)
            return self.collection

        # 定义字段 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]

        schema = CollectionSchema(fields, description="RAG 文档向量集合")

        # 创建集合
        self.collection = Collection(name=self.collection_name, schema=schema)

        # 创建索引
        self._create_index()

        return self.collection

    def _create_index(self):
        """为向量字段创建索引"""
        index_params = MILVUS_CONFIG["index_params"]

        # 检查索引是否已存在
        existing_indexes = self.collection.indexes
        if len(existing_indexes) > 0:
            print("索引已存在，跳过创建")
            return

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

    def insert(self, embeddings, texts, metadatas):
        """插入向量数据"""
        if not self.collection:
            self.create_collection()

        # 确保集合已加载到内存
        self.collection.load()

        # 构建插入数据（顺序必须与 schema 定义一致，排除 auto_id 字段）
        # schema 顺序: embedding, text, metadata
        entities = [
            embeddings,
            texts,
            [json.dumps(m) for m in metadatas]
        ]

        # 插入数据
        result = self.collection.insert(entities)

        # 刷新数据
        self.collection.flush()

        return result

    def search(self, query_embedding, top_k=5, filter_expr=None):
        """向量相似度搜索"""
        if not self.collection:
            # 尝试加载已存在的集合
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                raise ValueError("集合未初始化，请先创建或加载集合")

        # 确保集合已加载
        self.collection.load()

        search_params = MILVUS_CONFIG["search_params"]

        # 执行搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["text", "metadata"]
        )

        # 解析结果
        parsed_results = []
        for hits in results:
            for hit in hits:
                parsed_results.append({
                    "text": hit.entity.get("text"),
                    "metadata": json.loads(hit.entity.get("metadata")),
                    "distance": hit.distance,
                    "id": hit.id
                })

        return parsed_results

    def load_to_memory(self):
        """将所有数据加载到内存"""
        if self.loaded_to_memory:
            return
        
        texts, metadatas = self.get_all_texts_and_metadatas()
        self.in_memory_texts = texts
        self.in_memory_metadatas = metadatas
        
        # 获取所有向量
        if not self.collection:
            self.collection = Collection(self.collection_name)
        self.collection.load()
        
        results = self.collection.query(
            expr="id >= 0",
            output_fields=["embedding", "text", "metadata"],
            limit=10000
        )
        
        embeddings = []
        texts = []
        metadatas = []
        for result in results:
            embeddings.append(result.get("embedding"))
            texts.append(result.get("text", ""))
            metadatas.append(json.loads(result.get("metadata", "{}")))
        
        self.in_memory_embeddings = np.array(embeddings, dtype=np.float32)
        self.in_memory_texts = texts
        self.in_memory_metadatas = metadatas
        self.loaded_to_memory = True

    def search_in_memory(self, query_embedding, top_k=5):
        """在内存中进行向量搜索"""
        if not self.loaded_to_memory:
            self.load_to_memory()
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # 计算余弦相似度（点积）
        similarities = np.dot(self.in_memory_embeddings, query_vec)
        
        # 获取top_k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.in_memory_texts[idx],
                "metadata": self.in_memory_metadatas[idx],
                "distance": float(similarities[idx]),
                "id": idx
            })
        
        return results

    def get_all_texts_and_metadatas(self):
        """获取所有文本和元数据（用于构建BM25索引）"""
        if not self.collection:
            self.collection = Collection(self.collection_name)

        self.collection.load()
        
        # 查询所有数据（使用一个总是为真的表达式和足够大的limit）
        results = self.collection.query(
            expr="id >= 0",
            output_fields=["text", "metadata"],
            limit=10000  # 足够大的数字以获取所有数据
        )
        
        texts = []
        metadatas = []
        for result in results:
            texts.append(result.get("text", ""))
            metadatas.append(json.loads(result.get("metadata", "{}")))
        
        return texts, metadatas

    def delete_collection(self):
        """删除集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"已删除集合: {self.collection_name}")

    def get_collection_stats(self):
        """获取集合统计信息"""
        if not self.collection:
            self.collection = Collection(self.collection_name)

        self.collection.load()
        stats = {
            "name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "dimension": self.dimension
        }
        return stats

    def close(self):
        """关闭连接"""
        connections.disconnect("default")
        print("已断开 Milvus 连接")
