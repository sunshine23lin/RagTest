# RagTest 配置文件
# 支持多 LLM 提供商切换（阿里百炼、OpenRouter 等）

import os

# ==========================================
# LLM 提供商配置（可切换）
# ==========================================

# 当前使用的 LLM 提供商名称，可选值: "dashscope", "openrouter", "zhipu"
# 可通过环境变量 LLM_PROVIDER 覆盖
ACTIVE_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu")

# Embedding 和 Reranker 跟随大模型切换配置
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", ACTIVE_LLM_PROVIDER)

# 阿里百炼配置
LLM_PROVIDERS = {
    "dashscope": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", "111111111"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_model": "qwen-plus-2025-07-28",
        "fallback_model": "qwen3.5-plus",
        "embedding_model": "text-embedding-v3",
        "embedding_dimension": 1024,
        "reranker_model": "qwen-plus",
        "temperature": 0.1,
        "max_tokens": 1024
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY", "1111111111"),
        "base_url": "https://openrouter.ai/api/v1",
        "llm_model": os.getenv("OPENROUTER_MODEL", "qwen/qwen-plus"),
        "fallback_model": os.getenv("OPENROUTER_FALLBACK_MODEL", "qwen/qwen3.5-plus"),
        "embedding_model": None,  # OpenRouter 不支持 Embedding
        "embedding_dimension": None,
        "temperature": 0.1,
        "max_tokens": 1024
    },
    "zhipu": {
        "api_key": os.getenv("ZHIPU_API_KEY", "1111111"),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "llm_model": os.getenv("ZHIPU_LLM_MODEL", "glm-4-plus"),
        "fallback_model": os.getenv("ZHIPU_FALLBACK_MODEL", "glm-5"),
        "embedding_model": os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3"),
        "embedding_dimension": 2048,  # embedding-3 的向量维度
        "temperature": 0.1,
        "max_tokens": 1024
    }
}

# 获取指定提供商的配置
def get_llm_config(provider=None):
    """获取指定 LLM 提供商配置，默认使用当前激活的提供商"""
    if provider is None:
        provider = ACTIVE_LLM_PROVIDER
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"未知的 LLM 提供商: {provider}，可选值: {list(LLM_PROVIDERS.keys())}")
    return LLM_PROVIDERS[provider]

# 获取 Embedding 提供商配置（跟随大模型切换）
def get_embedding_config():
    """获取 Embedding 提供商配置"""
    provider = EMBEDDING_PROVIDER
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"未知的 Embedding 提供商: {provider}，可选值: {list(LLM_PROVIDERS.keys())}")
    config = LLM_PROVIDERS[provider]
    if config.get("embedding_model") is None:
        raise ValueError(f"提供商 {provider} 不支持 Embedding，请切换 EMBEDDING_PROVIDER 环境变量")
    return config

# 兼容旧代码的别名
DASHSCOPE_CONFIG = get_llm_config()  # LLM 配置（当前激活的提供商）
EMBEDDING_CONFIG = get_embedding_config()  # Embedding 配置（始终使用阿里百炼）

# ==========================================
# Milvus 配置
# ==========================================

# Milvus 配置
MILVUS_CONFIG = {
    "host": "localhost",
    "port": "19530",
    "collection_name": "rag_test_collection",
    "dimension": 2048,  # embedding-3 的向量维度
    "index_params": {
        "metric_type": "IP",  # 内积相似度
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    },
    "search_params": {
        "metric_type": "IP",
        "params": {"ef": 64}
    }
}

# BM25 配置
BM25_CONFIG = {
    "index_file": "bm25_index.json",
    "top_k": 20
}

# RRF 配置
RRF_CONFIG = {
    "k": 60,  # RRF 常数，通常取 60
    "bm25_weight": 2.0,  # 提高BM25权重，关键词匹配更精确
    "vector_weight": 1.0
}

# 重排配置
RERANKER_CONFIG = {
    "top_k": 20,  # 重排候选数
    "final_top_k": 8  # 最终返回数（增加召回率，确保命中关键数据）
}

# 文档切片配置
CHUNK_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "min_chunk_size": 100,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", ""]
}

# 文档解析配置
DOCUMENT_CONFIG = {
    "supported_extensions": [".pdf", ".docx"],
    "table_format": "markdown",
    "extract_images": False
}
