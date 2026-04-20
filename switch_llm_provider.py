# LLM 提供商切换工具
# 用于在不同 LLM 提供商之间快速切换

import os
import sys

def switch_provider(provider_name):
    """切换 LLM 提供商"""
    valid_providers = ["dashscope", "openrouter"]
    
    if provider_name not in valid_providers:
        print(f"错误: 无效的提供商名称 '{provider_name}'")
        print(f"可选值: {valid_providers}")
        sys.exit(1)
    
    # 设置环境变量
    os.environ["LLM_PROVIDER"] = provider_name
    
    # 显示当前配置
    from config import get_llm_config, LLM_PROVIDERS
    
    config = get_llm_config()
    
    print(f"\n✅ 已切换到提供商: {provider_name}")
    print(f"-" * 50)
    print(f"API 地址: {config['base_url']}")
    print(f"主模型: {config['llm_model']}")
    print(f"备用模型: {config.get('fallback_model', '无')}")
    print(f"温度: {config['temperature']}")
    print(f"最大 Token: {config.get('max_tokens', 1024)}")
    print(f"-" * 50)
    
    if provider_name == "openrouter":
        print("\n提示: OpenRouter 不提供 embedding 和 reranker 功能")
        print("需要单独配置 embedding 和 reranker 服务")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python switch_llm_provider.py <提供商名称>")
        print("可选值: dashscope, openrouter")
        print("\n示例:")
        print("  python switch_llm_provider.py dashscope")
        print("  python switch_llm_provider.py openrouter")
        sys.exit(1)
    
    switch_provider(sys.argv[1])
