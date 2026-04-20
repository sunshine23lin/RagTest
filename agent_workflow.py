# LangGraph 智能体工作流
# 文档理解、检索、抽取与综合分析
# 
# 核心流程：
# 1. BM25关键词检索 + 向量语义检索（双路召回）
# 2. RRF融合排序（Reciprocal Rank Fusion）
# 3. 上下文构建（智能处理表格数据，避免重复/破损数据干扰）
# 4. LLM答案生成
# 5. 后处理验证（修正信息来源格式）

import time
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from openai import OpenAI
from config import DASHSCOPE_CONFIG, RERANKER_CONFIG, RRF_CONFIG, ACTIVE_LLM_PROVIDER


class AgentState(TypedDict):
    """
    智能体状态定义
    用于LangGraph工作流中各节点之间的数据传递
    """
    question: str              # 用户问题
    question_type: str         # 问题类型
    bm25_results: List[dict]   # BM25关键词检索结果
    vector_results: List[dict] # 向量语义检索结果
    fused_results: List[dict]  # RRF融合后的结果
    reranked_results: List[dict] # 重排后的最终检索结果
    context: str               # 构建的上下文内容
    answer: str                # LLM生成的答案
    reasoning: str             # 推理过程
    confidence: float          # 置信度


class DocumentAnalysisAgent:
    """
    文档分析智能体工作流
    
    基于LangGraph构建的文档问答系统，支持：
    - 双路检索（BM25关键词 + 向量语义）
    - RRF融合排序
    - 智能上下文构建（表格转JSON、去重破损数据）
    - 全局分析类问题处理（全文压缩策略）
    - 自动检测项目名称不一致
    """

    def __init__(self, bm25_index, vector_store, hybrid_retriever, reranker, embedder=None, use_memory_search=False):
        """
        初始化智能体
        
        Args:
            bm25_index: BM25关键词索引
            vector_store: 向量数据库
            hybrid_retriever: 混合检索器（RRF融合）
            reranker: 重排器
            embedder: 嵌入模型（用于向量检索）
            use_memory_search: 是否使用内存搜索
        """
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.embedder = embedder
        self.use_memory_search = use_memory_search

        # 从配置中动态获取当前激活的 LLM 提供商配置
        self.api_key = DASHSCOPE_CONFIG["api_key"]
        self.base_url = DASHSCOPE_CONFIG["base_url"]
        self.llm_model = DASHSCOPE_CONFIG["llm_model"]
        self.fallback_model = DASHSCOPE_CONFIG.get("fallback_model", self.llm_model)
        self.temperature = DASHSCOPE_CONFIG["temperature"]
        self.max_tokens = DASHSCOPE_CONFIG.get("max_tokens", 1024)
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # 提取文档大标题（只提取一次，后续所有问题共用）
        self.doc_title = self._extract_doc_title()

        # 构建并编译工作流
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _extract_doc_title(self) -> str:
        """
        提取文档大标题（只提取一次，后续所有问题共用）
        
        提取策略：
        1. 优先从《xxx》格式提取（如《某海域500MW海上风电场工程初步设计说明书》）
        2. 备选：从非表格文本的前5行中提取长度>10且不含表格标记的行
        
        Returns:
            str: 文档大标题，清理了换行符和多余空格
        """
        import re
        
        # 从BM25索引获取所有文档
        docs = self.bm25_index.documents
        metadatas = self.bm25_index.metadatas
        
        # 策略1：优先从《xxx》格式提取
        for doc_text in docs:
            title_matches = re.findall(r'《([^》]+)》', doc_text)
            if title_matches:
                # 清理标题中的换行符和多余空格
                title = title_matches[0].replace('\n', '').replace('\r', '')
                title = re.sub(r'\s+', '', title)  # 移除所有空白字符
                return title
        
        # 策略2：从非表格类型的文本中提取第一行作为标题
        for doc_text, meta in zip(docs, metadatas):
            if meta.get("type") != "table":
                lines = doc_text.strip().split('\n')
                for line in lines[:5]:
                    line = line.strip()
                    if not line or re.match(r'^\d+$', line):
                        continue
                    # 只取长度较长且不含表格标记的行作为大标题
                    if len(line) > 10 and '|' not in line and not line.startswith('#'):
                        return line.replace('\n', '').replace('\r', '')
        
        return "文档"

    def _call_llm(self, system_prompt: str, user_prompt: str, use_fallback: bool = False) -> str:
        """
        调用大模型，支持备用模型切换（适配多提供商）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            use_fallback: 是否使用备用模型
            
        Returns:
            str: LLM生成的回答
        """
        model = self.fallback_model if use_fallback else self.llm_model
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _call_llm_with_fallback(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用大模型，优先使用主模型，失败时切换到备用模型
        
        容错机制：
        1. 先尝试主模型
        2. 检查回答质量（非空、非乱码、非错误提示）
        3. 如果主模型失败或回答质量差，切换到备用模型
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            str: LLM生成的回答
        """
        # 先尝试使用主模型
        try:
            answer = self._call_llm(system_prompt, user_prompt, use_fallback=False)
            # 检查回答质量
            if self._is_valid_answer(answer):
                return answer
        except Exception:
            pass
        
        # 切换到备用模型（qwen3.5-plus）
        return self._call_llm(system_prompt, user_prompt, use_fallback=True)

    def _is_valid_answer(self, answer: str) -> bool:
        """
        检查回答是否有效
        
        判断标准：
        1. 非空
        2. 不包含过多问号（乱码特征）
        3. 不包含明显的错误提示
        
        Args:
            answer: LLM生成的回答
            
        Returns:
            bool: 回答是否有效
        """
        if not answer or not answer.strip():
            return False
        # 检查是否包含乱码或问号
        if answer.count('?') > 5 or answer.count('？') > 5:
            return False
        # 检查是否包含明显的错误提示
        error_keywords = ["无法回答", "无法确定", "不知道", "未明确", "请补充"]
        if any(kw in answer for kw in error_keywords):
            return False
        return True

    def _build_workflow(self) -> StateGraph:
        """
        构建 LangGraph 工作流
        
        工作流节点顺序：
        1. retrieve_bm25: BM25关键词检索
        2. retrieve_vector: 向量语义检索
        3. rrf_fusion: RRF融合排序
        4. rerank: 重排（取Top-K）
        5. build_context: 构建上下文（智能处理表格）
        6. generate_answer: LLM生成答案
        7. verify_and_correct: 后处理验证（修正信息来源格式）
        
        Returns:
            StateGraph: 编译后的工作流
        """
        workflow = StateGraph(AgentState)

        # 添加工作流节点
        workflow.add_node("retrieve_bm25", self._retrieve_bm25)
        workflow.add_node("retrieve_vector", self._retrieve_vector)
        workflow.add_node("rrf_fusion", self._rrf_fusion)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("build_context", self._build_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_and_correct", self._verify_and_correct_answer)

        # 设置入口节点
        workflow.set_entry_point("retrieve_bm25")

        # 连接节点（线性流程）
        workflow.add_edge("retrieve_bm25", "retrieve_vector")
        workflow.add_edge("retrieve_vector", "rrf_fusion")
        workflow.add_edge("rrf_fusion", "rerank")
        workflow.add_edge("rerank", "build_context")
        workflow.add_edge("build_context", "generate_answer")
        workflow.add_edge("generate_answer", "verify_and_correct")
        workflow.add_edge("verify_and_correct", END)

        return workflow

    def _build_full_text_context(self, state: AgentState) -> dict:
        """
        全文压缩策略 - 用于全局分析类问题（如"检查不一致"、"分析差异"）
        
        策略说明：
        1. 提取所有文档，按页面分组
        2. 每页内容压缩（清理空白、限制长度）
        3. 自动检测项目名称不一致
        4. 拼接全文作为上下文
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 包含全文上下文的字典
        """
        import re
        
        # 从BM25索引获取所有文档
        docs = self.bm25_index.documents
        metadatas = self.bm25_index.metadatas
        
        # 使用实例变量中的文档标题
        doc_title = self.doc_title
        
        # 按页面分组
        page_groups = {}
        for doc_text, meta in zip(docs, metadatas):
            page = meta.get("page", 0)
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc_text)
        
        # 压缩每个页面的内容
        context_parts = [f"【文档标题】{doc_title}\n"]
        for page in sorted(page_groups.keys()):
            page_texts = page_groups[page]
            # 合并该页所有文本
            full_text = "\n".join(page_texts)
            # 清理多余空白和换行
            full_text = re.sub(r'\n+', '\n', full_text)
            full_text = re.sub(r' +', ' ', full_text)
            
            # 如果文本过长，截取关键部分（保留前3000字符）
            if len(full_text) > 3000:
                full_text = full_text[:3000] + "..."
            
            context_parts.append(f"【第{page}页】\n{full_text}")
        
        # 自动检测项目名称不一致
        name_inconsistencies = self._detect_name_inconsistencies(page_groups)
        if name_inconsistencies:
            context_parts.insert(1, f"【⚠️ 自动检测到的不一致】\n{name_inconsistencies}\n")
        
        # 拼接全文
        full_context = "\n\n".join(context_parts)
        
        return {"context": full_context}
    
    def _detect_name_inconsistencies(self, page_groups: dict) -> str:
        """
        自动检测项目名称不一致
        
        检测逻辑：
        1. 从每页提取项目名称（支持多种模式）
        2. 对比不同页的项目名称
        3. 通过重复字符检测OCR错误（如"上上"、"的的"）
        4. 通过出现频率判断正确名称
        
        Args:
            page_groups: 按页面分组的文本字典
            
        Returns:
            str: 不一致信息，如果没有不一致则返回空字符串
        """
        import re
        
        page_names = {}
        for page, texts in page_groups.items():
            full_text = " ".join(texts)
            # 提取项目名称（多种模式）
            name_patterns = [
                r'项目名称[为：:：]\s*([^\n。]+)',
                r'工程名称[为：:：]\s*([^\n。]+)',
                r'(某海域.*?风电场.*?工程)',
            ]
            for pattern in name_patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    # 清理项目名称
                    clean_name = matches[0].strip().lstrip('：:').strip()
                    if len(clean_name) > 5:  # 过滤太短的匹配
                        page_names[page] = clean_name
                        break
        
        # 对比不同页的项目名称
        inconsistencies = []
        unique_names = {}
        for page, name in page_names.items():
            # 标准化项目名称（去除空格）
            normalized = name.replace(' ', '')
            if normalized not in unique_names:
                unique_names[normalized] = []
            unique_names[normalized].append((page, name))
        
        # 如果有多个不同的项目名称，说明存在不一致
        if len(unique_names) > 1:
            # 合并全文判断项目类型
            full_text = ""
            for page, texts in page_groups.items():
                full_text += " ".join(texts)
            
            # 判断哪个名称正确
            correct_name = None
            for normalized_name in unique_names.keys():
                # 跳过明显OCR错误（重复字符）
                has_repeat_error = False
                for i in range(len(normalized_name) - 1):
                    if normalized_name[i:i+2] in ['上上', '下下', '中中', '的的', '了了']:
                        has_repeat_error = True
                        break
                if has_repeat_error:
                    continue
                
                # 统计该名称在全文中出现的次数
                name_count = full_text.count(normalized_name)
                if correct_name is None or name_count > full_text.count(correct_name):
                    correct_name = normalized_name
            
            # 如果没找到，用第一个
            if not correct_name:
                correct_name = list(unique_names.keys())[0]
            
            # 输出正确名称
            inconsistencies.append(f"正确项目名称：{correct_name}")
            
            # 输出错误名称
            for normalized_name, occurrences in unique_names.items():
                if normalized_name != correct_name:
                    for page, name in occurrences:
                        # 判断错误类型
                        has_repeat_error = False
                        for i in range(len(name) - 1):
                            if name[i:i+2] in ['上上', '下下', '中中', '的的', '了了']:
                                has_repeat_error = True
                                break
                        if has_repeat_error:
                            error_type = "OCR识别错误"
                        else:
                            error_type = "笔误"
                        inconsistencies.append(f"第{page}页: {name}（{error_type}）")
        
        if inconsistencies:
            return "项目名称在不同页面存在差异：\n" + "\n".join(inconsistencies)
        
        return ""

    def _retrieve_bm25(self, state: AgentState) -> dict:
        """
        BM25关键词检索节点
        
        特殊处理：
        - 全局分析类问题（如"检查不一致"）跳过检索，直接使用全文压缩策略
        - 表格数据加权：涉及数值的问题提高表格类型结果权重50%
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: BM25检索结果
        """
        question = state["question"]
        
        # 检测是否为全局分析类问题（如"检查不一致"、"分析差异"）
        global_analysis_keywords = ["不一致", "口径不一致", "差异", "矛盾", "对比", "检查全文"]
        is_global_query = any(kw in question for kw in global_analysis_keywords)
        
        # 全局分析类问题：直接使用全文压缩策略，不走BM25检索
        if is_global_query:
            return {"bm25_results": []}
        
        top_k = RRF_CONFIG.get("top_k", 20)
        all_results = self.bm25_index.search(question, top_k=top_k)
        
        # 表格数据加权：如果问题涉及数值/高度/风速等表格特征词，提高表格类型结果的权重
        table_keywords = ["风速", "高度", "容量", "功率", "温度", "压力", "流量", "m/s", "MW", "m", "℃", "kPa"]
        if any(kw in question for kw in table_keywords):
            for result in all_results:
                metadata = result.get("metadata", {})
                if metadata.get("type") == "table":
                    result["bm25_score"] = result.get("score", 0) * 1.5  # 表格数据权重提升50%
        
        return {"bm25_results": all_results}

    def _retrieve_vector(self, state: AgentState) -> dict:
        """
        向量语义检索节点
        
        特殊处理：
        - 全局分析类问题跳过检索，直接使用全文压缩策略
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 向量检索结果
        """
        question = state["question"]
        
        # 全局分析类问题：直接使用全文压缩策略，不走向量检索
        global_analysis_keywords = ["不一致", "口径不一致", "差异", "矛盾", "对比", "检查全文"]
        is_global_query = any(kw in question for kw in global_analysis_keywords)
        if is_global_query:
            return {"vector_results": []}
        
        top_k = RRF_CONFIG.get("top_k", 20)
        query_embedding = self.embedder.embed_query(question)
        
        if self.use_memory_search:
            results = self.vector_store.search_in_memory(query_embedding, top_k=top_k)
        else:
            results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return {"vector_results": results}

    def _rrf_fusion(self, state: AgentState) -> dict:
        """
        RRF（Reciprocal Rank Fusion）融合节点
        
        将BM25和向量检索结果融合，综合关键词匹配和语义相似度
        
        相关性检查：
        - 如果BM25无结果且向量检索分数低于0.6，认为问题不相关
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 融合后的结果
        """
        bm25_results = state.get("bm25_results", [])
        vector_results = state.get("vector_results", [])
        question = state.get("question", "")
        
        # 相关性检查：如果BM25无结果且向量检索分数较低，说明问题不相关
        if len(bm25_results) == 0 and len(vector_results) > 0:
            # BM25无结果，仅依赖向量检索时，检查向量检索的最高分数
            max_vector_score = max(r.get("distance", 0) for r in vector_results)
            # 如果向量检索分数低于阈值（0.6），认为不相关
            if max_vector_score < 0.6:
                fused_results = []
            else:
                fused_results = self.hybrid_retriever.rrf_fusion(bm25_results, vector_results)
        else:
            fused_results = self.hybrid_retriever.rrf_fusion(bm25_results, vector_results)
        
        return {"fused_results": fused_results}

    def _rerank(self, state: AgentState) -> dict:
        """
        重排节点 - 对检索结果按RRF分数排序
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 重排后的Top-K结果
        """
        fused_results = state.get("fused_results", [])
        
        # 按RRF分数排序
        fused_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        
        # 直接返回前K个结果
        reranked_results = fused_results[:RERANKER_CONFIG["final_top_k"]]
        
        return {"reranked_results": reranked_results}

    def _build_context(self, state: AgentState) -> dict:
        """
        上下文构建节点 - 智能处理表格数据，避免重复/破损数据干扰
        
        处理策略：
        1. 全局分析类问题：使用全文压缩策略
        2. 普通问题：
           - 检测并跳过破损的表格文本（没有Markdown表格标记但包含表格数据）
           - 将完整表格转换为JSON格式，便于LLM理解
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 包含上下文的字典
        """
        import re
        reranked_results = state.get("reranked_results", [])
        question = state.get("question", "")
        
        # 检测是否为全局分析类问题
        global_analysis_keywords = ["不一致", "口径不一致", "差异", "矛盾", "对比", "检查全文"]
        is_global_query = any(kw in question for kw in global_analysis_keywords)
        
        # 全局分析类问题：使用全文压缩策略
        if is_global_query:
            return self._build_full_text_context(state)
        
        # 使用实例变量中的文档标题
        doc_title = self.doc_title
        
        # 构建上下文
        context_parts = [f"【文档大标题】{doc_title}\n"]
        
        # 收集所有表格数据的内容特征（用于去重）
        table_content_signatures = set()
        for r in reranked_results:
            if r.get("metadata", {}).get("type") == "table":
                text = r.get("text", "")
                # 提取表格中的关键数据特征（设备名+数值模式）
                # 查找类似 "6633#_90m" 这样的列名模式
                col_patterns = re.findall(r'\w+#_\w+m', text)
                if col_patterns:
                    table_content_signatures.update(col_patterns)
        
        for i, result in enumerate(reranked_results, 1):
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            page = metadata.get("page", "unknown")
            doc_type = metadata.get("type", "text")
            
            # 如果存在表格数据，检查文本是否为破损的表格版本
            if doc_type == "text" and table_content_signatures:
                # 检查文本中是否包含与表格相同的设备/高度标识
                text_signatures = set(re.findall(r'\w+#\s+\w+m', text.replace('\n', ' ')))
                # 或者检查是否包含多行纯数字数据（没有Markdown表格标记）
                lines = text.strip().split('\n')
                numeric_lines = sum(1 for line in lines if re.search(r'\d+\.\d+', line) and '|' not in line)
                
                # 如果文本包含类似表格的数据特征，说明是破损的表格文本，跳过
                if numeric_lines >= 3 or text_signatures:
                    continue
            
            # 表格数据特殊处理：转换为JSON格式
            if doc_type == "table":
                import json
                # 解析Markdown表格
                lines = text.strip().split('\n')
                if len(lines) >= 2:
                    # 提取表头
                    header_line = lines[0]
                    columns = [c.strip() for c in header_line.split('|') if c.strip()]
                    
                    # 解析数据行
                    data_rows = []
                    for line in lines[2:]:  # 跳过表头和分隔行
                        if not line.strip():
                            continue
                        cells = [c.strip() for c in line.split('|') if c.strip()]
                        if len(cells) >= len(columns):
                            # 将每行转为字典：第一列作为行名，其他列作为键值对
                            row_dict = {}
                            row_name = cells[0] if cells else "unknown"
                            for j, col_name in enumerate(columns[1:], 1):
                                if j < len(cells):
                                    row_dict[col_name] = cells[j]
                            data_rows.append({columns[0]: row_name, "data": row_dict})
                    
                    # 生成JSON格式的表格数据
                    json_table = json.dumps(data_rows, ensure_ascii=False, indent=2)
                    
                    context_parts.append(
                        f"[{doc_title} 第{page}页, 类型: {doc_type}]\n"
                        f"{json_table}\n"
                    )
                else:
                    context_parts.append(
                        f"[{doc_title} 第{page}页, 类型: {doc_type}]\n{text}\n"
                    )
            else:
                context_parts.append(
                    f"[{doc_title} 第{page}页, 类型: {doc_type}]\n{text}\n"
                )
        
        context = "\n".join(context_parts)
        return {"context": context}

    def _generate_answer(self, state: AgentState) -> dict:
        """
        答案生成节点 - 调用LLM生成答案
        
        提示词包含：
        - 检索到的上下文资料
        - 用户问题
        - 回答要求（7条规则）
        - 答案格式模板
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 包含生成答案的字典
        """
        question = state["question"]
        context = state["context"]

        answer_prompt = f"""【资料】
{context}

【问题】{question}

你是电力设计研究院文档分析专家，请根据资料准确回答问题。

【回答要求】
1. 找不到相关信息时，回复"无法检索到相关资料，请提供更具体的问题"
2. 涉及多个项目时，逐一列出每个项目及其数值，不要遗漏
3. 同一数据存在多个数值或口径时，说明差异原因后给出合理答案
4. 涉及总量时，先列分项数据并计算总和，再与标注总量对比；不一致时说明原因
5. 检查不一致/差异/矛盾时，对比全文找出同一数据在不同位置的不同表述，包括：
   - 用词不规范（如专业术语、设备名称、技术参数在全文中的表述不统一）
6. 回答设备参数时，需包含型号、规格、数量等完整信息
7. 只输出资料中实际存在的数据，不要推测、编造或外推表格中不存在的数值
【答案格式】
答案：【精准原文提取结果，带单位/限定条件】
信息来源：《文档大标题》第X页（只取文档大标题，不要细化到小标题或表格内容）
补充说明：【多口径说明/取值依据，仅在数据有歧义时填写】

200字内"""

        answer = self._call_llm_with_fallback(
            "你是电力设计研究院文档分析专家",
            answer_prompt
        )
        return {"answer": answer}

    def _verify_and_correct_answer(self, state: AgentState) -> dict:
        """
        后处理验证节点 - 修正信息来源格式
        
        功能：
        - 统一信息来源中的文档标题格式
        - 支持多种格式的匹配和替换（方括号、多页码等）
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 包含修正后答案的字典
        """
        import re
        answer = state["answer"]
        doc_title = self.doc_title
        
        # 替换信息来源中的文档标题
        # 匹配各种格式的信息来源
        answer = re.sub(
            r'信息来源[：:].*?第(\d+)页',
            f'信息来源：《{doc_title}》第\\1页',
            answer
        )
        # 处理带方括号的格式
        answer = re.sub(
            r'信息来源[：:]【.*?】第(\d+)页',
            f'信息来源：《{doc_title}》第\\1页',
            answer
        )
        # 处理带方括号但页码在外的格式
        answer = re.sub(
            r'信息来源[：:]\[.*?\]第(\d+)页',
            f'信息来源：《{doc_title}》第\\1页',
            answer
        )
        # 处理多页码格式（如"第1、5、7页"）
        answer = re.sub(
            r'信息来源[：:].*?第([\d、]+)页',
            f'信息来源：《{doc_title}》第\\1页',
            answer
        )
        
        return {"answer": answer}

    def _verify_answer(self, state: AgentState) -> dict:
        """
        答案验证节点（当前未使用）
        
        用于验证答案准确性，返回JSON格式验证结果
        
        Args:
            state: 智能体状态
            
        Returns:
            dict: 包含验证反馈的字典
        """
        question = state["question"]
        answer = state["answer"]
        context = state["context"]

        verify_prompt = f"""【文档内容】
{context}

【问题】
{question}

【答案】
{answer}

验证答案准确性，返回 JSON：
{{"confidence": 0.95, "feedback": "验证反馈"}}"""

        verification = self._call_llm(
            "只返回 JSON 格式验证结果。",
            verify_prompt
        )

        return {"reasoning": verification}

    def ask(self, question: str) -> dict:
        """
        提问入口 - 单轮问答
        
        Args:
            question: 用户问题
            
        Returns:
            dict: 包含答案、问题类型、置信度等信息
        """
        initial_state = {
            "question": question,
            "question_type": "",
            "bm25_results": [],
            "vector_results": [],
            "fused_results": [],
            "reranked_results": [],
            "context": "",
            "answer": "",
            "reasoning": "",
            "confidence": 0.0
        }

        result = self.app.invoke(initial_state)

        return {
            "question": question,
            "answer": result["answer"],
            "question_type": result["question_type"],
            "confidence": result["reasoning"],
            "context_sources": len(result["reranked_results"])
        }

    def ask_batch(self, questions: List[str]) -> List[dict]:
        """
        批量提问
        
        Args:
            questions: 问题列表
            
        Returns:
            List[dict]: 答案列表
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'#'*60}")
            print(f"第 {i}/{len(questions)} 题")
            print(f"{'#'*60}")

            result = self.ask(question)
            results.append(result)

        return results
