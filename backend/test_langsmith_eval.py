"""
RAG 输出质量评估 —— 基于 LangSmith Evaluation 框架。
评估维度：答案相关性、忠实度（幻觉检测）、上下文相关性、答案正确性。
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import evaluate, Client  # noqa: E402
from langsmith.schemas import Run, Example  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# ── 配置 ──────────────────────────────────────────────
client = Client()
DATASET_NAME = "RAG"
COLLECTION_NAME = os.getenv("EVAL_COLLECTION_NAME", "default")

MODEL = os.getenv("MODEL")
API_KEY = os.getenv("ARK_API_KEY")
BASE_URL = os.getenv("BASE_URL")

_eval_model = init_chat_model(
    model=MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
)


# ── 结构化评分模型 ──────────────────────────────────────
class RelevanceScore(BaseModel):
    reason: str = Field(description="判断理由")
    score: int = Field(description="1=不相关, 2=部分相关, 3=相关, 4=高度相关, 5=完全匹配")


class FaithfulnessScore(BaseModel):
    reason: str = Field(description="判断理由，指出哪些陈述有/无文档支撑")
    score: int = Field(description="1=严重幻觉, 2=多处无支撑, 3=基本忠实, 4=高度忠实, 5=完全忠实")


class ContextRelevanceScore(BaseModel):
    reason: str = Field(description="判断理由")
    score: int = Field(description="1=全部无关, 2=少数相关, 3=半数相关, 4=多数相关, 5=全部相关")


class AnswerCorrectnessScore(BaseModel):
    reason: str = Field(description="对比参考答案，指出一致与不一致之处")
    score: int = Field(description="1=完全错误, 2=大部分错误, 3=部分正确, 4=大部分正确, 5=完全正确")


# ── 目标函数 ──────────────────────────────────────────
def rag_target(inputs: dict) -> dict:
    question = inputs["Input"]

    from app.agent.agent import chat_with_agent

    result = chat_with_agent(
        user_text=question,
        collection_name=COLLECTION_NAME,
        enable_rag=True,
    )

    rag_trace = result.get("rag_trace") or {}
    retrieved_docs = rag_trace.get("retrieved_chunks") or rag_trace.get("expanded_retrieved_chunks", [])

    return {
        "answer": result["response"],
        "question": question,
        "retrieved_docs": retrieved_docs,
        "rag_trace": rag_trace,
    }


# ── 评估器 1：答案相关性 ────────────────────────────────
def answer_relevance(run: Run, _example: Example) -> dict:
    outputs = run.outputs or {}
    question = outputs.get("question", "")
    answer = outputs.get("answer", "")

    prompt = f"""你是一个 RAG 系统的评估专家。请评估以下回答与问题的相关性。

问题：{question}

回答：{answer}

评估标准：
- 回答是否直接回应了问题？
- 回答是否包含与问题无关的内容？
- 回答是否完整覆盖了问题的核心意图？"""

    result = _eval_model.with_structured_output(RelevanceScore).invoke(prompt)
    return {"key": "answer_relevance", "score": result.score / 5.0, "comment": result.reason}


# ── 评估器 2：忠实度 / 幻觉检测 ─────────────────────────
def faithfulness(run: Run, _example: Example) -> dict:
    outputs = run.outputs or {}
    answer = outputs.get("answer", "")
    docs = outputs.get("retrieved_docs", [])

    if not docs:
        return {
            "key": "faithfulness",
            "score": 0,
            "comment": "未检索到任何文档，无法评估忠实度",
        }

    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {d.get('filename', '?')} (第{d.get('page_number', '?')}页):\n{d.get('text', '')}"
        for i, d in enumerate(docs)
    )

    prompt = f"""你是一个 RAG 系统的忠实度评估专家。判断回答中的每一条信息是否都可以从以下检索到的文档中得到支撑。

检索文档：
{docs_text}

回答：
{answer}

评估标准：
- 回答中的事实陈述是否都能在文档中找到依据？
- 回答是否添加了文档中没有的信息（幻觉）？
- 如果回答引用了文档外的常识知识，这不算幻觉"""

    result = _eval_model.with_structured_output(FaithfulnessScore).invoke(prompt)
    return {"key": "faithfulness", "score": result.score / 5.0, "comment": result.reason}


# ── 评估器 3：上下文相关性 ──────────────────────────────
def context_relevance(run: Run, _example: Example) -> dict:
    outputs = run.outputs or {}
    question = outputs.get("question", "")
    docs = outputs.get("retrieved_docs", [])

    if not docs:
        return {
            "key": "context_relevance",
            "score": 0,
            "comment": "未检索到任何文档",
        }

    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {d.get('filename', '?')}: {d.get('text', '')[:300]}"
        for i, d in enumerate(docs)
    )

    prompt = f"""你是一个检索质量评估专家。评估以下检索到的文档片段是否与用户问题相关。

用户问题：{question}

检索到的文档片段：
{docs_text}

评估标准：
- 文档内容是否与问题主题匹配？
- 文档是否包含回答该问题所需的信息？
- 是否有大量无关文档混入？"""

    result = _eval_model.with_structured_output(ContextRelevanceScore).invoke(prompt)
    return {"key": "context_relevance", "score": result.score / 5.0, "comment": result.reason}


# ── 评估器 4：答案正确性（对比参考答案）──────────────────
def answer_correctness(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    question = outputs.get("question", "")
    answer = outputs.get("answer", "")
    reference = (example.outputs or {}).get("Output", "")

    if not reference:
        return {
            "key": "answer_correctness",
            "score": 0,
            "comment": "无参考答案可供对比",
        }

    prompt = f"""你是一个法律知识评估专家。请对比 RAG 系统生成的回答与参考答案。

问题：{question}

参考答案：{reference}

RAG 系统回答：{answer}

评估标准：
- RAG 回答的核心法律结论是否与参考答案一致？
- 引用的法律条文是否正确？
- 是否存在关键信息遗漏或错误？"""

    result = _eval_model.with_structured_output(AnswerCorrectnessScore).invoke(prompt)
    return {"key": "answer_correctness", "score": result.score / 5.0, "comment": result.reason}


# ── 评估器 5：检索覆盖率 ────────────────────────────────
def retrieval_coverage(run: Run, _example: Example) -> dict:
    outputs = run.outputs or {}
    trace = outputs.get("rag_trace", {})
    stages = []

    if trace.get("retrieval_stage") == "initial":
        stages.append("initial_retrieval")
    if trace.get("retrieval_stage") == "expanded":
        stages.append("expanded_retrieval")
    if trace.get("grade_score"):
        stages.append(f"graded:{trace['grade_score']}")
    if trace.get("rewrite_strategy"):
        stages.append(f"rewrite:{trace['rewrite_strategy']}")
    if trace.get("rerank_applied"):
        stages.append("reranked")

    doc_count = len(outputs.get("retrieved_docs", []))
    score = min(1.0, doc_count / 5.0)

    return {
        "key": "retrieval_coverage",
        "score": score,
        "comment": f"检索到的文档数: {doc_count}, 执行阶段: {', '.join(stages)}",
    }


# ── 启动评估 ────────────────────────────────────────────
if __name__ == "__main__":
    evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            answer_relevance,
            faithfulness,
            context_relevance,
            answer_correctness,
            retrieval_coverage,
        ],
        experiment_prefix="RAG quality eval",
    )
