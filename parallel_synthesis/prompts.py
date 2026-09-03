from typing import Any, Dict, Optional


CONTEXT_QA_TASKS = {"2wiki_multihopqa"}
PARALLEL_DIALOGUE_TASKS = {
    "wildchat",
    "lmsys_chat",
    "ultrachat",
    "toucan",
    "toucan_single_parallel",
    "dta_tool",
}
TRACE_RECONSTRUCTION_TASKS = {"toucan_multi_parallel"}
PROMPT_INFERENCE_TASKS = {"flan"}
PRETRAINING_TASKS = {
    "2wiki_multihopqa",
    "browsecomp_textmas_toolcall",
    "wildchat",
    "lmsys_chat",
    "ultrachat",
    "toucan_single_parallel",
    "toucan_multi_parallel",
    "dta_tool",
    "flan",
}


def _default_system_message(args=None) -> str:
    model_name = getattr(args, "model_name", "").lower() if args else ""
    if "qwen" in model_name:
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return "You are a helpful assistant."


def format_fixed_parallel_cache_text(
    task: str,
    item: Dict[str, Any],
    cache_text: str,
    cache_idx: int,
    cache_meta: Optional[Dict[str, Any]] = None,
) -> str:
    text = str(cache_text).strip()
    if not text:
        return text
    return text


def _clip_context(text: str, args=None) -> str:
    rendered = str(text or "").strip()
    limit = int(getattr(args, "text_mas_context_length", -1)) if args else -1
    if limit is None or limit < 0:
        return rendered
    return rendered[:limit]


def _message(system_message: str, user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def _build_context_qa_prompt(
    question: str,
    *,
    context: str = "",
    include_previous_agent_context: bool = False,
    include_text_context: bool = False,
    args=None,
) -> str:
    rendered_question = str(question or "").strip()
    rendered_context = _clip_context(context, args) if include_text_context else str(context or "").strip()

    if include_previous_agent_context:
        reference_line = (
            "Read the previous agents' outputs carefully and synthesize the final answer."
            if include_text_context
            else "You are provided with latent summaries from previous agents as reference."
        )
        previous_block = ""
        if include_text_context and rendered_context:
            previous_block = f"""
Previous Agent Outputs:
{rendered_context}
"""
        return f"""
You are the final document-grounded QA agent in a multi-agent pipeline.
{reference_line}

Requirements:
- reason step by step using the information available from previous agents
- provide the final answer inside \\boxed{{YOUR_FINAL_ANSWER}} when possible
- if the available evidence is insufficient or irrelevant, respond with exactly: irrelevant to current context provided
- do not rely on outside knowledge beyond the provided question and previous-agent evidence
{previous_block}
Question:
{rendered_question}

Your response:
"""

    context_block = ""
    if rendered_context:
        context_block = f"""
Context Documents:
{rendered_context}
"""
    return f"""
You are a document-grounded QA agent in a multi-agent pipeline.
Read the provided documents carefully, think step by step, and summarize the facts relevant to the question before deciding on the answer.

Requirements:
- use only the provided documents
- provide the final answer inside \\boxed{{YOUR_FINAL_ANSWER}} when possible
- if the provided documents are insufficient or irrelevant to the question, respond with exactly: irrelevant to current context provided
- keep the response grounded in the evidence from the documents
{context_block}
Question:
{rendered_question}

Your response:
"""


def _build_context_qa_subagent_prompt(
    question: str,
    *,
    context: str = "",
    args=None,
) -> str:
    rendered_question = str(question or "").strip()
    rendered_context = str(context or "").strip()
    context_block = ""
    if rendered_context:
        context_block = f"""
Context Documents:
{rendered_context}
"""
    return f"""
You are a document-grounded QA sub-agent in a multi-agent pipeline.
You are given only the current chunk of documents, not the full set of available evidence.
Read the current documents carefully and first summarize the key entities, relations, events, and attributes stated in this chunk.
Then explain which parts of this chunk may help answer the question, including bridge facts where one document introduces an entity and another document describes that entity further.

Requirements:
- first summarize the key information in the current documents, even if some facts are only indirectly connected to the question
- preserve bridge entities and relation chains that may need to be combined with information from other chunks
- then highlight the parts of this chunk that may help answer the question
- use only the current documents and do not rely on outside knowledge
- if the current chunk already contains enough evidence to answer confidently, end with the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}
- if the current chunk is not sufficient to fully answer the question, do not output any boxed answer; provide only the relevant evidence you found
- if this chunk contains no relevant information, say that no relevant information was found in this chunk
- do not force a final answer when the current chunk only provides partial evidence
{context_block}
Question:
{rendered_question}

Your response:
"""


def _build_dialogue_branch_prompt(
    question: str,
    *,
    context: str = "",
) -> str:
    rendered_question = str(question or "").strip()
    rendered_context = str(context or "").strip()
    context_block = ""
    if rendered_context:
        context_block = f"""
Assigned Branch:
{rendered_context}
"""
    return f"""
You are a dialogue-and-tool-trace analyst in a parallel multi-agent pipeline.
You are given the current conversation/request and one assigned branch of prior dialogue or tool execution.
Summarize the facts, tool results, constraints, open issues, and commitments from this branch that matter for writing the next assistant response.

Requirements:
- preserve concrete names, numbers, identifiers, URLs, and tool outputs when they appear
- use only the provided conversation/request and assigned branch
- do not draft the final user-facing answer yet
{context_block}
Current Conversation / Request:
{rendered_question}

Your response:
"""


def _build_trace_reconstruction_branch_prompt(
    question: str,
    *,
    context: str = "",
) -> str:
    rendered_question = str(question or "").strip()
    rendered_context = str(context or "").strip()
    context_block = ""
    if rendered_context:
        context_block = f"""
Assigned Tool-Trace Chunk:
{rendered_context}
"""
    return f"""
You are analyzing one tool-trace chunk from a multi-turn conversation.
Infer what the original user task was and what the assistant planned to do before making the tool calls.

Requirements:
- use only the provided instruction and assigned tool-trace chunk
- identify the task as concretely as possible
- preserve useful tool names, entities, identifiers, and constraints
- do not write other sub-agents' answers
{context_block}
Instruction:
{rendered_question}

Your response:
"""


def _build_prompt_inference_branch_prompt(
    question: str,
    *,
    context: str = "",
) -> str:
    rendered_question = str(question or "").strip()
    rendered_context = str(context or "").strip()
    context_block = ""
    if rendered_context:
        context_block = f"""
Assigned Assistant Response:
{rendered_context}
"""
    return f"""
You are inferring an unseen user prompt from one assistant response branch.
Summarize what this response implies about the original user request.

Requirements:
- use only the provided instruction and assigned assistant response
- infer the likely original prompt as concretely as possible
- preserve topic, requested format, and any constraints implied by the response
- do not draft the final combined answer yet
{context_block}
Instruction:
{rendered_question}

Your response:
"""


def _build_baseline_user_prompt(question: str, context: str = "", args=None) -> str:
    task = getattr(args, "task", "")
    context = str(context or "").strip()

    if task in {"gsm8k", "aime2024", "aime2025"}:
        return f"""
Solve the input question step by step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
The final answer must be a single numerical value with no extra text or units inside the box.

Input Question:
{question}

Your response:
"""

    if task in {"gpqa", "medqa"}:
        return f"""
Solve the input question step by step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question:
{question}

Your response:
"""

    if task in {"mbppplus", "humanevalplus"}:
        return f"""
Write a correct and self-contained Python solution for the input question.
You must put all python code in a markdown code block. For example ```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.

Input Question:
{question}

Your response:
"""

    if task in PARALLEL_DIALOGUE_TASKS:
        if context:
            return _build_dialogue_branch_prompt(question, context=context)
        return f"""
You are an assistant.
Write the next assistant response for the conversation/request below.

Conversation / Request:
{question}

Your response:
"""

    if task in TRACE_RECONSTRUCTION_TASKS:
        if context:
            return _build_trace_reconstruction_branch_prompt(question, context=context)
        return f"""
You are reconstructing user tasks and assistant tool-call planning from multi-turn tool traces.
Follow the instruction below exactly.

Instruction:
{question}

Your response:
"""

    if task in PROMPT_INFERENCE_TASKS:
        if context:
            return _build_prompt_inference_branch_prompt(question, context=context)
        return f"""
You are inferring the original user prompt from several alternative assistant responses.
Follow the instruction below exactly.

Instruction:
{question}

Your response:
"""

    if task in CONTEXT_QA_TASKS:
        return _build_context_qa_prompt(question, context=context, args=args)

    return f"""
Question:
{question}

Solve the question step by step and clearly state the final answer at the end.

Your response:
"""


def _build_judger_user_prompt(
    question: str,
    *,
    context: str = "",
    include_text_context: bool = False,
    args=None,
) -> str:
    task = getattr(args, "task", "")
    if task in PRETRAINING_TASKS and not include_text_context:
        return str(question or "").strip()

    context = _clip_context(context, args) if include_text_context else ""
    context_block = ""
    if include_text_context and context:
        context_block = f"""
Content from Previous Agents:
{context}
"""

    if task in {"gsm8k", "aime2024", "aime2025"}:
        return f"""
You are a task summarizer.
Solve the input question step by step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
The final answer must be a single numerical value with no extra text or units inside the box.
{"Use the text context from previous agents as reference when helpful." if include_text_context else "You are provided with latent information from previous agents as reference."}
{context_block}
Input Question:
{question}

Your response:
"""

    if task in {"gpqa", "medqa"}:
        return f"""
You are a task summarizer.
Solve the input question step by step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.
{"Use the text context from previous agents as reference when helpful." if include_text_context else "You are provided with latent information from previous agents as reference."}
{context_block}
Input Question:
{question}

Your response:
"""

    if task in {"mbppplus", "humanevalplus"}:
        return f"""
You are a task summarizer.
Write a correct and self-contained Python solution for the input question.
You must put all python code in a markdown code block. For example ```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
{"Use the text context from previous agents as reference when helpful." if include_text_context else "You are provided with latent information from previous agents as reference."}
{context_block}
Input Question:
{question}

Your response:
"""

    if task in PARALLEL_DIALOGUE_TASKS:
        reference_line = (
            "Using summaries from previous agents, write the next assistant response for the conversation/request below."
            if include_text_context
            else "You are provided with latent summaries from previous agents as reference."
        )
        return f"""
You are the final assistant in a multi-agent pipeline.
{reference_line}
{context_block}
Conversation / Request:
{question}

Requirements:
- answer directly and naturally
- preserve concrete facts, tool outputs, and user constraints when they are relevant
- do not mention hidden agents, internal summaries, or latent information

Your response:
"""

    if task in TRACE_RECONSTRUCTION_TASKS:
        reference_line = (
            "Using summaries from previous agents, infer each original user task and assistant tool-call plan."
            if include_text_context
            else "You are provided with latent summaries from previous agents as reference."
        )
        return f"""
You are reconstructing multi-turn tool-use trajectories.
{reference_line}
{context_block}
Instruction:
{question}

Requirements:
- write one ordered answer covering every sub-agent trace
- infer the user task and the assistant planning text before the tools
- preserve tool names and concrete constraints when they are implied by the traces

Your response:
"""

    if task in PROMPT_INFERENCE_TASKS:
        reference_line = (
            "Using summaries from previous agents, infer the most likely original user prompt."
            if include_text_context
            else "You are provided with latent summaries from previous agents as reference."
        )
        return f"""
You are inferring an original user prompt from parallel assistant responses.
{reference_line}
{context_block}
Instruction:
{question}

Requirements:
- write one concise original prompt only
- synthesize the strongest shared signal from the available assistant responses
- do not mention hidden branches, previous agents, or internal summaries

Your response:
"""

    if task in CONTEXT_QA_TASKS:
        return _build_context_qa_prompt(
            question,
            context=context,
            include_previous_agent_context=True,
            include_text_context=include_text_context,
            args=args,
        )

    raise NotImplementedError(f"Task {task} is not implemented in the judger prompt.")


def build_baseline_prompt(question: str, context: str = "", args=None) -> list[dict[str, str]]:
    return _message(
        _default_system_message(args),
        _build_baseline_user_prompt(question, context=context, args=args),
    )


def build_judger_prompt(
    question: str,
    *,
    context: str = "",
    include_text_context: bool = False,
    args=None,
) -> list[dict[str, str]]:
    return _message(
        _default_system_message(args),
        _build_judger_user_prompt(
            question,
            context=context,
            include_text_context=include_text_context,
            args=args,
        ),
    )


def build_parallel_kv_prompt(role: str, question: str, context: str = "", args=None) -> list[dict[str, str]]:
    if role == "judger":
        return build_judger_prompt(question, context="", include_text_context=False, args=args)
    if getattr(args, "task", "") in CONTEXT_QA_TASKS:
        return _message(
            _default_system_message(args),
            _build_context_qa_subagent_prompt(question, context=context, args=args),
        )
    return build_baseline_prompt(question, context=context, args=args)


def build_text_mas_prompt(role: str, question: str, context: str = "", args=None) -> list[dict[str, str]]:
    if role == "judger":
        return build_judger_prompt(
            question,
            context=context,
            include_text_context=True,
            args=args,
        )
    if getattr(args, "task", "") in CONTEXT_QA_TASKS:
        return _message(
            _default_system_message(args),
            _build_context_qa_subagent_prompt(question, context=context, args=args),
        )
    return build_baseline_prompt(question, context=context, args=args)


def build_agent_messages_single_agent(question: str, context: str = "", args=None):
    return build_baseline_prompt(question, context=context, args=args)


__all__ = [
    "CONTEXT_QA_TASKS",
    "PARALLEL_DIALOGUE_TASKS",
    "PRETRAINING_TASKS",
    "PROMPT_INFERENCE_TASKS",
    "TRACE_RECONSTRUCTION_TASKS",
    "build_agent_messages_single_agent",
    "build_baseline_prompt",
    "build_judger_prompt",
    "build_parallel_kv_prompt",
    "build_text_mas_prompt",
    "format_fixed_parallel_cache_text",
]
