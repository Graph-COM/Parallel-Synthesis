from datetime import date
from typing import Any, Dict, List

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def _tool_schema_lines(tool_schemas: List[Dict[str, Any]]) -> str:
    lines = []
    for schema in tool_schemas:
        lines.append(str(schema))
    return "\n".join(lines)


def build_tool_system_prompt(tool_schemas: List[Dict[str, Any]]) -> str:
    today = date.today().isoformat()
    tools_blob = _tool_schema_lines(tool_schemas)
    return f"""
You are a deep research assistant with tool-use capability.
Current date: {today}

You may call one or more tools to solve the user query.
Tool signatures are listed below:
<tools>
{tools_blob}
</tools>

When you need a tool, output exactly one tool call using XML tags:
{TOOL_CALL_OPEN}
{{"name": "tool_name", "arguments": {{...}}}}
{TOOL_CALL_CLOSE}

If the tool is PythonInterpreter, put code right after the JSON in <code></code> tags:
{TOOL_CALL_OPEN}
{{"name": "PythonInterpreter", "arguments": {{}}}}
<code>
print("hello")
</code>
{TOOL_CALL_CLOSE}

Tool outputs will be returned in:
{TOOL_RESPONSE_OPEN}
...tool output...
{TOOL_RESPONSE_CLOSE}

When you are ready to finish, provide your final answer inside:
{ANSWER_OPEN}...{ANSWER_CLOSE}

Never fabricate tool outputs.
""".strip()


def build_baseline_user_prompt(
    question: str,
    files: List[str],
) -> str:
    attachment_note = ""
    if files:
        attachment_note = "\nAttached files you can parse with parse_file:\n" + "\n".join(files)
    return f"""
Target Question: {question}

You are a helpful assistant.
You must reason step-by-step to solve the provided Target Question.
Return the final answer only inside <answer>YOUR_FINAL_ANSWER</answer>.
{attachment_note}
""".strip()


def _build_toolcall_judger_prompt(
    *,
    question: str,
    files: List[str],
    prior_agents_text: str,
) -> str:
    attachment_note = ""
    if files:
        attachment_note = "\nAttached files (use parse_file if needed):\n" + "\n".join(files)
    return f"""
You are a Task Summarizer.

Use the previous agents' contexts as reference. They may be provided as explicit text or through KV memory.

Reason over their answers and evidence step by step, compare the candidate answers, and give the best possible final answer. If two or more agents give the same plausible answer, use it directly. If only one agent gives a plausible answer, consider whether its evidence fits the question instead of dismissing it just because it is a minority answer.

Use tools only when the previous agents do not give any plausible direct answer, or when all candidate answers are clearly unsupported. If you use a tool, use only the listed tools, make one targeted check, and then answer.

Content from Previous Agents:
{prior_agents_text}

Input Question:
{question}
{attachment_note}

Guidelines:
- prefer synthesizing prior agents over starting from scratch
- if agents disagree, choose the best-supported answer, not necessarily the majority answer
- if agents agree on a plausible answer, do not verify it with tools
- if uncertain, still choose the most plausible answer from the prior agents' work
- prefer a direct short candidate answer over refusals, tool traces, or generic explanations
- do not output an empty response, markdown fence, tool schema, raw XML, or raw tool response
- output only the final answer inside <answer>YOUR_FINAL_ANSWER</answer>

Your response:
""".strip()


def build_textmas_role_prompt(
    role: str,
    question: str,
    context: str,
    files: List[str],
) -> str:
    role = role.lower().strip()

    if role.startswith("sub_agent_"):
        return build_baseline_user_prompt(
            question=question,
            files=files,
        )

    if role == "judger":
        judger_context = str(context).strip() or "[none yet]"
        return _build_toolcall_judger_prompt(
            question=question,
            files=files,
            prior_agents_text=judger_context,
        )

    raise ValueError(f"Unsupported role: {role}")


def build_parallel_kv_judger_prompt(
    question: str,
    files: List[str],
) -> str:
    return _build_toolcall_judger_prompt(
        question=question,
        files=files,
        prior_agents_text="[provided through KV memory]",
    )
