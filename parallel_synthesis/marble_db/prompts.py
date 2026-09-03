import json
from datetime import date
from typing import Any, Dict, List


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def _tool_schema_lines(tool_schemas: List[Dict[str, Any]]) -> str:
    return "\n".join(str(schema) for schema in tool_schemas)


def build_db_tool_system_prompt(tool_schemas: List[Dict[str, Any]]) -> str:
    today = date.today().isoformat()
    tools_blob = _tool_schema_lines(tool_schemas)
    return f"""
You are a database diagnosis assistant with tool-use capability.
Current date: {today}

You may call one or more tools to solve the assigned database diagnosis task.
Tool signatures are listed below:
<tools>
{tools_blob}
</tools>

When you need a tool, output exactly one tool call using XML tags:
{TOOL_CALL_OPEN}
{{"name": "tool_name", "arguments": {{...}}}}
{TOOL_CALL_CLOSE}

Tool outputs will be returned in:
{TOOL_RESPONSE_OPEN}
...tool output...
{TOOL_RESPONSE_CLOSE}

When you are ready to finish, answer directly in plain text or JSON.
    Never fabricate tool outputs.
    """.strip()


def build_worker_system_prompt(
    *,
    agent_meta: Dict[str, Any],
    tool_schemas: List[Dict[str, Any]],
) -> str:
    agent_id = str(agent_meta.get("agent_id", "")).strip()
    profile = str(agent_meta.get("profile", "")).strip()
    marble_system_message = (
        f'You are "{agent_id}": "{profile}"\n'
        "As a role-playing agent, you embody a dynamic character with unique traits, motivations, and skills.\n"
        "Your goal is to engage not only with users but also with other agents in the environment. "
        "Collaborate, compete, or form alliances as you navigate through immersive storytelling and challenges. "
        "Interact meaningfully with fellow agents, contributing to the evolving narrative and responding creatively "
        "to their actions. Maintain consistency with your character's background and personality, and be prepared "
        "to adapt to the evolving dynamics of the scenario.\n"
        "Remember, your responses should enhance the experience and encourage user engagement while enriching "
        "interactions with other agents."
    )
    return marble_system_message + "\n\n" + build_db_tool_system_prompt(tool_schemas)


def _planner_base_prompt(
    *,
    question: str,
    current_progress: str,
    agents_meta: List[Dict[str, Any]],
) -> str:
    lines = [
        "You are an orchestrator assigning tasks to a group of agents based on their profiles, current progress, and task description.",
        "",
        "Task Description:",
        question,
        "",
        "Current Progress:",
        current_progress,
        "",
        "Agent Profiles:",
    ]
    for agent in agents_meta:
        agent_id = str(agent.get("agent_id", "")).strip()
        relationships = agent.get("relationships", []) or []
        profile = str(agent.get("profile", "")).strip()
        lines.append(f"- Agent ID: {agent_id}")
        lines.append(f"  Relationships: {relationships}")
        lines.append(f"  Profile: {profile}")
    return "\n".join(lines).strip()


def build_planner_messages(
    *,
    question: str,
    current_progress: str,
    agents_meta: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    prompt = _planner_base_prompt(
        question=question,
        current_progress=current_progress,
        agents_meta=agents_meta,
    ) + (
        "\n\nBased on the above, assign the next task to each agent that needs to perform an action.\n"
        "Provide the assignments in the following JSON format:\n\n"
        "{\n"
        '  "tasks": {\n'
        '    "agent1": "...",\n'
        '    "agent2": "..."\n'
        "  },\n"
        '  "continue": true\n'
        "}\n"
    )
    system_message = (
        "You are a task assignment system for multiple AI agents based on their profiles and current progress.\n"
        "Your task is to analyze the current state and assign the next task to each agent that requires an action.\n"
        "Don't ask agents to assign tasks to other agents.\n"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]


def build_worker_user_prompt(
    *,
    agent_meta: Dict[str, Any],
    assigned_task: str,
    all_agents_meta: List[Dict[str, Any]],
) -> str:
    agent_id = str(agent_meta.get("agent_id", "")).strip()
    _ = all_agents_meta
    return f"""
    Agent ID: {agent_id}
    Task: {assigned_task}

    Use the available tools if needed, then provide your final response in plain text.
    Start your final response with exactly:
    This is the response from {agent_id}:
    """.strip()


def render_agents_results_summary(agent_results: List[Dict[str, Any]]) -> str:
    summary = "Agent Results Summary:\n"
    for result in agent_results:
        if isinstance(result, dict) and len(result) == 1:
            agent_id, agent_output = next(iter(result.items()))
            summary += f"\nAgent {agent_id}:\n{agent_output}\n"
        else:
            summary += f"\n{result}\n"
    return summary


def _judger_task_brief(
    *,
    question: str,
    labels: List[str],
    max_root_causes: int,
) -> str:
    task_text = str(question or "").strip()
    cut = len(task_text)
    for marker in (
        "The planner should",
        "Agents can also chat",
        "Please make the decision after",
    ):
        idx = task_text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    task_text = task_text[:cut].strip()
    label_blob = ", ".join("'{}'".format(str(label).strip()) for label in labels if str(label).strip())
    parts = []
    if task_text:
        parts.append("Task:\n{}".format(task_text))
    parts.append(
        "Current Objective:\nIdentify the most likely root cause labels behind the current database performance issue."
    )
    parts.append(
        "Allowed Labels:\n{}".format(label_blob)
    )
    parts.append(
        "Decision Rule:\nSelect at most {} labels, and only choose labels that are supported by the previous agents' findings.".format(
            int(max(1, max_root_causes))
        )
    )
    return "\n\n".join(parts)


def _judger_output_instructions(
    *,
    labels: List[str],
    max_root_causes: int,
) -> str:
    label_blob = ", ".join("'{}'".format(str(label).strip()) for label in labels if str(label).strip())
    return (
        "Decide which root causes are best supported by the agents' findings.\n"
        "Before the final answer, write a short Rationale based on the previous agents' diagnoses.\n"
        "Briefly summarize what the agents found, compare the competing explanations, and choose the root causes that seem most likely overall.\n"
        "Use the agents' SQL/tool evidence when it is available, but do not require perfect evidence for every selected label.\n"
        "If some findings are weak or conflicting, make the best diagnosis from the strongest available signals rather than returning no answer.\n"
        "Do not invent row counts, lock states, SQL queries, tables, indexes, or agent conclusions that were not in the prior investigations.\n"
        "Return the final selected root-cause labels inside exactly one <answer>...</answer> block.\n"
        "Inside the answer block, output only a JSON array of labels.\n"
        "The answer array must contain at most {} labels.\n"
        "If only one label is well supported, return a one-element array.\n"
        "Choose labels only from this list:\n"
        "{}\n"
        "Do not put justification inside the <answer> block.\n"
        "Do not wrap the answer in Markdown fences.\n"
        "Do not output more than one <answer> block."
    ).format(int(max(1, max_root_causes)), label_blob)


def _judger_analysis_instructions() -> str:
    return (
        "You are the final database diagnosis judger.\n"
        "You are given the prior agents' investigations, including their findings, cited evidence, and any reported tool-use results.\n"
        "Your job is to inspect and summarize those investigations carefully, and produce the best final diagnosis.\n\n"
        "Work in this order:\n"
        "1. inspect the current task background and the allowed label set\n"
        "2. inspect what each previous agent investigated and what evidence they found\n"
        "3. choose the most strongly supported root causes and summarize why"
    )


def build_textmas_judger_prompt(
    *,
    question: str,
    summary_text: str,
    labels: List[str],
    max_root_causes: int,
) -> str:
    return (
        f"{_judger_analysis_instructions()}\n\n"
        f"{_judger_task_brief(question=question, labels=labels, max_root_causes=max_root_causes)}\n\n"
        f"Previous Agent Investigations:\n{summary_text}\n\n"
        f"{_judger_output_instructions(labels=labels, max_root_causes=max_root_causes)}"
    )


def build_parallel_kv_judger_prompt(
    *,
    question: str,
    labels: List[str],
    max_root_causes: int,
) -> str:
    return (
        f"{_judger_analysis_instructions()}\n\n"
        f"{_judger_task_brief(question=question, labels=labels, max_root_causes=max_root_causes)}\n\n"
        "Previous Agent Investigations:\n[provided through KV memory]\n\n"
        f"{_judger_output_instructions(labels=labels, max_root_causes=max_root_causes)}"
    )
