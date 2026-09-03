import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List


DEFAULT_MARBLE_DB_URL = (
    "https://raw.githubusercontent.com/ulab-uiuc/MARBLE/main/"
    "multiagentbench/database/database_main.jsonl"
)


def _download_dataset_text() -> str:
    try:
        with urllib.request.urlopen(DEFAULT_MARBLE_DB_URL, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Failed to download the MARBLE DB dataset from its public repository."
        ) from exc


def _relationship_map(row: Dict[str, Any]) -> Dict[str, List[str]]:
    rel_map: Dict[str, List[str]] = {}
    for raw in row.get("relationships", []) or []:
        if not isinstance(raw, list) or len(raw) != 3:
            continue
        left = str(raw[0]).strip()
        right = str(raw[1]).strip()
        rel = str(raw[2]).strip()
        if not left or not right:
            continue
        rel_map.setdefault(left, []).append(f"{left} {rel} {right}")
        rel_map.setdefault(right, []).append(f"{left} {rel} {right}")
    return rel_map


def _normalize_agent(agent: Dict[str, Any], rel_map: Dict[str, List[str]]) -> Dict[str, Any]:
    agent_id = str(agent.get("agent_id", "")).strip()
    return {
        "agent_id": agent_id,
        "type": str(agent.get("type", "")).strip(),
        "profile": str(agent.get("profile", "")).strip(),
        "relationships": list(rel_map.get(agent_id, [])),
    }


def load_marble_db_rows(
    *,
    max_samples: int = -1,
) -> Iterable[Dict[str, Any]]:
    text = _download_dataset_text()
    rows: List[Dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)

    if max_samples > 0:
        rows = rows[:max_samples]

    out: List[Dict[str, Any]] = []
    for row in rows:
        rel_map = _relationship_map(row)
        task = row.get("task", {}) if isinstance(row.get("task"), dict) else {}
        env = row.get("environment", {}) if isinstance(row.get("environment"), dict) else {}
        planner = (
            row.get("engine_planner", {})
            if isinstance(row.get("engine_planner"), dict)
            else {}
        )
        item = {
            "benchmark": "marble_db",
            "scenario": str(row.get("scenario", "database")).strip() or "database",
            "task_id": int(row.get("task_id", 0) or 0),
            "question": str(task.get("content", "")).strip(),
            "output_format": str(task.get("output_format", "")).strip(),
            "labels": [str(x).strip() for x in (task.get("labels", []) or []) if str(x).strip()],
            "gold_root_causes": [
                str(x).strip()
                for x in (task.get("root_causes", []) or [])
                if str(x).strip()
            ],
            "number_of_labels_pred": int(task.get("number_of_labels_pred", 2) or 2),
            "planner_initial_progress": str(
                planner.get("initial_progress", "Starting the simulation.")
            ).strip(),
            "agents_meta": [
                _normalize_agent(agent, rel_map)
                for agent in (row.get("agents", []) or [])
                if isinstance(agent, dict)
            ],
            "environment": {
                "init_sql": str(env.get("init_sql", "") or ""),
                "anomalies": list(env.get("anomalies", []) or []),
            },
            "meta": {"source_row": row},
        }
        out.append(item)

    for item in out:
        yield item
