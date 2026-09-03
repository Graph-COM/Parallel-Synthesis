from typing import Any, Dict

from parallel_synthesis.toolcall.tools import BaseTool, ToolRegistry, ToolSpec


class QueryDBTool(BaseTool):
    def __init__(self, session: Any) -> None:
        self.session = session
        self.spec = ToolSpec(
            name="query_db",
            description=(
                "Query the PostgreSQL database with the given SQL statement. "
                "Please keep to the pg_stat_statements table and related catalog views, "
                "and make sure your query itself will not cause the database to hang. "
                "Recommended to use one single query at a time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A SQL query to execute. Recommended tables include "
                            "`pg_stat_statements`, `pg_locks`, `pg_stat_user_indexes`, "
                            "`pg_indexes`, `pg_stat_all_tables`, `pg_stat_progress_vacuum`, "
                            "and `pg_stat_user_tables`. Only include query statements in this param."
                        ),
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        )

    def call(self, arguments: Dict[str, Any]) -> str:
        sql = str(arguments.get("sql", "")).strip()
        if not sql:
            return "[query_db] invalid arguments: sql must be a non-empty string"
        return str(self.session.query(sql))


def build_marble_db_tool_registry(session: Any) -> ToolRegistry:
    return ToolRegistry([QueryDBTool(session)])
