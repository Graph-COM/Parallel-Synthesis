import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


LOCAL_DB_ENV_DOCKER_DIR = Path(__file__).resolve().parent / "db_env_docker"


def split_sql_statements(sql: str) -> List[str]:
    statements = re.split(r";\s*\n", str(sql or ""))
    return [stmt.strip() for stmt in statements if stmt.strip()]


class MarbleDBSessionBase:
    def setup(self):
        return self

    def close(self) -> None:
        return None

    def query(self, sql: str) -> str:
        raise NotImplementedError

    def __enter__(self):
        return self.setup()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DisabledMarbleDBSession(MarbleDBSessionBase):
    def __init__(self, reason: str) -> None:
        self.reason = str(reason).strip() or "database backend disabled"

    def query(self, sql: str) -> str:
        _ = sql
        return (
            "[query_db] database backend is unavailable for this run. "
            f"Reason: {self.reason}"
        )


class MarbleDockerDBSession(MarbleDBSessionBase):
    def __init__(self, item: Dict[str, Any], args: Any) -> None:
        self.item = item
        self.args = args
        repo_dir_raw = str(getattr(args, "marble_repo_dir", "")).strip()
        self.repo_dir = Path(repo_dir_raw).expanduser() if repo_dir_raw else None
        if self.repo_dir is not None:
            self.compose_dir = self.repo_dir / "marble" / "environments" / "db_env_docker"
            self.compose_source = "external_marble_repo"
        else:
            self.compose_dir = LOCAL_DB_ENV_DOCKER_DIR
            self.compose_source = "vendored_local"
        self.anomaly_trigger_dir = self.compose_dir / "anomaly_trigger"
        self._setup_complete = False

    def _require_psycopg2(self):
        try:
            import psycopg2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "psycopg2 is required for the MARBLE DB backend. "
                "Install `psycopg2-binary` in the environment running this benchmark."
            ) from exc
        return psycopg2

    def _connect(self):
        psycopg2 = self._require_psycopg2()
        connection = psycopg2.connect(
            user=str(getattr(self.args, "db_user", "test")),
            password=str(getattr(self.args, "db_password", "Test123_456")),
            database=str(getattr(self.args, "db_name", "sysbench")),
            host=str(getattr(self.args, "db_host", "localhost")),
            port=str(getattr(self.args, "db_port", "5432")),
            connect_timeout=int(getattr(self.args, "db_connect_timeout_sec", 5)),
        )
        connection.autocommit = True
        return connection

    def _subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["POSTGRES_USER"] = str(getattr(self.args, "db_user", "test"))
        env["POSTGRES_PASSWORD"] = str(
            getattr(self.args, "db_password", "Test123_456")
        )
        env["POSTGRES_DB"] = str(getattr(self.args, "db_name", "sysbench"))
        env["POSTGRES_PORT"] = str(getattr(self.args, "db_port", "5432"))
        env["DB_HOST"] = str(getattr(self.args, "db_host", "localhost"))
        env["DB_PORT"] = str(getattr(self.args, "db_port", "5432"))
        env["DB_NAME"] = str(getattr(self.args, "db_name", "sysbench"))
        env["DB_USER"] = str(getattr(self.args, "db_user", "test"))
        env["DB_PASSWORD"] = str(getattr(self.args, "db_password", "Test123_456"))
        return env

    def _run_subprocess(
        self,
        cmd: List[str],
        *,
        cwd: Path,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        env = self._subprocess_env()
        if extra_env:
            env.update(extra_env)
        subprocess.run(cmd, cwd=str(cwd), check=True, env=env)

    def _docker_compose_cmd(self, *extra: str) -> List[str]:
        base = shlex.split(
            str(getattr(self.args, "db_docker_compose_cmd", "docker compose")).strip()
        )
        if not base:
            raise RuntimeError("--db_docker_compose_cmd must not be empty.")
        return base + list(extra)

    def _python_cmd(self, *extra: str) -> List[str]:
        base = shlex.split(
            str(getattr(self.args, "db_anomaly_python_cmd", "python3")).strip()
        )
        if not base:
            raise RuntimeError("--db_anomaly_python_cmd must not be empty.")
        return base + list(extra)

    def _start_containers(self) -> None:
        if not self.compose_dir.is_dir():
            raise RuntimeError(
                "MARBLE DB docker directory not found at "
                f"{self.compose_dir} (source={self.compose_source})."
            )
        if bool(getattr(self.args, "db_restart_containers_per_sample", True)):
            self._run_subprocess(
                self._docker_compose_cmd("down", "-v"),
                cwd=self.compose_dir,
            )
        self._run_subprocess(
            self._docker_compose_cmd("up", "-d", "--remove-orphans"),
            cwd=self.compose_dir,
        )

    def _wait_for_db(self) -> None:
        deadline = time.time() + float(getattr(self.args, "db_ready_timeout_sec", 120))
        last_error = ""
        while time.time() < deadline:
            try:
                connection = self._connect()
                connection.close()
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1.0)
        raise RuntimeError(f"Database did not become ready in time. Last error: {last_error}")

    def _initialize_database(self) -> None:
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute("SET client_min_messages TO WARNING;")
            cursor.execute(
                "SELECT pg_terminate_backend(pid) "
                "FROM pg_stat_activity "
                "WHERE pid <> pg_backend_pid() "
                "AND application_name LIKE 'marble_%';"
            )
            init_sql = str(self.item.get("environment", {}).get("init_sql", "") or "")
            for statement in split_sql_statements(init_sql):
                cursor.execute(statement)
            cursor.execute("RESET client_min_messages;")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
            cursor.execute("SELECT pg_stat_statements_reset();")
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            connection.close()

    def _trigger_anomalies(self) -> None:
        anomalies = self.item.get("environment", {}).get("anomalies", []) or []
        for anomaly in anomalies:
            if not isinstance(anomaly, dict):
                continue
            anomaly_type = str(anomaly.get("anomaly", "")).strip()
            if not anomaly_type:
                continue
            cmd = self._python_cmd("main.py", "--anomaly", anomaly_type)
            duration = float(getattr(self.args, "db_anomaly_duration_sec", 5.0))
            if duration > 0:
                cmd.extend(["--duration", str(duration)])
            for key in ("threads", "ncolumn", "nrow", "colsize"):
                if key not in anomaly:
                    continue
                value = anomaly.get(key)
                if value is None or str(value).strip() == "":
                    continue
                cmd.extend([f"--{key}", str(value)])
            self._run_subprocess(cmd, cwd=self.anomaly_trigger_dir)

    def setup(self):
        if self._setup_complete:
            return self
        self._start_containers()
        self._wait_for_db()
        self._initialize_database()
        self._trigger_anomalies()
        time.sleep(float(getattr(self.args, "db_post_setup_sleep_sec", 2.0)))
        self._setup_complete = True
        return self

    def close(self) -> None:
        if not self._setup_complete:
            return
        if bool(getattr(self.args, "db_shutdown_after_item", False)):
            self._run_subprocess(
                self._docker_compose_cmd("down", "-v"),
                cwd=self.compose_dir,
            )
        self._setup_complete = False

    def query(self, sql: str) -> str:
        connection = None
        cursor = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            statement_timeout_ms = int(
                getattr(self.args, "db_statement_timeout_ms", 15000)
            )
            cursor.execute(f"SET statement_timeout = {statement_timeout_ms};")
            sql_queries = split_sql_statements(sql)
            if not sql_queries:
                return "[query_db] empty SQL query"
            result = []
            for sql_query in sql_queries:
                cursor.execute(sql_query)
                if cursor.description is not None:
                    result = cursor.fetchall()
            cursor.close()
            connection.close()
            preview = str(result)
            max_chars = int(getattr(self.args, "db_max_result_chars", 12000))
            if len(preview) > max_chars:
                preview = preview[:max_chars] + "... [truncated]"
            return (
                "Your query on the database was successful"
                f"{'' if len(result) else ' but no data was returned'}. \n"
                f"Your query is: {sql_queries} \n"
                f"Result: {preview}"
            )
        except Exception as exc:
            return (
                "An error occurred while you tried to query the database: "
                f"{str(exc)}"
            )
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            except Exception:
                pass
            try:
                if connection is not None:
                    connection.close()
            except Exception:
                pass


def build_db_session(args: Any, item: Dict[str, Any]) -> MarbleDBSessionBase:
    backend = str(getattr(args, "db_backend", "disabled")).strip().lower()
    if backend == "disabled":
        return DisabledMarbleDBSession(
            "use --db_backend=marble_docker to enable live local DB queries"
        )
    if backend == "marble_docker":
        return MarbleDockerDBSession(item, args)
    raise ValueError(f"Unsupported db_backend: {backend}")
