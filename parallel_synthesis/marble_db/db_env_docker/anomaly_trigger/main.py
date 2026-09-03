"""
Minimal local anomaly trigger for the MARBLE database benchmark.

This is not a verbatim copy of MARBLE's full DB environment. It is a smaller
local equivalent that preserves the anomaly signal our current benchmark uses:
PostgreSQL state plus pg_stat_statements / pg_locks evidence for the five
anomaly types present in the released database dataset.
"""

import argparse
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg2


SUPPORTED_ANOMALIES = [
    "INSERT_LARGE_DATA",
    "LOCK_CONTENTION",
    "VACUUM",
    "REDUNDANT_INDEX",
    "FETCH_LARGE_DATA",
]


def _env(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return value


def db_connect(application_name, *, autocommit=True):
    conn = psycopg2.connect(
        dbname=_env("DB_NAME", "sysbench"),
        user=_env("DB_USER", "test"),
        password=_env("DB_PASSWORD", "Test123_456"),
        host=_env("DB_HOST", "localhost"),
        port=_env("DB_PORT", "5432"),
        connect_timeout=5,
        application_name=str(application_name),
    )
    conn.autocommit = bool(autocommit)
    return conn


def run_sql(sql, application_name="marble_setup", fetch=False):
    conn = db_connect(application_name, autocommit=True)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        if fetch and cur.description is not None:
            return cur.fetchall()
        return None
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def column_defs(ncolumns, colsize):
    ncolumns = max(1, min(int(ncolumns), 20))
    colsize = max(8, min(int(colsize), 200))
    return ", ".join(
        "c{idx} varchar({size})".format(idx=i, size=colsize)
        for i in range(1, ncolumns + 1)
    )


def random_value_exprs(ncolumns, colsize):
    ncolumns = max(1, min(int(ncolumns), 20))
    colsize = max(8, min(int(colsize), 200))
    return ", ".join(
        "substr(md5(random()::text), 1, {size})".format(size=colsize)
        for _ in range(ncolumns)
    )


def ensure_base_table(table_name, ncolumns, colsize, *, rows=0):
    run_sql("DROP TABLE IF EXISTS {table};".format(table=table_name))
    run_sql(
        "CREATE TABLE {table} (id integer PRIMARY KEY, {cols}, created_at timestamptz DEFAULT now());".format(
            table=table_name,
            cols=column_defs(ncolumns, colsize),
        )
    )
    if int(rows) > 0:
        insert_rows(table_name, ncolumns, colsize, rows=int(rows), start_id=1)


def insert_rows(table_name, ncolumns, colsize, *, rows, start_id):
    values = random_value_exprs(ncolumns, colsize)
    end_id = int(start_id) + int(rows) - 1
    sql = (
        "INSERT INTO {table} "
        "SELECT g, {values}, now() "
        "FROM generate_series({start_id}, {end_id}) AS g;"
    ).format(
        table=table_name,
        values=values,
        start_id=int(start_id),
        end_id=end_id,
    )
    run_sql(sql, application_name="marble_insert_rows")


def run_parallel_calls(worker_fn, workers, total_calls):
    max_workers = max(1, min(int(workers), 8))
    total = max(max_workers, int(total_calls))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, idx) for idx in range(total)]
        for future in futures:
            future.result()


def anomaly_insert_large_data(args):
    table_name = "marble_insert_large_data"
    rows_per_insert = max(200, min(int(args.nrow), 4000))
    calls = 12
    ensure_base_table(table_name, args.ncolumn, args.colsize, rows=0)

    def worker(call_idx):
        start_id = 1 + int(call_idx) * rows_per_insert
        insert_rows(
            table_name,
            args.ncolumn,
            args.colsize,
            rows=rows_per_insert,
            start_id=start_id,
        )

    run_parallel_calls(worker, args.threads, calls)


def _spawn_background_process(role, table_name, colsize, hold_sec):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--internal_role",
        role,
        "--table_name",
        table_name,
        "--colsize",
        str(int(colsize)),
        "--hold_sec",
        str(float(hold_sec)),
    ]
    with open(os.devnull, "wb") as sink:
        subprocess.Popen(
            cmd,
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )


def anomaly_lock_contention(args):
    table_name = "marble_lock_contention"
    hold_sec = max(60.0, float(args.duration))
    rows = max(100, min(int(args.nrow), 2000))
    ensure_base_table(table_name, args.ncolumn, args.colsize, rows=rows)
    _spawn_background_process("lock_holder", table_name, args.colsize, hold_sec)
    time.sleep(1.5)
    waiter_count = max(2, min(int(args.threads), 4))
    for _ in range(waiter_count):
        _spawn_background_process("lock_waiter", table_name, args.colsize, hold_sec)
    time.sleep(1.5)


def anomaly_vacuum(args):
    table_name = "marble_vacuum"
    rows = max(1000, min(int(args.nrow), 5000))
    ensure_base_table(table_name, args.ncolumn, args.colsize, rows=rows)
    run_sql(
        "ALTER TABLE {table} SET (autovacuum_enabled = false);".format(table=table_name),
        application_name="marble_vacuum_setup",
    )
    delete_before = max(2, int(rows * 0.9))
    run_sql(
        "DELETE FROM {table} WHERE id < {bound};".format(
            table=table_name,
            bound=delete_before,
        ),
        application_name="marble_vacuum_delete",
    )
    run_sql(
        "VACUUM FULL {table};".format(table=table_name),
        application_name="marble_vacuum_full",
    )


def anomaly_redundant_index(args):
    table_name = "marble_redundant_index"
    rows = max(1000, min(int(args.nrow), 5000))
    ensure_base_table(table_name, args.ncolumn, args.colsize, rows=rows)
    index_sql = [
        "CREATE INDEX IF NOT EXISTS idx_{table}_c1_a ON {table} (c1);",
        "CREATE INDEX IF NOT EXISTS idx_{table}_c1_b ON {table} (c1);",
        "CREATE INDEX IF NOT EXISTS idx_{table}_c1_c2_a ON {table} (c1, c2);",
        "CREATE INDEX IF NOT EXISTS idx_{table}_c1_c2_b ON {table} (c1, c2);",
    ]
    for sql in index_sql:
        run_sql(
            sql.format(table=table_name),
            application_name="marble_redundant_index",
        )
    run_sql(
        "SELECT * FROM {table} WHERE c1 IS NOT NULL ORDER BY id LIMIT 100;".format(
            table=table_name
        ),
        application_name="marble_redundant_index_probe",
    )


def anomaly_fetch_large_data(args):
    table_name = "marble_fetch_orders"
    run_sql("DROP TABLE IF EXISTS {table};".format(table=table_name))
    run_sql(
        "CREATE TABLE {table} ("
        "id integer PRIMARY KEY, "
        "order_priority varchar(15), "
        "order_date date, "
        "payload text"
        ");".format(table=table_name),
        application_name="marble_fetch_setup",
    )
    rows = max(5000, min(int(args.nrow), 20000))
    run_sql(
        "INSERT INTO {table} "
        "SELECT g, "
        "CASE WHEN random() > 0.5 THEN '1-URGENT' ELSE '5-LOW' END, "
        "(date '1996-03-01' + ((random() * 800)::int)), "
        "repeat(substr(md5(random()::text), 1, 32), 8) "
        "FROM generate_series(1, {rows}) AS g;".format(table=table_name, rows=rows),
        application_name="marble_fetch_seed",
    )

    def worker(_):
        run_sql(
            "SELECT * FROM {table} ORDER BY id LIMIT 100;".format(table=table_name),
            application_name="marble_fetch_large_data",
        )

    run_parallel_calls(worker, args.threads, 24)


def run_internal_lock_holder(table_name, colsize, hold_sec):
    conn = db_connect("marble_lock_holder", autocommit=False)
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE {table} "
            "SET c1 = substr(md5(random()::text), 1, {colsize}) "
            "WHERE id = 1;".format(
                table=table_name,
                colsize=max(8, min(int(colsize), 200)),
            )
        )
        time.sleep(float(hold_sec))
        conn.commit()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def run_internal_lock_waiter(table_name, colsize, hold_sec):
    conn = db_connect("marble_lock_waiter", autocommit=False)
    cur = conn.cursor()
    try:
        cur.execute("SET statement_timeout = %s;", (int((float(hold_sec) + 15.0) * 1000),))
        cur.execute(
            "UPDATE {table} "
            "SET c1 = substr(md5(random()::text), 1, {colsize}) "
            "WHERE id = 1;".format(
                table=table_name,
                colsize=max(8, min(int(colsize), 200)),
            )
        )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal MARBLE DB anomaly trigger")
    parser.add_argument("--anomaly", choices=SUPPORTED_ANOMALIES)
    parser.add_argument("--threads", type=int, default=100)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--ncolumn", type=int, default=20)
    parser.add_argument("--nrow", type=int, default=20000)
    parser.add_argument("--colsize", type=int, default=100)
    parser.add_argument("--internal_role", type=str, default="")
    parser.add_argument("--table_name", type=str, default="")
    parser.add_argument("--hold_sec", type=float, default=60.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.internal_role == "lock_holder":
        run_internal_lock_holder(args.table_name, args.colsize, args.hold_sec)
        return
    if args.internal_role == "lock_waiter":
        run_internal_lock_waiter(args.table_name, args.colsize, args.hold_sec)
        return

    random.seed(0)
    if args.anomaly == "INSERT_LARGE_DATA":
        anomaly_insert_large_data(args)
        return
    if args.anomaly == "LOCK_CONTENTION":
        anomaly_lock_contention(args)
        return
    if args.anomaly == "VACUUM":
        anomaly_vacuum(args)
        return
    if args.anomaly == "REDUNDANT_INDEX":
        anomaly_redundant_index(args)
        return
    if args.anomaly == "FETCH_LARGE_DATA":
        anomaly_fetch_large_data(args)
        return
    raise ValueError("Unsupported anomaly: {value}".format(value=args.anomaly))


if __name__ == "__main__":
    main()
