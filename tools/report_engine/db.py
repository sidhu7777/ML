import os
import pandas as pd
from sqlalchemy import create_engine, text, bindparam
from dotenv import load_dotenv

load_dotenv()

_ENGINE = None


def init_engine(engine):
    """
    Initialize a shared SQLAlchemy engine from the main app.
    """
    global _ENGINE
    _ENGINE = engine


def get_engine():
    """
    Return the shared engine or create one from DATABASE_URL.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL for report engine")

    # Keep connections healthy for long-running jobs
    _ENGINE = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 30},
    )
    return _ENGINE


def _connect():
    return get_engine().connect()


# =====================================================
# DEBUG / INSPECTION HELPERS
# =====================================================

def list_tables():
    with _connect() as conn:
        rows = conn.execute(text("SHOW TABLES")).fetchall()
    print("Tables in database:")
    for t in rows:
        print("-", t[0])


def describe_table(table_name: str):
    with _connect() as conn:
        rows = conn.execute(text(f"DESCRIBE {table_name}")).fetchall()
    print(f"\nColumns in {table_name}:")
    for col in rows:
        print(col)


# =====================================================
# CORE DATA ACCESS FUNCTIONS
# =====================================================

def get_project_by_id(project_id: int, conn=None):
    close_conn = False
    if conn is None:
        conn = _connect()
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM tbl_project
            WHERE id = :project_id
        """)
        row = conn.execute(query, {"project_id": project_id}).mappings().first()
        return dict(row) if row else None
    finally:
        if close_conn:
            conn.close()


def get_network_logs_for_sessions(session_ids: list[int], conn=None) -> pd.DataFrame:
    if not session_ids:
        return pd.DataFrame()

    close_conn = False
    if conn is None:
        conn = _connect()
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM tbl_network_log
            WHERE session_id IN :session_ids
        """).bindparams(bindparam("session_ids", expanding=True))

        df = pd.read_sql(query, conn, params={"session_ids": session_ids})
        return df
    finally:
        if close_conn:
            conn.close()


def get_project_regions(project_id: int, conn=None) -> list[dict]:
    close_conn = False
    if conn is None:
        conn = _connect()
        close_conn = True

    try:
        query = text("""
            SELECT
                id,
                name,
                ST_AsText(region) AS region_wkt
            FROM map_regions
            WHERE tbl_project_id = :project_id
              AND status = 1
        """)
        rows = conn.execute(query, {"project_id": project_id}).mappings().all()
        return [dict(r) for r in rows]
    finally:
        if close_conn:
            conn.close()


def get_user_thresholds(user_id: int, debug: bool = False, conn=None) -> dict | None:
    close_conn = False
    if conn is None:
        conn = _connect()
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM thresholds
            WHERE user_id = :user_id
            LIMIT 1
        """)
        row = conn.execute(query, {"user_id": user_id}).mappings().first()
        data = dict(row) if row else None
    finally:
        if close_conn:
            conn.close()

    if debug:
        print("\n================ DB THRESHOLD ROW =================")
        print(f"user_id = {user_id}")
        if not data:
            print("NO ROW RETURNED FROM DB")
            return None
        for k, v in data.items():
            print(f"{k}: {repr(v)}")
        print("===================================================\n")

    return data


def get_user_by_id(user_id: int, conn=None) -> dict | None:
    close_conn = False
    if conn is None:
        conn = _connect()
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM tbl_user
            WHERE id = :user_id
            LIMIT 1
        """)
        row = conn.execute(query, {"user_id": user_id}).mappings().first()
        return dict(row) if row else None
    finally:
        if close_conn:
            conn.close()


def update_project_download_path(project_id: int, download_path: str, conn=None) -> None:
    """
    Update tbl_project.Download_path for the given project.
    """
    close_conn = False
    if conn is None:
        conn = _connect()
        close_conn = True

    try:
        query = text("""
            UPDATE tbl_project
            SET Download_path = :download_path
            WHERE id = :project_id
        """)
        conn.execute(query, {"download_path": download_path, "project_id": project_id})
        conn.commit()
    finally:
        if close_conn:
            conn.close()
