# extensions.py

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(
    engine_options={
        "pool_pre_ping": True,   # 🛡 prevents stale connections
        "pool_recycle": 180,     # 🧽 reconnect every 3 minutes
        "pool_size": 20,         # 🚀 more connections for multi-thread jobs
        "max_overflow": 40,      # allow extra connections
    }
)