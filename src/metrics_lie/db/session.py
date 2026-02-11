from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

DB_PATH = Path(".spectra_registry/spectra.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite with pysqlite (included in Python stdlib)
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
_db_init_lock = Lock()
_db_initialized = False


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def _ensure_local_sqlite_schema() -> None:
    """Initialize SQLite schema once per process for fresh environments."""
    global _db_initialized
    if _db_initialized:
        return
    with _db_init_lock:
        if _db_initialized:
            return
        init_db()
        _db_initialized = True


@contextmanager
def get_session() -> Session:
    """Context manager for database sessions."""
    _ensure_local_sqlite_schema()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
