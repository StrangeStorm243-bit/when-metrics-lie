from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    task = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    n_trials = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=False)
    dataset_fingerprint = Column(String, nullable=False)
    dataset_schema_json = Column(Text, nullable=False)  # JSON string
    scenarios_json = Column(Text, nullable=False)  # JSON string
    created_at = Column(String, nullable=False)  # ISO8601 UTC string
    spec_schema_version = Column(String, nullable=True)
    # Canonical JSON snapshot of the experiment spec (semantics), used for deterministic reruns.
    spec_json = Column(Text, nullable=False, default="")

    # Relationships
    runs = relationship("Run", back_populates="experiment", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.experiment_id"), nullable=False)
    status = Column(String, nullable=False)  # queued, running, completed, failed
    created_at = Column(String, nullable=False)  # ISO8601 UTC string
    started_at = Column(String, nullable=True)  # ISO8601 UTC string
    finished_at = Column(String, nullable=True)  # ISO8601 UTC string
    results_path = Column(String, nullable=False)
    artifacts_dir = Column(String, nullable=False)
    seed_used = Column(Integer, nullable=False)
    error = Column(Text, nullable=True)
    # Optional linkage to the original run when this is a rerun.
    rerun_of = Column(String, nullable=True)

    # Relationships
    experiment = relationship("Experiment", back_populates="runs")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    kind = Column(String, nullable=False)  # e.g. "plot"
    path = Column(String, nullable=False)  # relative path
    meta_json = Column(Text, nullable=False)  # JSON string
    created_at = Column(String, nullable=False)  # ISO8601 UTC string

    # Relationships
    run = relationship("Run", back_populates="artifacts")

