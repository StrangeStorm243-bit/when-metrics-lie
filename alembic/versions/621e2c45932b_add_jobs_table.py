"""add jobs table

Revision ID: 621e2c45932b
Revises: 2a4add_spec_json_and_rerun_of
Create Date: 2026-01-29 10:15:34.544852

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "621e2c45932b"
down_revision: Union[str, Sequence[str], None] = "2a4add_spec_json_and_rerun_of"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("experiment_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("started_at", sa.String(), nullable=True),
        sa.Column("finished_at", sa.String(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result_run_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.experiment_id"]),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
        sa.PrimaryKeyConstraint("job_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("jobs")
