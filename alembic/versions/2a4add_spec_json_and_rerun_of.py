"""add spec_json to experiments and rerun_of to runs

Revision ID: 2a4add_spec_json_and_rerun_of
Revises: 1e8f3e64b284
Create Date: 2026-01-29 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2a4add_spec_json_and_rerun_of"
down_revision: Union[str, Sequence[str], None] = "1e8f3e64b284"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add spec_json and rerun_of."""
    # Add spec_json to experiments; allow existing rows with empty string.
    op.add_column(
        "experiments",
        sa.Column("spec_json", sa.Text(), nullable=False, server_default=""),
    )

    # Add rerun_of to runs for optional linkage between runs.
    op.add_column(
        "runs",
        sa.Column("rerun_of", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema by removing spec_json and rerun_of."""
    op.drop_column("runs", "rerun_of")
    op.drop_column("experiments", "spec_json")
