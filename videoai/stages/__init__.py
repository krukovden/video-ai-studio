"""Importing this package registers every stage in the registry."""
from videoai.stages import (  # noqa: F401
    s01_ingest,
    s02_quality,
    s02b_sync,
    s03_transcribe,
    s04_analyze,
    s05_plan,
    s06_render_draft,
)
