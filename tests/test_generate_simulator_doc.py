"""Reproducibility tests for src/generate_simulator_doc.py and docs/simulator_export.md."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.generate_simulator_doc import build_simulator_export_markdown

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_INPUTS = [
    ROOT / "export" / "manifest.json",
    ROOT / "export" / "clutch_shift.json",
    ROOT / "export" / "model_a.json",
    ROOT / "export" / "player_offsets.csv",
    ROOT / "docs" / "baseline_quality.json",
    ROOT / "docs" / "predictive_comparison.json",
    ROOT / "docs" / "simulator_export.md",
]

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_INPUTS),
    reason="export/ or docs/ inputs required by src/generate_simulator_doc.py are absent.",
)


def test_generate_simulator_doc_reproduces_committed_markdown(tmp_path: Path) -> None:
    """Running src/generate_simulator_doc.py from a repo root reproduces docs/simulator_export.md byte-for-byte."""
    committed = (ROOT / "docs" / "simulator_export.md").read_text(encoding="utf-8")

    # No timestamp lines exist in docs/simulator_export.md, so zero lines need to be ignored.
    assert build_simulator_export_markdown(ROOT) == committed

    copy_root = tmp_path / "repo_copy"
    (copy_root / "src").mkdir(parents=True)
    shutil.copy2(
        ROOT / "src" / "generate_simulator_doc.py",
        copy_root / "src" / "generate_simulator_doc.py",
    )
    shutil.copytree(ROOT / "export", copy_root / "export")
    (copy_root / "docs").mkdir()
    shutil.copy2(
        ROOT / "docs" / "baseline_quality.json",
        copy_root / "docs" / "baseline_quality.json",
    )
    shutil.copy2(
        ROOT / "docs" / "predictive_comparison.json",
        copy_root / "docs" / "predictive_comparison.json",
    )

    proc = subprocess.run(
        [sys.executable, "src/generate_simulator_doc.py"],
        cwd=copy_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    regenerated = (copy_root / "docs" / "simulator_export.md").read_text(encoding="utf-8")
    assert regenerated == committed
