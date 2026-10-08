"""Exercise plotting through the installed command's Python entry script."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset,kind", [("iris", "curve"), ("digits", "curve"),
                                           ("iris", "histogram")])
def test_cli_saves_plot_without_display(tmp_path, dataset, kind):
    pytest.importorskip("matplotlib")
    output = tmp_path / "plots" / f"{dataset}-{kind}.png"
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"), OPENBLAS_NUM_THREADS="1")
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "admm-runner.py"), dataset,
         "--plot", kind, "--repetitions", "1", "--iterations", "2",
         "--seed", "42", "--output", str(output)],
        capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert f"Saved plot to {output}" in result.stdout


def test_cli_rejects_output_without_plot(tmp_path):
    output = tmp_path / "unused.png"
    result = subprocess.run(
        [sys.executable, str(ROOT / "admm-runner.py"), "iris", "--output", str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "--output requires --plot" in result.stderr
    assert not output.exists()
