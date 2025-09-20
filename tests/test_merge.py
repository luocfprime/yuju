from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from yuju.__main__ import app
from yuju.utils import load_plt_file

runner = CliRunner()


def test_merge_files(tmp_path: Path, test_data):
    # Prepare paths
    data_dir = Path(test_data)
    x_file = data_dir / "x.plt"
    y_file = data_dir / "y.plt"
    z_file = data_dir / "z.plt"
    expected_file = data_dir / "merged.plt"
    out_file = tmp_path / "out.plt"

    assert x_file.exists()
    assert y_file.exists()
    assert z_file.exists()
    assert expected_file.exists()

    # Run CLI
    result = runner.invoke(
        app,
        [
            "merge",
            "--data-x",
            str(x_file),
            "--data-y",
            str(y_file),
            "--data-z",
            str(z_file),
            "--data-out",
            str(out_file),
            "--log-level",
            "INFO",
        ],
    )

    # CLI should exit cleanly
    assert result.exit_code == 0, f"CLI failed with output: {result.stdout}"

    # Load data
    expected = load_plt_file(expected_file)
    actual = load_plt_file(out_file)

    # Verify content
    assert np.allclose(actual, expected), "Merged file does not match expected merged.plt"
