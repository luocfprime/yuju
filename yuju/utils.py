import logging
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_plt_file(file_path: str | Path) -> np.ndarray:
    """
    Load Tecplot-like .plt file containing scalar or vector data.

    Automatically skips header lines (variables, zone) and detects
    the number of columns from the first data line.

    Args:
        file_path (str | Path): Path to file.

    Returns:
        np.ndarray: Parsed array of shape (N, n_columns)
    """
    logger.info(f"Parsing file: {file_path}")
    points = []

    with open(file_path) as f:
        lines = f.readlines()

    # Skip header lines
    data_lines = [
        line.strip() for line in lines if line.strip() and not line.strip().lower().startswith(("variables", "zone"))
    ]

    if not data_lines:
        raise ValueError("No data lines found in the file")

    # Determine number of columns from the first data line
    n_cols = len(data_lines[0].split())

    # Build regex for n_cols floating-point numbers
    float_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    data_pattern = re.compile(r"^\s*" + r"\s+".join([float_re] * n_cols) + r"\s*$")

    for line in tqdm(data_lines, desc="Processing data"):
        match = data_pattern.match(line)
        if not match:
            continue
        points.append(list(map(np.float64, match.groups())))

    logger.info(f"Loaded {len(points):,} points out of {len(data_lines):,} data lines.")
    if len(points) == 0:
        raise ValueError("No points found after parsing")

    return np.array(points, dtype=np.float64)


def write_plt_file_vec(path: Path, points: np.ndarray) -> None:
    """Write points to a Tecplot ASCII .plt file (FEPOINT format).

    This function writes nodal data into a Tecplot-compatible
    `.plt` file in **FEPOINT** format, with six variables:
    `x, y, z, u, v, w`.

    The output file will only contain node data (scattered points).
    No element connectivity is written (`E=0`).

    Args:
        path (Path): Destination file path with `.plt` extension.
        points (np.ndarray): Array of shape `(n, 6)` containing point data:
            - Column 0: x-coordinate
            - Column 1: y-coordinate
            - Column 2: z-coordinate
            - Column 3: u-value
            - Column 4: v-value
            - Column 5: w-value

    Raises:
        ValueError: If `points` does not have exactly 6 columns.

    Example:
        >>> import numpy as np
        >>> from pathlib import Path
        >>> data = np.random.rand(10, 6)
        >>> write_plt_file(Path("output.plt"), data)

    Notes:
        - Values are written in scientific notation with six decimal places.
        - Use Tecplot's scatter/point plotting to visualize the data.
    """
    variables = ["x", "y", "z", "u", "v", "w"]

    if points.shape[1] != len(variables):
        raise ValueError(f"Expected {len(variables)} columns, got {points.shape[1]}")

    n = len(points)
    with open(path, "w") as f:
        f.write("VARIABLES = " + ", ".join(f'"{v}"' for v in variables) + "\n")
        f.write(f"ZONE N={n}, E=0, F=FEPOINT, ET=POINT\n")
        np.savetxt(f, points, fmt="%.6e")
