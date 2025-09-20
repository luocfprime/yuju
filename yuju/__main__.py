# flake8: noqa: B008

import logging
from pathlib import Path

import numpy as np
import typer

from yuju.__about__ import __application__
from yuju.app import PolyscopeApp
from yuju.utils import load_plt_file, write_plt_file_vec

logger = logging.getLogger(__name__)

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


def configure_logging(log_level: str):
    # Map text log level to numeric
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter(f"Invalid log level: {log_level}")

    # By default, root logger follows the chosen level
    root_level = numeric_level

    # Special case: if DEBUG is chosen, don't expose 3rd-party debug
    if numeric_level == logging.DEBUG:
        root_level = logging.INFO

    # Configure root logger
    logging.basicConfig(level=root_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure application logger separately
    app_logger = logging.getLogger(__application__)
    if numeric_level == logging.DEBUG:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(numeric_level)


@app.command()
def merge(
    data_x_path: Path = typer.Option(
        ...,
        "--data-x",
        "-x",
        help="Path to data file (e.g. Tecplot .plt file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    data_y_path: Path = typer.Option(
        ...,
        "--data-y",
        "-y",
        help="Path to data file (e.g. Tecplot .plt file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    data_z_path: Path = typer.Option(
        ...,
        "--data-z",
        "-z",
        help="Path to data file (e.g. Tecplot .plt file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    data_out_path: Path = typer.Option(
        ...,
        "--data-out",
        "-o",
        help="Path to output data file (e.g. Tecplot .plt file)",
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        case_sensitive=False,
    ),
):
    """Merge 3 Tecplot files into one."""
    configure_logging(log_level)

    data_x = load_plt_file(data_x_path)
    data_y = load_plt_file(data_y_path)
    data_z = load_plt_file(data_z_path)

    if not data_x.shape[0] == data_y.shape[0] == data_z.shape[0]:
        logger.error("Input files must have the same number of points")
        raise typer.Exit()
    if not data_x.shape[1] == data_y.shape[1] == data_z.shape[1] == 4:
        logger.error("Input files must have 4 columns")
        raise typer.Exit()

    if not np.allclose(data_x[:, :3], data_y[:, :3]) and not np.allclose(data_x[:, :3], data_z[:, :3]):
        logger.error("Input files must have the same positional-coordinates")
        raise typer.Exit()

    pos = data_x[:, :3]
    u = data_x[:, 3:]
    v = data_y[:, 3:]
    w = data_z[:, 3:]

    data_out = np.hstack([pos, u, v, w])
    write_plt_file_vec(data_out_path, data_out)

    logger.info(f"Wrote to {data_out_path}")


@app.command()
def viz(
    data_path: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to data file (e.g. Tecplot .plt file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    mesh_path: Path = typer.Option(
        ...,
        "--mesh",
        "-m",
        help="Path to mesh file (e.g. .stl file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        case_sensitive=False,
    ),
):
    """Aggregating vectors within a specified bounding box."""
    configure_logging(log_level)
    app_instance = PolyscopeApp(data_path, mesh_path)
    app_instance.run()


def main():
    app()


if __name__ == "__main__":
    main()
