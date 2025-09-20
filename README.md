# yuju

[![Python Versions](https://img.shields.io/pypi/pyversions/yuju)](https://pypi.org/project/yuju/)
[![PyPI Version](https://img.shields.io/pypi/v/yuju)](https://pypi.org/project/yuju/)

An interactive tool for visualizing vector fields and analyzing torque distributions.

- **Git repository**: <https://github.com/luocfprime/yuju/>


- **Documentation** <https://luocfprime.github.io/yuju/>


## Install

Prerequisites: You must have at least one Python package manager installed (e.g. [uv](https://docs.astral.sh/uv/getting-started/installation/)).

Install it from PyPI:

```bash
uv tool install yuju
```

Or, if you want to run it once without installing it, you can use the `uv run` command:

```bash
uv run --with yuju yuju xxx  # xxx being the subcommand you want to run
```


## Usage

```text
$ yuju -h
                                                                                                                                                   
 Usage: yuju [OPTIONS] COMMAND [ARGS]...                                                                                                           
                                                                                                                                                   
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion            Install completion for the current shell.                                                                       │
│ --show-completion               Show completion for the current shell, to copy it or customize the installation.                                │
│ --help                -h        Show this message and exit.                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ merge   Merge 3 Tecplot files into one.                                                                                                         │
│ viz     Aggregating vectors within a specified bounding box.                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT.
