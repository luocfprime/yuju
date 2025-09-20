import os

import pytest


@pytest.fixture(scope="session")
def project_root(pytestconfig) -> str:
    return str(pytestconfig.rootdir)


@pytest.fixture(scope="session")
def test_data(project_root) -> str:
    return os.path.join(project_root, "tests", "data")


def pytest_addoption(parser):
    parser.addoption(
        "--refresh-golden",
        action="store_true",
        default=False,
        help="Refresh golden standard files instead of comparing against them.",
    )


@pytest.fixture
def refresh_golden(request):
    return request.config.getoption("--refresh-golden")
