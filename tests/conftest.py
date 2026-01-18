import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
