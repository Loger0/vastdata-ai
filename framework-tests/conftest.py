# Vastbase connection configuration for framework tests
import os
import pytest

# pytest-asyncio required import
import pytest_asyncio  # noqa: F401

# Vastbase connection defaults (override via environment variables)
VASTBASE_HOST = os.environ.get("VASTBASE_HOST", "localhost")
VASTBASE_PORT = int(os.environ.get("VASTBASE_PORT", "15432"))
VASTBASE_DATABASE = os.environ.get("VASTBASE_DATABASE", "vastbase")
VASTBASE_USER = os.environ.get("VASTBASE_USER", "vexdb")
VASTBASE_PASSWORD = os.environ.get("VASTBASE_PASSWORD", "")


@pytest.fixture(scope="session", autouse=True)
def _setup_vastbase_connections():
    """Establish pyvastbase connections with expected aliases."""
    from pyvastbase import connect

    # Register connections with the aliases the tests expect
    connect(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        using="sync",
    )
    connect(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        using="async",
    )
    yield
