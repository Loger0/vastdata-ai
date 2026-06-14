# Vastbase connection configuration for framework tests
import os

# pytest-asyncio required import
import pytest_asyncio  # noqa: F401

# Vastbase connection defaults (override via environment variables)
VASTBASE_HOST = os.environ.get("VASTBASE_HOST", "localhost")
VASTBASE_PORT = int(os.environ.get("VASTBASE_PORT", "15432"))
VASTBASE_DATABASE = os.environ.get("VASTBASE_DATABASE", "vastbase")
VASTBASE_USER = os.environ.get("VASTBASE_USER", "vexdb")
VASTBASE_PASSWORD = os.environ.get("VASTBASE_PASSWORD", "Vexdb@123")
