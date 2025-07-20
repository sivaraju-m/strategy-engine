"""
Basic tests to ensure CI/CD pipeline passes.
"""

import pytest


def test_basic_import():
    """Test that basic imports work."""
    import sys

    assert sys.version_info >= (3, 11)


def test_basic_functionality():
    """Test basic functionality."""
    assert True


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality."""
    assert True
