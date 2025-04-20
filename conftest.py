"""Fallback asyncio test runner.

If pytest-asyncio is not installed, this lightweight plugin ensures that
async def tests are collected and executed instead of being skipped by
Pytest.  It registers a simple event loop and runs the coroutine inline.
This keeps CI green without adding an external dependency.
"""
import asyncio
import inspect

import pytest


def pytest_configure(config):
    # Ensure the custom marker is recognised to silence warnings
    config.addinivalue_line("markers", "asyncio: mark a test as asynchronous")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Execute ``async def`` test functions inline using the default event loop."""
    test_fn = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_fn):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        sig = inspect.signature(test_fn)
        accepted = {k: v for k, v in pyfuncitem.funcargs.items() if k in sig.parameters}
        loop.run_until_complete(test_fn(**accepted))
        loop.close()
        return True  # indicate we handled the test
