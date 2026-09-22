import os
import sys

# This project has no setup.py/pyproject.toml, so `src/` is only on
# sys.path when it's the entry script's directory (see CLAUDE.md). Pytest
# doesn't run that way, so put it on the path explicitly here.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest


@pytest.fixture(scope='session')
def app():
    # webapp.textsimilar's `@app.route(...)` decorators run once, the first
    # time `webapp.textsimilar.routes` is imported (Python caches module
    # imports), and bind to whichever Flask app is active at that moment.
    # A second create_app() call later in the session would produce a Flask
    # app with no routes registered on it, so this fixture is session-scoped
    # to guarantee exactly one create_app() call for the whole test run.
    from webapp.textsimilar import Environment, create_app

    return create_app(Environment.TEST)


@pytest.fixture()
def client(app):
    return app.test_client()
