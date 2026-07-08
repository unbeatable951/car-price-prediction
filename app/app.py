"""
app/app.py
===========
Flask application entry point, using the "application factory" pattern
(create_app()) rather than a bare module-level `app = Flask(__name__)`.

WHY THE FACTORY PATTERN
---------------------------
1. Testability: tests can call create_app() to get a fresh, isolated
   app instance per test, instead of importing one shared global `app`
   that accumulates state across the whole test suite.
2. Configurability: create_app() can accept a config object/dict later
   (e.g. a separate TestConfig) without duplicating app setup code.
3. This is the pattern Flask's own documentation recommends for any
   app expected to grow past a single-file script.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from app.predict import init_predictor  # noqa: E402
from app.routes import api_bp  # noqa: E402


def configure_logging() -> None:
    """
    Sets up root logging ONCE at app startup, using the same
    format/level as every other module in this project (config.py),
    so Flask's own logs and our ML pipeline's logs look identical in
    the same log stream (important when reading Docker/Render logs).
    """
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


def create_app() -> Flask:
    """Application factory: builds and returns a configured Flask app."""
    configure_logging()
    logger = logging.getLogger(__name__)

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # CORS enabled so the frontend (served from templates/ or a
    # separate static host) can call this API from the browser without
    # being blocked by same-origin policy.
    CORS(app)

    app.register_blueprint(api_bp)

    # Load the ML model NOW, at startup, not on the first request.
    # If the model file is missing/corrupted, the app fails to start
    # with a clear error in the logs — far easier to diagnose than a
    # 500 error appearing only when the first real user hits /predict.
    try:
        init_predictor()
    except Exception as e:
        logger.error(
            f"Model failed to load at startup: {e}. "
            f"The app will still start, but /predict and /health will "
            f"report the model as unavailable until this is fixed."
        )

    logger.info("Flask app created successfully.")
    return app


# Module-level app object — required by Gunicorn (see Dockerfile:
# CMD ["gunicorn", "app.app:app", ...]) since Gunicorn imports this
# module and looks for a WSGI-callable named `app`.
app = create_app()


if __name__ == "__main__":
    # Local development server only. In production, Gunicorn imports
    # `app` directly and this block never runs (see Dockerfile).
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
