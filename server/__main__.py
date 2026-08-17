"""Entry point so the backend can be started with ``python -m server``.

Replaces the old ``app.run(host='0.0.0.0', port=5000, debug=True)`` block.
That was unsafe in two ways: ``debug=True`` enables the Werkzeug interactive
debugger (arbitrary code execution for anyone who can reach it), and binding
``0.0.0.0`` exposed it to the whole network.

By default we now serve with **waitress**, a production-grade threaded WSGI
server, bound to loopback. One slow request no longer blocks every other
client, which was the behaviour of the single-threaded dev server.
"""

import argparse

from .app import app
from .config import settings
from .logging_config import get_logger, setup_logging

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


def main() -> None:
    """Parse CLI arguments and start the WSGI server."""
    parser = argparse.ArgumentParser(description="Run the FakeNewsAI backend.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1, loopback only).")
    parser.add_argument("--port", type=int, default=5000, help="Bind port.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Waitress worker threads.")
    parser.add_argument("--dev", action="store_true",
                        help="Use the Flask dev server with the reloader. "
                             "Never use this on a reachable interface.")
    args = parser.parse_args()

    if args.dev:
        # debug=True is deliberately NOT passed: the reloader is useful, the
        # interactive debugger is a remote code execution hazard.
        logger.warning("Starting Flask development server (single process).")
        app.run(host=args.host, port=args.port, debug=False,
                use_reloader=True, threaded=True)
        return

    try:
        from waitress import serve
    except ImportError:
        logger.error(
            "waitress is not installed. Run `pip install -r requirements.txt`, "
            "or start with --dev for the development server."
        )
        raise SystemExit(1)

    logger.info("Serving on http://%s:%d with %d threads",
                args.host, args.port, args.threads)
    serve(app, host=args.host, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
