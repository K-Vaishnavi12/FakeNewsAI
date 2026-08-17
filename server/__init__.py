"""FakeNewsAI backend package.

Run the HTTP API with::

    python -m server            # waitress, loopback:5000
    python -m server --dev      # Flask dev server

Run a one-off analysis from the terminal with::

    python -m server.cli --prompt "some claim"
"""

__all__ = ["app", "agent", "cli", "config", "constants", "ml_model"]
