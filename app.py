"""Compatibility Streamlit entrypoint.

This project uses `streamlit_app.py` as the canonical UI. Keeping `app.py`
allows existing deployment configs to continue working.
"""

from streamlit_app import *  # noqa: F401,F403
