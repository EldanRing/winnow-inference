"""Authentication for local checks; credentials never enter saved requests."""

import os
from pathlib import Path


def api_key():
    path = os.environ.get("WINNOW_API_KEY_FILE")
    key = Path(path).read_text().strip() if path else os.environ.get("WINNOW_API_KEY", "")
    if "\n" in key or "\r" in key:
        raise ValueError("Provide one Winnow API key")
    return key


def request_headers():
    headers = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        headers["Authorization"] = "Bearer " + key
    return headers
