"""Commit the forecast ledger to GitHub via the REST contents API.

ISO-NE blocks GitHub Actions' IPs, so the ledger cannot be refreshed in CI — but the deployed
Streamlit app *can* reach ISO-NE. This module lets the app persist each day's locked forecast by
committing the CSV straight to the repo with a fine-grained token, so the git history keeps its
role as tamper-evident proof (each commit's diff shows exactly which forecast rows were added,
timestamped by GitHub).

The token is read from Streamlit secrets / env (never hard-coded) and needs only
``contents: read/write`` on this one repo.
"""

from __future__ import annotations

import base64
import logging

import requests

logger = logging.getLogger(__name__)

_API = "https://api.github.com"
_AUTHOR_NAME = "Arun Surendranath"
_AUTHOR_EMAIL = "53822804+arunnath011@users.noreply.github.com"


def commit_text_file(
    path: str,
    text: str,
    message: str,
    *,
    token: str,
    repo: str,
    branch: str = "main",
    retries: int = 2,
) -> bool:
    """Create/update *path* on *repo* with *text* via the contents API. Returns True on success.

    Uses the file's current blob SHA for optimistic concurrency; on a 409 (another writer landed
    first) it re-reads and retries, so concurrent app sessions can't clobber each other.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"{_API}/repos/{repo}/contents/{path}"
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")

    for attempt in range(retries + 1):
        try:
            head = requests.get(url, headers=headers, params={"ref": branch}, timeout=30)
            sha = head.json().get("sha") if head.status_code == 200 else None

            payload: dict = {
                "message": message,
                "content": encoded,
                "branch": branch,
                "committer": {"name": _AUTHOR_NAME, "email": _AUTHOR_EMAIL},
                "author": {"name": _AUTHOR_NAME, "email": _AUTHOR_EMAIL},
            }
            if sha:
                payload["sha"] = sha

            put = requests.put(url, headers=headers, json=payload, timeout=30)
            if put.status_code in (200, 201):
                logger.info("Committed ledger to %s/%s.", repo, path)
                return True
            if put.status_code == 409:  # SHA race — another session committed first
                logger.warning("Ledger commit conflict (attempt %d); retrying.", attempt + 1)
                continue
            logger.warning("Ledger commit failed (%d): %s", put.status_code, put.text[:200])
            return False
        except requests.RequestException as exc:
            logger.warning("Ledger commit request error (attempt %d): %s", attempt + 1, exc)

    logger.warning("Ledger commit gave up after %d attempts.", retries + 1)
    return False
