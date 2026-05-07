"""
utils/security.py
Centralized security helpers: URL validation, shell arg sanitization, secret access.
All services should import from here rather than calling shlex/os.environ directly.
"""
import os
import re
import shlex
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Allowlisted hostnames for yt-dlp-routed downloads
#
# Must stay in sync with the platform host sets in services/ingestion/_pure.py.
# A test in tests/test_security.py enforces this — add new platforms there too.
# ---------------------------------------------------------------------------
_ALLOWED_HOSTS = frozenset({
    # YouTube
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "music.youtube.com",
    "m.youtube.com",
    # Bandcamp (subdomains matched separately below via endswith check)
    "bandcamp.com",
    # SoundCloud
    "soundcloud.com",
    "www.soundcloud.com",
    "on.soundcloud.com",
    # TikTok
    "tiktok.com",
    "www.tiktok.com",
    "vm.tiktok.com",
    "m.tiktok.com",
    # Instagram
    "instagram.com",
    "www.instagram.com",
    # Facebook
    "facebook.com",
    "www.facebook.com",
    "fb.com",
    "fb.watch",
})

# Control characters (excluding normal whitespace) and null bytes
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# URL Validation
# ---------------------------------------------------------------------------

def validate_url(url: str) -> str:
    """
    Validate that a URL is a legitimate HTTPS YouTube link.

    Args:
        url: Raw URL string from user input.

    Returns:
        The original URL string if valid.

    Raises:
        ValueError: with a safe, descriptive message if the URL is rejected.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")

    url = url.strip()

    if not url:
        raise ValueError("URL must not be empty.")

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError(
            f"Only HTTPS URLs are accepted (got scheme: '{parsed.scheme}')."
        )

    # Reject URLs with embedded credentials (user:pass@host)
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not accepted.")

    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS and not host.endswith(".bandcamp.com"):
        raise ValueError(
            f"URL host '{host}' is not on the allowlist. "
            f"Accepted hosts: {', '.join(sorted(_ALLOWED_HOSTS))} "
            f"(or any *.bandcamp.com subdomain)"
        )

    return url


# ---------------------------------------------------------------------------
# Shell Argument Sanitization
# ---------------------------------------------------------------------------

def sanitize_shell_arg(s: str) -> str:
    """
    Sanitize a string before it is passed as a shell argument.

    Strips null bytes and control characters, then applies shlex.quote()
    to produce a safely shell-escaped token.

    Args:
        s: Raw string (e.g. artist name, title) to sanitize.

    Returns:
        A shell-safe quoted string.
    """
    if not isinstance(s, str):
        s = str(s)

    # Remove null bytes and control characters
    s = _CONTROL_CHAR_RE.sub("", s)
    s = s.replace("\x00", "")

    return shlex.quote(s)


# ---------------------------------------------------------------------------
# Secret Access
# ---------------------------------------------------------------------------

def require_secret(key: str) -> str:
    """
    Retrieve a required secret from the environment.

    Tries os.environ first, then st.secrets (if running in Streamlit).
    Raises EnvironmentError with a descriptive message if absent.
    Never logs or surfaces the full secret value.

    Args:
        key: Environment variable / secret name.

    Returns:
        The secret value string.

    Raises:
        EnvironmentError: if the secret is not found.
    """
    # 1. Standard environment variable
    value = os.environ.get(key)

    # 2. Streamlit secrets (HF Spaces injects these at runtime)
    if value is None:
        try:
            import streamlit as st
            value = st.secrets.get(key)
        except Exception:
            pass

    if not value:
        raise EnvironmentError(
            f"Required secret '{key}' is not set. "
            f"Add it to your Hugging Face Space Secrets or local .env file."
        )

    return value


def redact_secret(value: str) -> str:
    """
    Return a redacted representation of a secret for safe logging.
    e.g. "abc123xyz" → "***...xyz"
    """
    if len(value) <= 4:
        return "***"
    return f"***...{value[-4:]}"
