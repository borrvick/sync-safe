"""
tests/test_security.py
Unit tests for utils/security.py
"""
import os
import pytest
from unittest.mock import patch

from utils.security import validate_url, sanitize_shell_arg, require_secret, redact_secret


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------

class TestValidateUrl:
    def test_valid_youtube_watch(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_url(url) == url

    def test_valid_youtu_be(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert validate_url(url) == url

    def test_valid_music_youtube(self):
        url = "https://music.youtube.com/watch?v=abc123"
        assert validate_url(url) == url

    def test_valid_mobile_youtube(self):
        url = "https://m.youtube.com/watch?v=abc123"
        assert validate_url(url) == url

    def test_rejects_http(self):
        with pytest.raises(ValueError, match="HTTPS"):
            validate_url("http://www.youtube.com/watch?v=abc")

    def test_rejects_non_youtube_host(self):
        with pytest.raises(ValueError, match="allowlist"):
            validate_url("https://evil.com/watch?v=abc")

    def test_rejects_ssrf_internal(self):
        with pytest.raises(ValueError, match="allowlist"):
            validate_url("https://169.254.169.254/latest/meta-data/")

    def test_rejects_embedded_credentials(self):
        with pytest.raises(ValueError, match="credentials"):
            validate_url("https://user:pass@www.youtube.com/watch?v=abc")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="empty"):
            validate_url("")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="string"):
            validate_url(None)

    def test_strips_leading_trailing_whitespace(self):
        url = "  https://www.youtube.com/watch?v=abc  "
        result = validate_url(url)
        assert result == url.strip()


# ---------------------------------------------------------------------------
# sanitize_shell_arg
# ---------------------------------------------------------------------------

class TestSanitizeShellArg:
    def test_simple_string(self):
        result = sanitize_shell_arg("hello world")
        assert "hello world" in result
        assert result.startswith("'") or result.startswith('"')

    def test_strips_null_bytes(self):
        result = sanitize_shell_arg("hello\x00world")
        assert "\x00" not in result

    def test_strips_control_characters(self):
        result = sanitize_shell_arg("hello\x01\x07world")
        assert "\x01" not in result
        assert "\x07" not in result

    def test_shell_metacharacters_are_quoted(self):
        result = sanitize_shell_arg("$(rm -rf /)")
        # shlex.quote wraps in single quotes, neutralising the $()
        assert "$" not in result or result.startswith("'")

    def test_semicolon_injection(self):
        # shlex.quote wraps the entire string in single quotes, neutralising ;
        result = sanitize_shell_arg("track; rm -rf /")
        assert result.startswith("'") and result.endswith("'")

    def test_non_string_coerced(self):
        result = sanitize_shell_arg(12345)
        assert "12345" in result

    def test_empty_string(self):
        result = sanitize_shell_arg("")
        assert result == "''"


# ---------------------------------------------------------------------------
# require_secret
# ---------------------------------------------------------------------------

class TestRequireSecret:
    def test_returns_value_from_environ(self):
        with patch.dict(os.environ, {"TEST_KEY": "supersecret"}):
            assert require_secret("TEST_KEY") == "supersecret"

    def test_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="TEST_KEY"):
                require_secret("TEST_KEY")

    def test_error_message_is_descriptive(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="Hugging Face"):
                require_secret("MISSING_KEY")

    def test_prefers_environ_over_st_secrets(self):
        with patch.dict(os.environ, {"LASTFM_API_KEY": "env_value"}):
            assert require_secret("LASTFM_API_KEY") == "env_value"


# ---------------------------------------------------------------------------
# redact_secret
# ---------------------------------------------------------------------------

class TestRedactSecret:
    def test_redacts_long_secret(self):
        result = redact_secret("27c96c62f19ae5abe6608e83bad25cee")
        assert result == "***...5cee"
        assert "27c96c62" not in result

    def test_redacts_short_secret(self):
        assert redact_secret("abc") == "***"

    def test_redacts_exactly_four_chars(self):
        assert redact_secret("abcd") == "***"
