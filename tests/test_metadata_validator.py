"""
tests/test_metadata_validator.py
Unit tests for services/metadata_validator.py pure functions.
"""
from __future__ import annotations

import pytest

from services.metadata_validator import (
    MetadataValidator,
    validate_isrc,
    validate_splits,
)


# ---------------------------------------------------------------------------
# validate_isrc
# ---------------------------------------------------------------------------

class TestValidateIsrc:
    def test_valid_with_dashes(self) -> None:
        assert validate_isrc("US-ABC-23-12345") is True

    def test_valid_without_dashes(self) -> None:
        assert validate_isrc("USABC2312345") is True

    def test_valid_lowercase_normalised(self) -> None:
        assert validate_isrc("us-abc-23-12345") is True

    def test_invalid_too_short(self) -> None:
        assert validate_isrc("USABC231234") is False

    def test_invalid_too_long(self) -> None:
        assert validate_isrc("USABC23123456") is False

    def test_invalid_bad_country_prefix(self) -> None:
        # Country code must be 2 letters; digit not allowed in first 2 chars
        assert validate_isrc("1SABC2312345") is False

    def test_empty_string(self) -> None:
        assert validate_isrc("") is False

    def test_valid_alphanumeric_registrant(self) -> None:
        # Registrant (positions 3-5) may contain digits per ISO 3901
        assert validate_isrc("GB3JK2312345") is True


# ---------------------------------------------------------------------------
# validate_splits
# ---------------------------------------------------------------------------

class TestValidateSplits:
    def test_exact_100(self) -> None:
        assert validate_splits([50.0, 50.0]) is True

    def test_three_writers_rounding(self) -> None:
        # 33.33 + 33.33 + 33.34 = 100.00 — within tolerance
        assert validate_splits([33.33, 33.33, 33.34]) is True

    def test_under_100(self) -> None:
        assert validate_splits([40.0, 40.0]) is False

    def test_over_100(self) -> None:
        assert validate_splits([60.0, 50.0]) is False

    def test_empty_list(self) -> None:
        assert validate_splits([]) is False

    def test_single_writer_100(self) -> None:
        assert validate_splits([100.0]) is True

    def test_within_tolerance(self) -> None:
        # 0.005 deviation — within SPLIT_TOLERANCE of 0.01
        assert validate_splits([50.005, 49.995]) is True

    def test_exceeds_tolerance(self) -> None:
        # 0.05 deviation (50.03 + 49.92 = 99.95) — outside SPLIT_TOLERANCE of 0.01
        assert validate_splits([50.03, 49.92]) is False


# ---------------------------------------------------------------------------
# MetadataValidator.validate
# ---------------------------------------------------------------------------

_FULL_FIELDS: dict[str, str] = {
    "title":     "Blinding Lights",
    "artist":    "The Weeknd",
    "pro":       "ASCAP (US)",
    "publisher": "Warner Chappell",
}


class TestMetadataValidator:
    def test_valid_complete(self) -> None:
        result = MetadataValidator().validate(
            fields=_FULL_FIELDS,
            splits=[50.0, 50.0],
            isrc="US-ABC-23-12345",
        )
        assert result.valid is True
        assert result.rejection_reason is None
        assert result.missing_fields == []
        assert result.isrc_valid is True
        assert result.split_total == 100.0

    def test_missing_required_field(self) -> None:
        fields = dict(_FULL_FIELDS)
        fields["publisher"] = ""
        result = MetadataValidator().validate(fields=fields, splits=[100.0])
        assert result.valid is False
        assert "publisher" in result.missing_fields

    def test_invalid_isrc_fails(self) -> None:
        result = MetadataValidator().validate(
            fields=_FULL_FIELDS,
            splits=[100.0],
            isrc="NOTANISRC",
        )
        assert result.valid is False
        assert result.isrc_valid is False
        assert "ISRC" in (result.rejection_reason or "")

    def test_bad_splits_fails(self) -> None:
        result = MetadataValidator().validate(
            fields=_FULL_FIELDS,
            splits=[40.0, 40.0],
        )
        assert result.valid is False
        assert "splits" in (result.rejection_reason or "").lower()

    def test_no_isrc_provided_passes_isrc_check(self) -> None:
        # ISRC is not required — blank is valid
        result = MetadataValidator().validate(
            fields=_FULL_FIELDS,
            splits=[100.0],
            isrc="",
        )
        assert result.isrc_valid is True

    def test_no_splits_provided_passes_splits_check(self) -> None:
        # Splits are optional when no writers entered
        result = MetadataValidator().validate(
            fields=_FULL_FIELDS,
            splits=[],
            isrc="",
        )
        assert result.valid is True
        assert result.split_error is None

    def test_split_error_magnitude(self) -> None:
        result = MetadataValidator().validate(
            fields=_FULL_FIELDS,
            splits=[60.0, 30.0],  # sum = 90 → error = 10.0
        )
        assert result.split_error is not None
        assert abs(result.split_error - 10.0) < 0.01

    def test_multiple_errors_combined_in_rejection_reason(self) -> None:
        fields = {k: "" for k in _FULL_FIELDS}
        result = MetadataValidator().validate(
            fields=fields,
            splits=[40.0],
            isrc="BAD",
        )
        assert result.valid is False
        reason = result.rejection_reason or ""
        assert "Missing" in reason
        assert "ISRC" in reason
        assert "splits" in reason.lower()
