"""
services/metadata_validator.py
Pre-flight track metadata validation for sync intake.

Validates:
- Required fields (title, artist, PRO, publisher)
- ISRC format (CC-XXX-YY-NNNNN — ISO 3901)
- Writer/publisher split sheet (splits must sum to 100 %)

Design:
- All core logic is pure functions — no I/O, no Streamlit imports.
- MetadataValidator is stateless; instantiate per call.
- Returns MetadataValidationResult even on partial data so the UI can
  show granular field-level feedback rather than a single pass/fail.
"""
from __future__ import annotations

from typing import Optional

from core.models import MetadataValidationResult

from ._pure import validate_isrc, validate_splits

# Required string fields in the intake form
_REQUIRED_FIELDS: list[str] = ["title", "artist", "pro", "publisher"]


def _check_required(fields: dict[str, str]) -> list[str]:
    """Return a list of required field names that are missing or blank."""
    return [f for f in _REQUIRED_FIELDS if not fields.get(f, "").strip()]


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class MetadataValidator:
    """
    Validates track rights metadata submitted at intake.

    Usage:
        result = MetadataValidator().validate(fields, splits, isrc)
    """

    def validate(
        self,
        fields: dict[str, str],
        splits: list[float],
        isrc: str = "",
    ) -> MetadataValidationResult:
        """
        Validate intake metadata and return a structured result.

        Args:
            fields: Mapping of field names to string values.
                    Expected keys: "title", "artist", "pro", "publisher".
            splits: List of writer/publisher percentage splits that must
                    sum to 100.0 (± CONSTANTS.SPLIT_TOLERANCE).
            isrc:   ISRC string; empty string means not provided.

        Returns:
            MetadataValidationResult with field-level detail.
        """
        missing = _check_required(fields)

        isrc_provided = bool(isrc.strip())
        isrc_valid    = validate_isrc(isrc) if isrc_provided else True  # not required unless provided

        split_total = round(float(sum(splits)), 4) if splits else 0.0
        splits_ok   = validate_splits(splits) if splits else True       # not required unless provided

        split_error: Optional[float] = (
            round(abs(split_total - 100.0), 4) if splits else None
        )

        # Rejection: hard-required fields missing OR splits provided but wrong
        reasons: list[str] = []
        if missing:
            reasons.append(f"Missing required fields: {', '.join(missing)}")
        if isrc_provided and not isrc_valid:
            reasons.append(f"Invalid ISRC format: '{isrc}'")
        if splits and not splits_ok:
            reasons.append(
                f"Writer splits sum to {split_total:.2f}% (must equal 100%)"
            )

        rejection_reason = "; ".join(reasons) if reasons else None

        return MetadataValidationResult(
            valid=rejection_reason is None,
            missing_fields=missing,
            split_total=split_total,
            split_error=split_error,
            isrc_valid=isrc_valid,
            rejection_reason=rejection_reason,
        )
