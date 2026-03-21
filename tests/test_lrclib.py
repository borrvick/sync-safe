"""
tests/test_lrclib.py
Smoke-tests for LRCLib lyrics lookup and LRC parsing.

Run from the project root:
    python tests/test_lrclib.py

No test framework required — exits 0 on pass, 1 on any failure.
The LRCLib tests hit the live API; skip them with --offline flag.
"""
from __future__ import annotations

from services.lyrics_provider import LRCLibClient, _best_result, _parse_lrc

# ---------------------------------------------------------------------------
# Known-good test case (live API)
# ---------------------------------------------------------------------------

TITLE  = "Espresso"
ARTIST = "Sabrina Carpenter"

# Phrases that must appear in the lyrics if the lookup is working correctly
EXPECTED_PHRASES = [
    "that's that me",   # LRCLib has "That's that me, espresso" — match the stable prefix
    "say you can't sleep",
    "i'm working late",
]


def test_parse_lrc_pure() -> None:
    """_parse_lrc correctly converts LRC lines to TranscriptSegments."""
    lrc = (
        "[00:05.12]First line\n"
        "[00:10.48]Second line\n"
        "[00:15.00]Third line\n"
    )
    segments = _parse_lrc(lrc)
    assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
    assert segments[0].start == 5.12
    assert segments[0].text == "First line"
    assert segments[0].end == segments[1].start
    assert segments[1].text == "Second line"
    # Last segment gets the fixed tail
    assert segments[2].end == segments[2].start + 5.0
    print("  ✓ _parse_lrc: correct segment count, timestamps, and text")


def test_parse_lrc_skips_empty_lines() -> None:
    """_parse_lrc skips instrumental markers (lines with no text after timestamp)."""
    lrc = (
        "[00:01.00]Lyrics here\n"
        "[00:05.00]\n"           # instrumental marker — no text
        "[00:09.00]More lyrics\n"
    )
    segments = _parse_lrc(lrc)
    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
    print("  ✓ _parse_lrc: instrumental markers correctly skipped")


def test_parse_lrc_three_digit_centiseconds() -> None:
    """_parse_lrc handles 3-digit fractional seconds (milliseconds)."""
    lrc = "[00:12.345]Hello world\n"
    segments = _parse_lrc(lrc)
    assert len(segments) == 1
    assert segments[0].start == round(12 + 0.345, 2)
    print("  ✓ _parse_lrc: 3-digit fractional seconds handled correctly")


def test_best_result_prefers_synced() -> None:
    """_best_result selects the first entry with syncedLyrics."""
    results = [
        {"id": 1, "syncedLyrics": None, "plainLyrics": "some lyrics"},
        {"id": 2, "syncedLyrics": "[00:01.00]synced line"},
        {"id": 3, "syncedLyrics": "[00:02.00]another"},
    ]
    best = _best_result(results)
    assert best is not None
    assert best["id"] == 2
    print("  ✓ _best_result: correctly selects first entry with syncedLyrics")


def test_best_result_empty() -> None:
    """_best_result returns None on an empty list."""
    assert _best_result([]) is None
    print("  ✓ _best_result: returns None for empty results")


def test_lrclib_live(offline: bool = False) -> None:
    """LRCLibClient.get_lyrics returns correct lyrics for Espresso (live API)."""
    if offline:
        print("  ~ test_lrclib_live: SKIPPED (--offline)")
        return

    client = LRCLibClient()
    print(f"  Querying LRCLib: '{TITLE}' by {ARTIST}...")
    segments = client.get_lyrics(TITLE, ARTIST)

    assert segments is not None, (
        "LRCLib returned None — track not found or API unreachable. "
        "Re-run with --offline to skip network tests."
    )
    assert len(segments) > 0, "LRCLib returned an empty segment list"

    full_text = " ".join(s.text for s in segments).lower()
    for phrase in EXPECTED_PHRASES:
        assert phrase in full_text, (
            f"Expected phrase not found in lyrics: '{phrase}'\n"
            f"First 5 segments:\n" +
            "\n".join(f"  [{s.start:.2f}s] {s.text}" for s in segments[:5])
        )

    print(f"  ✓ LRCLib live: {len(segments)} segments, all expected phrases found")
    print("  First 5 segments:")
    for seg in segments[:5]:
        print(f"    [{seg.start:6.2f}s] {seg.text}")


def test_lrclib_missing_title() -> None:
    """LRCLibClient.get_lyrics returns None when title/artist are blank."""
    client = LRCLibClient()
    result = client.get_lyrics("", "Sabrina Carpenter")
    assert result is None, "Expected None for blank title"
    result2 = client.get_lyrics("Espresso", "")
    assert result2 is None, "Expected None for blank artist"
    print("  ✓ LRCLibClient: correctly returns None for blank title/artist")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    offline = "--offline" in sys.argv

    tests = [
        test_parse_lrc_pure,
        test_parse_lrc_skips_empty_lines,
        test_parse_lrc_three_digit_centiseconds,
        test_best_result_prefers_synced,
        test_best_result_empty,
        test_lrclib_missing_title,
        lambda: test_lrclib_live(offline=offline),
    ]

    passed = failed = 0
    for test_fn in tests:
        name = getattr(test_fn, "__name__", "anonymous")
        try:
            test_fn()
            passed += 1
        except AssertionError as exc:
            print(f"  ✗ {name}: FAILED — {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name}: ERROR — {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
