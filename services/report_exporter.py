"""
services/report_exporter.py

ReportExporter — maps an AnalysisResult to a flat TrackReport and
serialises it to a UTF-8 CSV bytes payload.

Design:
- build() is a pure function: no I/O, no side effects, deterministic output.
- to_csv() uses csv.DictWriter; JSON blob columns are written as escaped strings.
- Both methods are intentionally thin — all domain logic lives in core/report.py.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from core.models import AnalysisResult
from core.report import TrackReport, _dumps, _track_id


class ReportExporter:
    """Converts a completed AnalysisResult into a storable TrackReport."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, result: AnalysisResult) -> TrackReport:
        """
        Map every field from result into a flat TrackReport.

        Pure function — no I/O.  All Optional fields default to None when the
        corresponding pipeline stage did not run or returned no data.

        NOTE: this function intentionally exceeds the 40-line guideline.
        It is a flat mapping (no branching logic, no loops) — splitting it
        into sub-functions would create artificial seams with no cohesion.
        Cyclomatic complexity is 1.
        """
        audio       = result.audio
        structure   = result.structure
        forensics   = result.forensics
        compliance  = result.compliance
        authorship  = result.authorship
        quality     = result.audio_quality
        popularity  = result.popularity
        legal       = result.legal
        stems       = result.stem_validation
        meta_val    = result.metadata_validation
        theme_mood  = result.theme_mood

        title  = audio.metadata.get("title", "")
        artist = audio.metadata.get("artist", "")

        # Derive duration from the last beat or last section end
        duration = 0.0
        if structure:
            if structure.beats:
                duration = float(structure.beats[-1])
            elif structure.sections:
                duration = float(structure.sections[-1].end)

        # ── Structure counts ──────────────────────────────────────────
        sections      = structure.sections if structure else []
        intro_s       = sum(s.end - s.start for s in sections if s.label == "intro")
        verse_count   = sum(1 for s in sections if s.label == "verse")
        chorus_count  = sum(1 for s in sections if s.label == "chorus")
        bridge_count  = sum(1 for s in sections if s.label == "bridge")
        bpm: float | None = None
        if structure and isinstance(structure.bpm, (int, float)):
            bpm = float(structure.bpm)

        # ── Forensics ─────────────────────────────────────────────────
        f_flags: list[str] = []
        f_notes: list[str] = []
        if forensics:
            f_flags = list(forensics.flags)
            f_notes = list(forensics.forensic_notes)

        # ── Compliance ────────────────────────────────────────────────
        c_flags       = compliance.flags if compliance else []
        confirmed     = [f for f in c_flags if f.confidence == "confirmed"]
        potential     = [f for f in c_flags if f.confidence == "potential"]
        hard          = [f for f in c_flags if f.confidence == "confirmed" and f.severity == "hard"]
        soft          = [f for f in c_flags if f.confidence == "confirmed" and f.severity == "soft"]

        # ── Authorship scores dict ────────────────────────────────────
        a_scores = authorship.scores if authorship else {}

        return TrackReport(
            # Identity
            track_id        = _track_id(title, artist, duration),
            scan_timestamp  = datetime.now(timezone.utc).isoformat(),

            # Audio / Ingestion
            title           = title,
            artist          = artist,
            source          = audio.source,
            sample_rate     = audio.sample_rate,
            yt_view_count   = audio.engagement.get("view_count", 0),
            yt_like_count   = audio.engagement.get("like_count", 0),

            # Structure
            bpm             = bpm,
            key             = structure.key if structure else "",
            section_count   = len(sections),
            beat_count      = len(structure.beats) if structure else 0,
            duration_s      = duration,
            intro_s         = intro_s,
            verse_count     = verse_count,
            chorus_count    = chorus_count,
            bridge_count    = bridge_count,

            # Forensics — Verdict
            forensic_verdict    = forensics.verdict if forensics else "",
            ai_probability      = forensics.ai_probability if forensics else 0.0,
            forensic_flag_count = len(f_flags),
            c2pa_flag           = forensics.c2pa_flag if forensics else False,
            c2pa_origin         = forensics.c2pa_origin if forensics else "",
            is_vocal            = forensics.is_vocal if forensics else False,

            # Forensics — Raw Signals
            ibi_variance                = forensics.ibi_variance if forensics else 1.0,
            loop_score                  = forensics.loop_score if forensics else 0.0,
            loop_autocorr_score         = forensics.loop_autocorr_score if forensics else 0.0,
            repetition_index            = forensics.repetition_index if forensics else None,
            spectral_slop               = forensics.spectral_slop if forensics else 0.0,
            synthid_score               = forensics.synthid_score if forensics else 0.0,
            centroid_instability_score  = forensics.centroid_instability_score if forensics else -1.0,
            harmonic_ratio_score        = forensics.harmonic_ratio_score if forensics else -1.0,
            kurtosis_variability        = forensics.kurtosis_variability if forensics else -1.0,
            decoder_peak_score          = forensics.decoder_peak_score if forensics else 0.0,
            spectral_centroid_mean      = forensics.spectral_centroid_mean if forensics else -1.0,
            self_similarity_entropy     = forensics.self_similarity_entropy if forensics else -1.0,
            noise_floor_ratio           = forensics.noise_floor_ratio if forensics else -1.0,
            onset_strength_cv           = forensics.onset_strength_cv if forensics else -1.0,
            spectral_flatness_var       = forensics.spectral_flatness_var if forensics else -1.0,
            subbeat_grid_deviation      = forensics.subbeat_grid_deviation if forensics else -1.0,
            pitch_quantization_score    = forensics.pitch_quantization_score if forensics else -1.0,
            ultrasonic_noise_ratio      = forensics.ultrasonic_noise_ratio if forensics else -1.0,
            infrasonic_energy_ratio     = forensics.infrasonic_energy_ratio if forensics else -1.0,
            phase_coherence_differential = forensics.phase_coherence_differential if forensics else -1.0,
            plr_std                     = forensics.plr_std if forensics else -1.0,
            voiced_noise_floor          = forensics.voiced_noise_floor if forensics else -1.0,

            # Audio Quality
            integrated_lufs     = quality.integrated_lufs if quality else None,
            true_peak_dbfs      = quality.true_peak_dbfs if quality else None,
            loudness_range_lu   = quality.loudness_range_lu if quality else None,
            delta_spotify       = quality.delta_spotify if quality else None,
            delta_apple_music   = quality.delta_apple_music if quality else None,
            delta_youtube       = quality.delta_youtube if quality else None,
            delta_broadcast     = quality.delta_broadcast if quality else None,
            true_peak_warning   = quality.true_peak_warning if quality else None,
            dialogue_score      = quality.dialogue_score if quality else None,
            dialogue_label      = quality.dialogue_label if quality else "",

            # Stem Validation
            mono_compatible     = stems.mono_compatible if stems else None,
            phase_correlation   = stems.phase_correlation if stems else None,
            cancellation_db     = stems.cancellation_db if stems else None,
            mid_side_ratio      = stems.mid_side_ratio if stems else None,
            stem_flag_count     = len(stems.flags) if stems else 0,

            # Compliance
            compliance_grade        = compliance.grade if compliance else "",
            total_flag_count        = len(c_flags),
            confirmed_flag_count    = len(confirmed),
            potential_flag_count    = len(potential),
            hard_flag_count         = len(hard),
            soft_flag_count         = len(soft),
            sting_flag              = compliance.sting.flag if compliance else None,
            sting_ending_type       = compliance.sting.ending_type if compliance else "",
            sting_final_energy_ratio = compliance.sting.final_energy_ratio if compliance else None,
            energy_evolution_flag   = compliance.evolution.flag if compliance else None,
            stagnant_windows        = compliance.evolution.stagnant_windows if compliance else 0,
            total_windows           = compliance.evolution.total_windows if compliance else 0,
            intro_flag              = compliance.intro.flag if compliance else None,
            intro_seconds           = compliance.intro.intro_seconds if compliance else 0.0,
            intro_source            = compliance.intro.source if compliance else "",

            # Authorship
            authorship_verdict      = authorship.verdict if authorship else "",
            authorship_signal_count = authorship.signal_count if authorship else 0,
            roberta_score           = authorship.roberta_score if authorship else None,
            burstiness_score        = a_scores.get("burstiness"),
            unique_word_ratio       = a_scores.get("unique_word_ratio"),
            rhyme_density           = a_scores.get("rhyme_density"),
            repetition_score        = a_scores.get("repetition_score"),

            # Theme & Mood
            mood                = theme_mood.mood if theme_mood else "",
            theme_confidence    = theme_mood.confidence if theme_mood else 0.0,
            groq_enriched       = theme_mood.groq_enriched if theme_mood else False,

            # Popularity & Cost
            popularity_score    = popularity.popularity_score if popularity else None,
            popularity_tier     = popularity.tier if popularity else "",
            lastfm_listeners    = popularity.listeners if popularity else 0,
            lastfm_playcount    = popularity.playcount if popularity else 0,
            spotify_score       = popularity.spotify_score if popularity else None,
            sync_cost_low       = popularity.sync_cost_low if popularity else None,
            sync_cost_high      = popularity.sync_cost_high if popularity else None,

            # Legal
            isrc        = legal.isrc if legal else None,
            pro_match   = legal.pro_match if legal else None,

            # Metadata Validation
            metadata_valid      = meta_val.valid if meta_val else None,
            missing_fields_count = len(meta_val.missing_fields) if meta_val else 0,
            split_total         = meta_val.split_total if meta_val else None,
            split_error         = meta_val.split_error if meta_val else None,
            isrc_valid          = meta_val.isrc_valid if meta_val else None,

            # JSON Blobs
            compliance_flags_json   = _dumps(c_flags),
            ai_segments_json        = _dumps(forensics.ai_segments if forensics else []),
            sections_json           = _dumps(sections),
            transcript_json         = _dumps(result.transcript),
            similar_tracks_json     = _dumps(result.similar_tracks),
            sync_cuts_json          = _dumps(result.sync_cuts),
            forensic_notes_json     = _dumps(f_notes),
            forensic_flags_json     = _dumps(f_flags),
            themes_json             = _dumps(theme_mood.themes if theme_mood else []),
            theme_keywords_json     = _dumps(theme_mood.raw_keywords if theme_mood else []),
        )

    def to_csv(self, report: TrackReport) -> bytes:
        """
        Serialise a TrackReport to a UTF-8 CSV bytes payload.

        Returns raw bytes suitable for st.download_button(data=...).
        """
        row = report.model_dump()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        return buf.getvalue().encode("utf-8")
