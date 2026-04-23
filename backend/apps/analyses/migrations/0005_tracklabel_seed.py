"""
Seed the TrackLabel table with the standard sync licensing categories.
These power the label dropdown in the frontend.
"""
from __future__ import annotations

from django.db import migrations

_LABELS = [
    ("tv-commercial",  "TV Commercial",            "Advertisements, spots, and branded content", 10),
    ("tv-drama",       "TV Drama / Documentary",   "Scripted series, reality TV, docs",           20),
    ("film-feature",   "Film (Feature)",           "Theatrical feature films",                    30),
    ("film-trailer",   "Film (Trailer / Promo)",   "Theatrical trailers and teasers",             40),
    ("podcast",        "Podcast & Streaming",      "Podcast intros/outros, streaming shows",      50),
    ("video-game",     "Video Game",               "In-game music, trailers, cutscenes",          60),
    ("corporate",      "Corporate & Brand",        "Internal videos, brand films, explainers",    70),
    ("social-media",   "Social Media & Online",    "Short-form content, viral clips, reels",      80),
    ("live-event",     "Live Event & Performance", "Concert films, live streams, events",         90),
]


def seed_labels(apps: object, schema_editor: object) -> None:
    TrackLabel = apps.get_model("analyses", "TrackLabel")  # type: ignore[attr-defined] — historical model proxy has no type stubs
    for slug, name, description, sort_order in _LABELS:
        TrackLabel.objects.get_or_create(
            slug=slug,
            defaults={"name": name, "description": description, "sort_order": sort_order},
        )


def unseed_labels(apps: object, schema_editor: object) -> None:
    TrackLabel = apps.get_model("analyses", "TrackLabel")  # type: ignore[attr-defined] — historical model proxy has no type stubs
    TrackLabel.objects.filter(slug__in=[row[0] for row in _LABELS]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("analyses", "0004_tracklabel"),
    ]

    operations = [
        migrations.RunPython(seed_labels, reverse_code=unseed_labels),
    ]
