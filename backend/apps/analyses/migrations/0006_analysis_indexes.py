"""
Add indexes for analysis query patterns:
- source_url: deduplication lookup in POST /api/analyses/
- (source_url, status): composite for the dedup filter on source_url+COMPLETE
- user: FK filter in for_user() queryset
"""
from __future__ import annotations

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("analyses", "0005_tracklabel_seed"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddIndex(
            model_name="analysis",
            index=models.Index(fields=["source_url"], name="analysis_source_url_idx"),
        ),
        migrations.AddIndex(
            model_name="analysis",
            index=models.Index(
                fields=["source_url", "status"],
                name="analysis_source_url_status_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="analysis",
            index=models.Index(fields=["user"], name="analysis_user_idx"),
        ),
    ]
