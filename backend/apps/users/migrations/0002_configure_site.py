"""
Set the django.contrib.sites Site record from SITE_DOMAIN / SITE_NAME env vars.

allauth builds all absolute URLs in auth emails (password reset, confirmation)
using the domain stored in the Sites framework. The Django default is
"example.com", which makes every outbound link broken in production.

This migration runs on every deploy via Railway's releaseCommand
(`python manage.py migrate --noinput`) and is safe to re-run — it's an
upsert on the fixed SITE_ID=1 record.
"""
import os

from django.apps.registry import Apps
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor


def set_site_domain(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    Site = apps.get_model("sites", "Site")
    domain = os.environ.get("SITE_DOMAIN", "localhost")
    name = os.environ.get("SITE_NAME", "Sync-Safe")
    Site.objects.update_or_create(
        id=1,
        defaults={"domain": domain, "name": name},
    )


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0001_initial"),
        ("sites", "0002_alter_domain_unique"),
    ]

    operations = [
        migrations.RunPython(set_site_domain, migrations.RunPython.noop),
    ]
