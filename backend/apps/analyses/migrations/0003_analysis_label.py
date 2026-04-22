from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analyses", "0002_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysis",
            name="label",
            field=models.CharField(blank=True, default="", max_length=100),
            preserve_default=False,
        ),
    ]
