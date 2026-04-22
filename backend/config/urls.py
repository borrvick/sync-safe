from django.contrib import admin
from django.urls import include, path

from core.views import HealthView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("admin/", admin.site.urls),
    path("api/auth/", include("apps.users.urls")),
    path("api/analyses/", include("apps.analyses.urls")),
    path("api/webhooks/", include("apps.analyses.webhook_urls")),
    path("accounts/", include("allauth.urls")),
]
