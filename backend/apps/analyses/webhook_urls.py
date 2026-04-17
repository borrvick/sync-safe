from django.urls import path

from .webhook_views import AnalysisCompleteWebhookView

urlpatterns = [
    path("analysis-complete/", AnalysisCompleteWebhookView.as_view(), name="webhook-analysis-complete"),
]
