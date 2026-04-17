from django.urls import path

from .views import AnalysisDetailView, AnalysisListCreateView

urlpatterns = [
    path("", AnalysisListCreateView.as_view(), name="analysis-list-create"),
    path("<uuid:pk>/", AnalysisDetailView.as_view(), name="analysis-detail"),
]
