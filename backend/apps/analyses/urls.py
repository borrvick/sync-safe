from django.urls import path

from .views import AnalysisDetailView, AnalysisLabelView, AnalysisListCreateView

urlpatterns = [
    path("", AnalysisListCreateView.as_view(), name="analysis-list-create"),
    path("<uuid:pk>/", AnalysisDetailView.as_view(), name="analysis-detail"),
    path("<uuid:pk>/label/", AnalysisLabelView.as_view(), name="analysis-label"),
]
