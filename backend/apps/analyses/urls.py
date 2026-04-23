from django.urls import path

from .views import AnalysisDetailView, AnalysisLabelView, AnalysisListCreateView, TrackLabelListView

urlpatterns = [
    path("", AnalysisListCreateView.as_view(), name="analysis-list-create"),
    path("labels/", TrackLabelListView.as_view(), name="track-label-list"),
    path("<uuid:pk>/", AnalysisDetailView.as_view(), name="analysis-detail"),
    path("<uuid:pk>/label/", AnalysisLabelView.as_view(), name="analysis-label"),
]
