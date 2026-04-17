from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    MeView,
    PasswordResetConfirmView,
    PasswordResetView,
    RegisterView,
    ResendVerificationView,
    ThrottledTokenObtainPairView,
)

urlpatterns = [
    path("register/", RegisterView.as_view(), name="auth-register"),
    path("login/", ThrottledTokenObtainPairView.as_view(), name="auth-login"),
    path("token/refresh/", TokenRefreshView.as_view(), name="auth-token-refresh"),
    path("me/", MeView.as_view(), name="auth-me"),
    path("resend-verification/", ResendVerificationView.as_view(), name="auth-resend-verification"),
    path("password/reset/", PasswordResetView.as_view(), name="auth-password-reset"),
    path("password/reset/confirm/", PasswordResetConfirmView.as_view(), name="auth-password-reset-confirm"),
]
