"""
apps/users/views.py
"""
from __future__ import annotations

import logging

from allauth.account.internal.flows.email_verification import send_verification_email_for_user
from django.contrib.auth import get_user_model
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import BadHeaderError, send_mail
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework_simplejwt.views import TokenObtainPairView

from .serializers import (
    PasswordResetConfirmSerializer,
    PasswordResetSerializer,
    RegisterSerializer,
    UserSerializer,
)
from .throttles import LoginRateThrottle, RegisterRateThrottle


class ThrottledTokenObtainPairView(TokenObtainPairView):
    """Adds LoginRateThrottle to simplejwt's login view without modifying the library."""
    throttle_classes = [LoginRateThrottle]

logger = logging.getLogger(__name__)
User = get_user_model()


class RegisterView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [RegisterRateThrottle]

    def post(self, request: Request) -> Response:
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        return Response(UserSerializer(request.user).data)


class ResendVerificationView(APIView):
    """
    POST /api/auth/resend-verification/
    Resends the email confirmation link for the authenticated user.
    Noop if the user's email is already verified.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request: Request) -> Response:
        try:
            send_verification_email_for_user(request, request.user)
        except (OSError, BadHeaderError) as e:
            logger.error("Failed to send verification email to %s: %s", request.user.email, e)
            return Response(
                {"detail": "Could not send verification email. Please try again later."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"detail": "Verification email sent."})


class PasswordResetView(APIView):
    """
    POST /api/auth/password/reset/
    Sends a password-reset email.

    Always returns 200 regardless of whether the email exists — this prevents
    account enumeration (an attacker cannot probe which emails are registered).
    """

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        serializer = PasswordResetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data["email"]

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            # Return 200 anyway — do not reveal whether the email exists
            return Response({"detail": "If that email is registered, a reset link has been sent."})

        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)
        reset_url = f"{request.build_absolute_uri('/api/auth/password/reset/confirm/')}?uid={uid}&token={token}"

        try:
            send_mail(
                subject="Reset your Sync-Safe password",
                message=f"Click the link to reset your password (expires in 1 hour):\n\n{reset_url}",
                from_email=None,  # uses DEFAULT_FROM_EMAIL from settings
                recipient_list=[email],
                fail_silently=False,
            )
        except (OSError, BadHeaderError) as e:
            logger.error("Failed to send password reset email to %s: %s", email, e)
            return Response(
                {"detail": "Could not send reset email. Please try again later."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response({"detail": "If that email is registered, a reset link has been sent."})


class PasswordResetConfirmView(APIView):
    """
    POST /api/auth/password/reset/confirm/
    Validates the reset token and sets the new password.
    Returns 400 on invalid/expired token — no detail about why (enumeration-proof).
    """

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        serializer = PasswordResetConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            uid = force_str(urlsafe_base64_decode(data["uid"]))
            user = User.objects.get(pk=uid)
        except (User.DoesNotExist, ValueError, TypeError):
            return Response({"detail": "Invalid or expired reset link."}, status=status.HTTP_400_BAD_REQUEST)

        if not default_token_generator.check_token(user, data["token"]):
            return Response({"detail": "Invalid or expired reset link."}, status=status.HTTP_400_BAD_REQUEST)

        user.set_password(data["new_password"])
        user.save(update_fields=["password"])
        logger.info("Password reset completed for user %s", user.pk)
        return Response({"detail": "Password has been reset."})
