"""
apps/users/views.py
"""
from __future__ import annotations

import logging

from allauth.account.internal.flows.email_verification import send_verification_email_for_user
from django.contrib.auth import get_user_model
from django.core.mail import BadHeaderError
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

logger = logging.getLogger(__name__)

from .serializers import RegisterSerializer, UserSerializer

User = get_user_model()


class RegisterView(APIView):
    permission_classes = [AllowAny]

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
            # SMTP misconfiguration or network failure — surface a clean 503
            logger.error("Failed to send verification email to %s: %s", request.user.email, e)
            return Response(
                {"detail": "Could not send verification email. Please try again later."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"detail": "Verification email sent."})
