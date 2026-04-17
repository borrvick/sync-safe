"""
apps/users/adapters.py
Custom allauth adapters for our UUID-based, username-free User model.
"""
from __future__ import annotations

import logging
from typing import Any

from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.conf import settings
from django.db import IntegrityError
from django.http import HttpRequest

logger = logging.getLogger(__name__)


class AccountAdapter(DefaultAccountAdapter):
    """
    Removes the username field from allauth's signup flow.
    Our User model uses email as the sole identifier.
    """

    def save_user(
        self,
        request: HttpRequest,
        user: Any,
        form: Any,
        commit: bool = True,
    ) -> Any:
        user = super().save_user(request, user, form, commit=False)
        # Ensure username is never set — our model has no username field
        if hasattr(user, "username"):
            user.username = None  # type: ignore[assignment]
        if commit:
            try:
                user.save()
            except IntegrityError as e:
                # Duplicate email — can happen under race conditions or retried submissions
                logger.warning("AccountAdapter.save_user IntegrityError for email=%s: %s", user.email, e)
                raise
        return user


class SocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Handles OAuth user creation for GitHub and Google.

    allauth's default adapter tries to populate a username field that
    our User model doesn't have. This adapter skips that step and
    creates users via our custom UserManager (email + UUID PK).
    """

    def populate_user(
        self,
        request: HttpRequest,
        sociallogin: Any,
        data: dict[str, Any],
    ) -> Any:
        user = super().populate_user(request, sociallogin, data)
        # Clear any username allauth tried to set — not in our schema
        if hasattr(user, "username"):
            user.username = None  # type: ignore[assignment]
        return user

    def is_open_for_signup(self, request: HttpRequest, sociallogin: Any) -> bool:
        # Respect any future ACCOUNT_ALLOW_REGISTRATION setting
        return getattr(settings, "ACCOUNT_ALLOW_REGISTRATION", True)
