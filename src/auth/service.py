"""Authentication middleware for JWT token validation."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from ...config.settings import settings
from ...utils import logger


class AuthMiddleware:
    """Middleware for handling authentication."""

    def __init__(self, jwt_secret: str = None):
        """
        Initialize auth middleware.

        Args:
            jwt_secret: JWT secret key (defaults to settings)
        """
        self.jwt_secret = jwt_secret or settings.jwt_secret
        self.jwt_algorithm = settings.jwt_algorithm

    def generate_jwt(self, user_id: int, token_usage="access") -> str:
        """
        Create an access or refresh token for the user.

        Args:
            user_id: User ID to create token for
            token_usage: Token usage ('access' or 'refresh')

        Returns:
            JWT token
        """
        logger.info("Generating %s token..", token_usage)
        if token_usage == "access":
            expires_delta = timedelta(minutes=60)
        elif token_usage == "refresh":
            expires_delta = timedelta(days=7)
        else:
            raise ValueError("Invalid token usage")

        payload = {
            "sub": str(user_id),
            "exp": datetime.now(timezone.utc) + expires_delta,
            "token_usage": token_usage,
        }

        encoded_jwt = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        return encoded_jwt

    def get_current_refresh_user(
        self,
        token: Annotated[str, Depends(OAuth2PasswordBearer(tokenUrl="/login"))],
    ):
        """Get the current user from the refresh token."""
        try:
            logger.info("Decoding refresh token..")
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            user_id: int = int(payload.get("sub"))
            if user_id is None:
                raise JWTError("User ID is None in payload")
            if payload.get("token_usage") != "refresh":
                raise JWTError("Invalid Token usage. Refresh token expected")
            return {"user_id": user_id}
        except JWTError as e:
            logger.exception(e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate user",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    def get_current_user(
        self,
        token: Annotated[str, Depends(OAuth2PasswordBearer(tokenUrl="/login"))],
    ):
        """Get the current user from the token."""
        try:
            logger.info("Decoding access token..")
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            user_id: int = int(payload.get("sub"))
            if user_id is None:
                raise JWTError("User ID is None in payload")
            if payload.get("token_usage") != "access":
                raise JWTError("Invalid Token usage. access token expected")
            return user_id
        except JWTError as e:
            logger.exception(e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate user",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e
