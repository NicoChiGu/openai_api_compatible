from collections.abc import Mapping
from typing import Optional

import openai
from httpx import Timeout

from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)


DEFAULT_OPENAI_TOTAL_TIMEOUT = 315.0
DEFAULT_OPENAI_READ_TIMEOUT = 300.0
DEFAULT_OPENAI_WRITE_TIMEOUT = 10.0
DEFAULT_OPENAI_CONNECT_TIMEOUT = 5.0
DEFAULT_VALIDATE_CONNECT_TIMEOUT = 10.0


def resolve_read_timeout(credentials: Mapping, default: float = DEFAULT_OPENAI_READ_TIMEOUT) -> Optional[float]:
    """
    Parse read_timeout from credentials.

    `-1` means no read timeout (None). Empty or invalid values fall back to the default.
    """
    raw_value = credentials.get("read_timeout")
    if raw_value in (None, ""):
        return default

    try:
        timeout_value = float(raw_value)
    except (TypeError, ValueError):
        return default

    if timeout_value == -1:
        return None

    if timeout_value <= 0:
        return default

    return timeout_value


def build_openai_timeout(credentials: Mapping) -> Timeout:
    read_timeout = resolve_read_timeout(credentials)

    return Timeout(
        DEFAULT_OPENAI_TOTAL_TIMEOUT if read_timeout is not None else None,
        read=read_timeout,
        write=DEFAULT_OPENAI_WRITE_TIMEOUT,
        connect=DEFAULT_OPENAI_CONNECT_TIMEOUT,
    )


def build_validate_timeout(credentials: Mapping) -> tuple[float, Optional[float]]:
    return (DEFAULT_VALIDATE_CONNECT_TIMEOUT, resolve_read_timeout(credentials))


class _CommonOpenAI:
    def _to_credential_kwargs(self, credentials: Mapping) -> dict:
        """
        Transform credentials to kwargs for model instance

        :param credentials:
        :return:
        """
        credentials_kwargs = {
            "api_key": credentials["api_key"],
            "timeout": build_openai_timeout(credentials),
            "max_retries": 1,
        }

        if credentials.get("endpoint_url"):
            openai_api_base = credentials["endpoint_url"].rstrip("/")
            credentials_kwargs["base_url"] = openai_api_base + "/v1"

        if "openai_organization" in credentials:
            credentials_kwargs["organization"] = credentials["openai_organization"]

        return credentials_kwargs

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [openai.APIConnectionError, openai.APITimeoutError],
            InvokeServerUnavailableError: [openai.InternalServerError],
            InvokeRateLimitError: [openai.RateLimitError],
            InvokeAuthorizationError: [openai.AuthenticationError, openai.PermissionDeniedError],
            InvokeBadRequestError: [
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
                openai.APIError,
            ],
        }
