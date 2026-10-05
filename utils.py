from __future__ import annotations

from typing import Any

NETWORK_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "TimeoutError",
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
    "TransportError",
}

RATE_LIMIT_ERROR_NAMES = {
    "RateLimitError",
    "RateLimit",
    "TooManyRequestsError",
}


def classify_exception(exc: Exception) -> tuple[str, str]:
    """Convierte errores de SDK diferentes a categorías comunes."""
    name = type(exc).__name__
    text = str(exc).strip() or name

    if name in RATE_LIMIT_ERROR_NAMES or "rate limit" in text.lower() or "429" in text:
        return "rate_limit", f"Límite de solicitudes alcanzado: {text}"

    if name in NETWORK_ERROR_NAMES or "timeout" in text.lower() or "connection" in text.lower():
        return "network", f"Error de red/conexión: {text}"

    if name in {"AuthenticationError", "AuthenticationException"} or "api key" in text.lower():
        return "authentication", f"Error de autenticación: {text}"

    return "provider_error", text


def safe_error(exc: Exception) -> tuple[str, str]:
    try:
        return classify_exception(exc)
    except Exception:
        return "provider_error", "Error desconocido del proveedor"
