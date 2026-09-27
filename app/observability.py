import time
import uuid
import logging
import json
from typing import Any, Dict, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Sensitivity Filter Keys
SENSITIVE_KEYS = {
    "api_key", "groq_api_key", "password", "secret", "token",
    "authorization", "bearer", "cookie", "set-cookie", "credentials"
}


def sanitize_log_data(data: Any) -> Any:
    """
    Recursively scrubs sensitive keys, credentials, and API keys from log records.
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if any(s in key.lower() for s in SENSITIVE_KEYS):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    return data


class JSONFormatter(logging.Formatter):
    """
    Formats log records as structured JSON strings for log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom context attributes attached via extra={...}
        if hasattr(record, "request_id"):
            log_obj["request_id"] = getattr(record, "request_id")
        if hasattr(record, "endpoint"):
            log_obj["endpoint"] = getattr(record, "endpoint")
        if hasattr(record, "metrics"):
            log_obj["metrics"] = getattr(record, "metrics")

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(sanitize_log_data(log_obj))


def setup_logger(name: str = "rag_engine", level: str = "INFO") -> logging.Logger:
    """
    Initializes and configures the root logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


logger = setup_logger()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Middleware for correlation tracking (Request-ID) and timing metrics.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        # Extract or generate Request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = str(duration_ms)

        # Structured request logging
        logger.info(
            f"HTTP {request.method} {request.url.path} -> {response.status_code}",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "metrics": {
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client_ip": request.client.host if request.client else "unknown"
                }
            }
        )

        return response