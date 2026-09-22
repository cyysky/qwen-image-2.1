"""Shared exception types so routes can return clean upstream errors."""

from __future__ import annotations


class UpstreamError(RuntimeError):
    """An upstream (image endpoint or enhancer) returned an error."""

    def __init__(self, status: int, message: str, *, url: str = "", body: str = ""):
        super().__init__(message)
        self.status = status
        self.message = message
        self.url = url
        self.body = body

    def to_dict(self) -> dict:
        return {
            "error": self.message,
            "upstream_status": self.status,
            "upstream_url": self.url,
            "upstream_body": self.body[:4000],
        }


class BadRequest(RuntimeError):
    """The caller sent something we cannot use (mapped to HTTP 400)."""
