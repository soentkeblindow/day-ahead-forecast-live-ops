"""Unit tests for ops/store_sources.py's redact_secrets (publication spec,
Sicherheitsprüfung section 3.1)."""

from __future__ import annotations

from energy_price_forecast.ops.store_sources import redact_secrets


def test_redact_secrets_replaces_a_security_token() -> None:
    text = (
        "HTTPError: 404 Client Error: Not Found for url: "
        "https://web-api.tp.entsoe.eu/api?documentType=A11&in_Domain=10YFR-RTE------C"
        "&out_Domain=10Y1001A1001A82H&securityToken=4359c395-caf3-44aa-ad56-5cd216700952"
    )

    redacted = redact_secrets(text)

    assert "4359c395" not in redacted
    assert "securityToken=***REDACTED***" in redacted


def test_redact_secrets_is_a_no_op_without_a_token() -> None:
    text = "ConnectionError: Failed to establish a new connection"

    assert redact_secrets(text) == text


def test_redact_secrets_stops_at_the_next_query_param() -> None:
    text = "url?securityToken=abc123&periodStart=202608010000"

    redacted = redact_secrets(text)

    assert redacted == "url?securityToken=***REDACTED***&periodStart=202608010000"
