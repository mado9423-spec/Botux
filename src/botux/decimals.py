"""Decimal helpers. All prices, quantities and balances are ``Decimal``: never ``float``."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation

ZERO = Decimal(0)
ONE = Decimal(1)


def D(value: str | int | Decimal) -> Decimal:
    """Build a Decimal from str/int/Decimal. Floats are rejected on purpose (binary rounding)."""
    if isinstance(value, bool | float):
        raise TypeError(f"refusing to build a Decimal from {type(value).__name__}: {value!r}")
    if isinstance(value, Decimal):
        return value
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"not a valid decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"decimal must be finite: {value!r}")
    return result


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` DOWN to a multiple of ``step`` (step == 0 means 'no constraint').

    Rounding is always toward zero for non-negative inputs: we never round a quantity or a
    budget up, because that could exceed a risk limit or the available balance.
    """
    if step == 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` UP to a multiple of ``step`` (used to make simulated costs pessimistic)."""
    if step == 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def is_multiple_of(value: Decimal, step: Decimal) -> bool:
    """True when ``value`` is an exact multiple of ``step`` (or step is 0 = unconstrained)."""
    if step == 0:
        return True
    return value % step == 0


def fmt(value: Decimal) -> str:
    """Plain (non-scientific) string, as the Binance API expects."""
    text = format(value.normalize(), "f")
    return text if "." not in text else text.rstrip("0").rstrip(".") or "0"
