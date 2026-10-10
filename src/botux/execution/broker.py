"""The interface between order management and a venue (paper simulator or live connector)."""

from __future__ import annotations

from typing import Protocol

from botux.models import Balance, OrderRequest, OrderResult


class Broker(Protocol):
    """Contract implemented by :class:`PaperBroker` and :class:`LiveBroker`.

    Error semantics (shared by both implementations):

    * ``OrderRejected``       - definitively not executed.
    * ``OutcomeUnknownError`` - may or may not have been executed; caller must reconcile.
    * ``NetworkError`` with ``not_sent=True`` - definitively not executed.
    """

    def place_order(self, request: OrderRequest) -> OrderResult: ...

    def get_order(self, symbol: str, client_order_id: str) -> OrderResult | None:
        """Current state, or None when the venue has no such order."""
        ...

    def cancel_order(self, symbol: str, client_order_id: str) -> OrderResult | None:
        """Cancel an open order; None when it is not open (already filled/cancelled/unknown)."""
        ...

    def open_orders(self, symbol: str) -> list[OrderResult]: ...

    def balances(self) -> dict[str, Balance]: ...
