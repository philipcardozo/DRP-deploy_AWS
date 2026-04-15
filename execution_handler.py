"""
Layer 4 — Execution Handler: Smart Order Routing.

Bridges the StrategyOrchestrator to Interactive Brokers with a
passive-execution "chase" algorithm that minimises slippage.

Architecture
────────────
    StrategyOrchestrator
         │ on_signal callback
         ▼
    ExecutionHandler (EWrapper + EClient — separate clientId)
         │
         ├── PlaceOrder at mid-price (passive limit)
         │
         ├── ChaseThread per order ──► modify limit every N secs
         │        (up to max_steps times toward the aggressive side)
         │
         ├── orderStatus / execDetails callbacks update OrderBook
         │
         └── FillCSVLogger ──► fills.csv (post-trade audit trail)

Order lifecycle state machine
──────────────────────────────
    PENDING_SUBMIT
         │ placeOrder() called
         ▼
    SUBMITTED
         │ first partial fill
         ▼
    PARTIALLY_FILLED
         │ remaining == 0          │ cancel()
         ▼                         ▼
       FILLED                  CANCELLED
         │ broker rejection
         ▼
      REJECTED

Chase algorithm
───────────────
1. Calculate mid = (bid + ask) / 2; post limit at mid.
2. Start a per-order daemon thread.
3. Every `chase_interval` seconds: if not fully filled, move the
   limit one tick toward the aggressive side (bid+tick for sells,
   ask−tick for buys).
4. After `max_chase_steps` moves: if still open, optionally
   convert to MKT or leave as a passive limit (configurable).

Tick-size rules (CBOE options)
───────────────────────────────
• Price < $3.00  → tick = $0.01
• Price ≥ $3.00  → tick = $0.05
"""

from __future__ import annotations

import csv
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable

try:
    from ibapi.client  import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order    import Order
    _IBAPI_AVAILABLE = True
except ImportError:
    _IBAPI_AVAILABLE = False

from orchestrator import TradeSignal, SignalAction, Position

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# IBKR stubs
# ════════════════════════════════════════════════════════════════════

if not _IBAPI_AVAILABLE:
    class EWrapper:                              # type: ignore[no-redef]
        pass

    class EClient:                               # type: ignore[no-redef]
        def __init__(self, wrapper):
            self.wrapper   = wrapper
            self._connected = False
            self._next_oid  = 1

        def connect(self, host, port, clientId):
            logger.info("[EXEC STUB] Connect %s:%s clientId=%s", host, port, clientId)
            self._connected = True

        def disconnect(self):
            self._connected = False

        def isConnected(self) -> bool:
            return self._connected

        def run(self):
            while self._connected:
                time.sleep(0.25)

        def reqIds(self, _):
            self.wrapper.nextValidId(self._next_oid)

        def placeOrder(self, orderId, contract, order):
            logger.info(
                "[EXEC STUB] placeOrder id=%d %s %s %s @ %.4f",
                orderId, order.action, order.totalQuantity,
                contract.symbol, order.lmtPrice,
            )
            # Simulate an immediate partial fill then full fill
            def _sim_fill():
                time.sleep(0.05)
                self.wrapper.orderStatus(
                    orderId, "Filled", order.totalQuantity, 0,
                    order.lmtPrice, 0, 0, order.lmtPrice, 0, "", 0.0)
            threading.Thread(target=_sim_fill, daemon=True).start()

        def cancelOrder(self, orderId, manualCancelOrderTime=""):
            logger.info("[EXEC STUB] cancelOrder id=%d", orderId)
            self.wrapper.orderStatus(
                orderId, "Cancelled", 0, 0, 0, 0, 0, 0, 0, "", 0.0)

        def reqCurrentTime(self):
            self.wrapper.currentTime(int(time.time()))

    class Contract:                              # type: ignore[no-redef]
        __slots__ = ("symbol", "secType", "exchange", "currency",
                     "lastTradeDateOrContractMonth", "strike",
                     "right", "multiplier")
        def __init__(self):
            for s in self.__slots__:
                object.__setattr__(self, s, "")
            object.__setattr__(self, "strike", 0.0)

    class Order:                                 # type: ignore[no-redef]
        def __init__(self):
            self.orderId        = 0
            self.action         = "BUY"
            self.totalQuantity  = 0
            self.orderType      = "LMT"
            self.lmtPrice       = 0.0
            self.tif            = "DAY"
            self.transmit       = True
            self.outsideRth     = False


# ════════════════════════════════════════════════════════════════════
# Data containers
# ════════════════════════════════════════════════════════════════════

class OrderStatus(Enum):
    PENDING_SUBMIT   = auto()
    SUBMITTED        = auto()
    PARTIALLY_FILLED = auto()
    FILLED           = auto()
    CANCELLED        = auto()
    REJECTED         = auto()


@dataclass
class OrderRecord:
    """Full lifecycle record for a single IBKR order."""
    order_id:       int
    signal_action:  str
    contract:       object          # Contract
    action:         str             # "BUY" | "SELL"
    quantity:       int
    order_type:     str             # "LMT" | "MKT"
    limit_price:    float
    submit_time:    datetime        = field(default_factory=datetime.now)
    status:         OrderStatus     = OrderStatus.PENDING_SUBMIT
    filled_qty:     int             = 0
    remaining_qty:  int             = 0
    avg_fill_price: float           = 0.0
    last_fill_time: datetime | None = None
    commission:     float           = 0.0
    chase_step:     int             = 0
    reject_reason:  str             = ""
    # Derived
    slippage:       float           = 0.0   # avg_fill - limit at submission


@dataclass
class FillRecord:
    """Single execution / partial fill — written to fills.csv."""
    timestamp:      str
    order_id:       int
    symbol:         str
    action:         str
    filled_qty:     int
    avg_fill_price: float
    commission:     float
    slippage:       float
    status:         str
    chase_step:     int
    signal_action:  str


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _option_tick_size(price: float) -> float:
    """CBOE standard tick size for equity options."""
    return 0.01 if price < 3.0 else 0.05


def _make_limit_order(action: str, qty: int,
                      limit_price: float, order_id: int) -> Order:
    o               = Order()
    o.orderId       = order_id
    o.action        = action
    o.totalQuantity = qty
    o.orderType     = "LMT"
    o.lmtPrice      = round(limit_price, 2)
    o.tif           = "DAY"
    o.transmit      = True
    return o


def _make_market_order(action: str, qty: int, order_id: int) -> Order:
    o               = Order()
    o.orderId       = order_id
    o.action        = action
    o.totalQuantity = qty
    o.orderType     = "MKT"
    o.lmtPrice      = 0.0
    o.tif           = "DAY"
    o.transmit      = True
    return o


# ════════════════════════════════════════════════════════════════════
# Fill logger (background CSV writer)
# ════════════════════════════════════════════════════════════════════

_FILL_COLUMNS = [
    "timestamp", "order_id", "symbol", "action", "filled_qty",
    "avg_fill_price", "commission", "slippage", "status",
    "chase_step", "signal_action",
]


class FillLogger:
    def __init__(self, path: str = "fills.csv"):
        self._path  = Path(path)
        self._q: queue.Queue[FillRecord] = queue.Queue()
        self._stop  = threading.Event()
        self._file  = None
        self._writer = None
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        exists = self._path.exists()
        self._file   = open(self._path, "a", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=_FILL_COLUMNS,
                                       extrasaction="ignore")
        if not exists or os.path.getsize(self._path) == 0:
            self._writer.writeheader()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="fill-logger")
        self._thread.start()

    def log(self, rec: FillRecord) -> None:
        self._q.put_nowait(rec)

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        while not self._q.empty():
            try:
                self._writer.writerow(self._q.get_nowait().__dict__)
            except Exception:
                pass
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rec = self._q.get(timeout=0.2)
                self._writer.writerow(rec.__dict__)
                while True:
                    try:
                        self._writer.writerow(self._q.get_nowait().__dict__)
                    except queue.Empty:
                        break
            except queue.Empty:
                continue


# ════════════════════════════════════════════════════════════════════
# Execution Handler
# ════════════════════════════════════════════════════════════════════

class ExecutionHandler(EWrapper, EClient):
    """
    Smart-routing execution layer for the Regime Volatility Arbitrage Engine.

    Connects to TWS on a dedicated clientId (separate from the market-data
    ConnectionManager) so order events and tick events never share a socket.

    Parameters
    ──────────
    host : str
    port : int         7497 paper / 7496 live
    client_id : int    Must differ from ConnectionManager.client_id
    contract : Contract
        Default option contract to trade (can be overridden per-signal).
    chase_interval : float
        Seconds to wait before repricing an unfilled limit order.
    max_chase_steps : int
        Maximum number of limit price adjustments before giving up.
    market_on_timeout : bool
        If True, convert to MKT order after max_chase_steps.
        If False, leave as final limit price.
    on_fill : Callable | None
        Callback invoked on every complete fill (FILLED status).
    fills_log_path : str
        Path for the fills CSV audit trail.
    """

    def __init__(
        self,
        host:               str      = "127.0.0.1",
        port:               int      = 7497,
        client_id:          int      = 2,
        contract:           object | None = None,
        chase_interval:     float    = 10.0,
        max_chase_steps:    int      = 5,
        market_on_timeout:  bool     = False,
        on_fill:            Callable[[OrderRecord], None] | None = None,
        fills_log_path:     str      = "fills.csv",
    ):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self.host              = host
        self.port              = port
        self.client_id         = client_id
        self.default_contract  = contract
        self.chase_interval    = chase_interval
        self.max_chase_steps   = max_chase_steps
        self.market_on_timeout = market_on_timeout
        self.on_fill           = on_fill

        # Order book: order_id → OrderRecord
        self._book:   dict[int, OrderRecord] = {}
        self._book_lock = threading.Lock()

        # Per-order chase threads
        self._chase_threads: dict[int, threading.Thread] = {}
        self._chase_stop:    dict[int, threading.Event]  = {}

        # Market quote cache (updated via StrategyOrchestrator's MarketState)
        self._bid: float = 0.0
        self._ask: float = 0.0
        self._spot: float = 0.0

        # Order ID counter (seeded from TWS via nextValidId)
        self._next_oid      = 100
        self._oid_lock      = threading.Lock()
        self._oid_ready_evt = threading.Event()

        # Session metrics
        self.orders_placed  = 0
        self.orders_filled  = 0
        self.orders_cancelled = 0
        self.total_slippage = 0.0

        # Infrastructure
        self._reader_thread: threading.Thread | None = None
        self._stop   = threading.Event()
        self._logger = FillLogger(fills_log_path)

    # ════════════════════════════════════════════════════════════════
    # Lifecycle
    # ════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Connect to TWS and launch the daemon reader thread."""
        self._stop.clear()
        self._logger.open()
        try:
            self.connect(self.host, self.port, self.client_id)
        except Exception as exc:
            logger.error("ExecutionHandler connect failed: %s", exc)

        self._reader_thread = threading.Thread(
            target=self._run_loop,
            name="exec-reader",
            daemon=True,
        )
        self._reader_thread.start()

        # Request first valid order ID (blocks up to 5 s)
        self.reqIds(1)
        self._oid_ready_evt.wait(timeout=5.0)
        logger.info("ExecutionHandler started  port=%d  clientId=%d  nextOid=%d",
                    self.port, self.client_id, self._next_oid)

    def stop(self) -> None:
        """Cancel all open orders, flush fills, and disconnect."""
        self._stop.set()
        self._cancel_all_open()
        # Stop all chase threads
        for stop_evt in self._chase_stop.values():
            stop_evt.set()
        if self.isConnected():
            self.disconnect()
        self._logger.close()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3.0)
        logger.info(
            "ExecutionHandler stopped | placed=%d filled=%d "
            "cancelled=%d avg_slippage=$%.4f",
            self.orders_placed, self.orders_filled,
            self.orders_cancelled, self._avg_slippage(),
        )

    def __enter__(self) -> "ExecutionHandler":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ════════════════════════════════════════════════════════════════
    # Public: execute a StrategyOrchestrator signal
    # ════════════════════════════════════════════════════════════════

    def execute(self, signal: TradeSignal) -> int | None:
        """
        Entry point called by StrategyOrchestrator.on_signal.

        Maps the TradeSignal action to an order action and dispatches
        to the smart router.

        Returns
        ──────
        order_id (int) if an order was placed, None otherwise.
        """
        if signal.action in (SignalAction.HOLD, SignalAction.DELTA_HEDGE):
            logger.debug("execute: skipping action=%s", signal.action)
            return None

        contract = self.default_contract or _default_spy_option()

        if signal.action == SignalAction.ENTER_LONG:
            return self._place_passive(contract, "BUY", qty=1,
                                       signal_action=signal.action.value)

        if signal.action == SignalAction.ADD_LONG:
            return self._place_passive(contract, "BUY", qty=1,
                                       signal_action=signal.action.value)

        if signal.action in (SignalAction.CLOSE_ALL, SignalAction.STOP_LOSS):
            # For exits, use aggressive limit (cross the spread) or MKT
            return self._place_passive(contract, "SELL",
                                       qty=signal.position.qty or 1,
                                       signal_action=signal.action.value,
                                       aggressive=True)
        return None

    # ════════════════════════════════════════════════════════════════
    # Smart routing: passive limit → chase
    # ════════════════════════════════════════════════════════════════

    def _place_passive(
        self,
        contract:      object,
        action:        str,
        qty:           int     = 1,
        signal_action: str     = "",
        aggressive:    bool    = False,
    ) -> int:
        """
        Place a limit order at mid-price (or best bid/ask for aggressive
        exits) and start the chase thread.

        Parameters
        ──────────
        aggressive : bool
            If True, post at the ask (for buys) or bid (for sells)
            rather than mid — used on forced exit signals.
        """
        mid   = self._mid_price()
        bid   = self._bid
        ask   = self._ask

        if mid <= 0:
            # No quote — fall back to market order
            logger.warning("No quote available — placing MKT order")
            return self._place_market(contract, action, qty, signal_action)

        if aggressive:
            limit = ask if action == "BUY" else bid
        else:
            limit = mid

        limit = max(round(limit, 2), 0.01)
        oid   = self._next_order_id()

        rec = OrderRecord(
            order_id=oid,
            signal_action=signal_action,
            contract=contract,
            action=action,
            quantity=qty,
            order_type="LMT",
            limit_price=limit,
        )

        with self._book_lock:
            self._book[oid] = rec

        order = _make_limit_order(action, qty, limit, oid)
        self.placeOrder(oid, contract, order)
        rec.status = OrderStatus.SUBMITTED
        self.orders_placed += 1

        logger.info(
            "PASSIVE  %s %d × %s @ %.2f  oid=%d  [mid=%.2f bid=%.2f ask=%.2f]",
            action, qty, getattr(contract, "symbol", "?"),
            limit, oid, mid, bid, ask,
        )

        # Spawn chase thread
        stop_evt = threading.Event()
        self._chase_stop[oid] = stop_evt
        chase_thread = threading.Thread(
            target=self._chase_loop,
            args=(oid, stop_evt),
            name=f"chase-{oid}",
            daemon=True,
        )
        self._chase_threads[oid] = chase_thread
        chase_thread.start()

        return oid

    def _chase_loop(self, oid: int, stop_evt: threading.Event) -> None:
        """
        Per-order daemon thread that adjusts the limit price every
        `chase_interval` seconds until filled or max steps reached.

        For a BUY order: each step moves the limit UP by one tick (toward ask).
        For a SELL order: each step moves the limit DOWN by one tick (toward bid).
        """
        step = 0
        while not stop_evt.is_set() and step < self.max_chase_steps:
            stop_evt.wait(timeout=self.chase_interval)
            if stop_evt.is_set():
                return

            with self._book_lock:
                rec = self._book.get(oid)
                if rec is None:
                    return
                if rec.status in (OrderStatus.FILLED,
                                  OrderStatus.CANCELLED,
                                  OrderStatus.REJECTED):
                    return

            # Move limit one tick toward the aggressive side
            tick    = _option_tick_size(rec.limit_price)
            new_lmt = (rec.limit_price + tick
                       if rec.action == "BUY"
                       else rec.limit_price - tick)
            new_lmt = max(round(new_lmt, 2), 0.01)

            logger.info(
                "CHASE step=%d  oid=%d  %s  %.2f → %.2f",
                step + 1, oid, rec.action, rec.limit_price, new_lmt,
            )

            # Modify the order in TWS (same orderId = modify)
            mod_order = _make_limit_order(
                rec.action, rec.quantity - rec.filled_qty, new_lmt, oid)
            self.placeOrder(oid, rec.contract, mod_order)

            with self._book_lock:
                rec.limit_price = new_lmt
                rec.chase_step  = step + 1

            step += 1

        # Max steps reached and still open
        with self._book_lock:
            rec = self._book.get(oid)
            if rec is None:
                return
            if rec.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                              OrderStatus.REJECTED):
                return

        if self.market_on_timeout:
            logger.warning(
                "CHASE exhausted oid=%d — converting to MKT order", oid)
            mkt_order = _make_market_order(
                rec.action, rec.quantity - rec.filled_qty, oid)
            with self._book_lock:
                rec.order_type = "MKT"
            self.placeOrder(oid, rec.contract, mkt_order)
        else:
            logger.info(
                "CHASE exhausted oid=%d — leaving passive limit @ %.2f",
                oid, rec.limit_price)

    def _place_market(self, contract, action, qty, signal_action) -> int:
        oid = self._next_order_id()
        rec = OrderRecord(
            order_id=oid, signal_action=signal_action,
            contract=contract, action=action, quantity=qty,
            order_type="MKT", limit_price=0.0,
        )
        with self._book_lock:
            self._book[oid] = rec
        order = _make_market_order(action, qty, oid)
        self.placeOrder(oid, contract, order)
        rec.status = OrderStatus.SUBMITTED
        self.orders_placed += 1
        logger.info("MARKET  %s %d × %s  oid=%d",
                    action, qty, getattr(contract, "symbol", "?"), oid)
        return oid

    # ════════════════════════════════════════════════════════════════
    # EWrapper callbacks — order lifecycle
    # ════════════════════════════════════════════════════════════════

    def orderStatus(
        self, orderId, status, filled, remaining,
        avgFillPrice, permId, parentId,
        lastFillPrice, clientId, whyHeld, mktCapPrice,
    ):
        """
        Core order state machine.  Maps TWS status strings to
        OrderStatus enum values and updates the order book.
        """
        with self._book_lock:
            rec = self._book.get(orderId)
            if rec is None:
                return

            prev_status = rec.status

            if status == "Filled":
                rec.status         = OrderStatus.FILLED
                rec.filled_qty     = int(filled)
                rec.remaining_qty  = 0
                rec.avg_fill_price = avgFillPrice
                rec.last_fill_time = datetime.now()
                rec.slippage       = avgFillPrice - rec.limit_price
                self.orders_filled += 1
                self.total_slippage += rec.slippage

            elif status == "PartiallyFilled":
                rec.status         = OrderStatus.PARTIALLY_FILLED
                rec.filled_qty     = int(filled)
                rec.remaining_qty  = int(remaining)
                rec.avg_fill_price = avgFillPrice
                rec.last_fill_time = datetime.now()

            elif status in ("Cancelled", "ApiCancelled"):
                rec.status       = OrderStatus.CANCELLED
                rec.remaining_qty = int(remaining)
                self.orders_cancelled += 1

            elif status == "Inactive":
                rec.status = OrderStatus.REJECTED

            elif status == "Submitted":
                if rec.status == OrderStatus.PENDING_SUBMIT:
                    rec.status = OrderStatus.SUBMITTED
                rec.remaining_qty = int(remaining)

        # Log fill
        if rec.status == OrderStatus.FILLED:
            self._log_fill(rec)
            self._stop_chase(orderId)
            if self.on_fill:
                try:
                    self.on_fill(rec)
                except Exception as exc:
                    logger.error("on_fill callback raised: %s", exc)

            sym = getattr(rec.contract, "symbol", "?")
            logger.info(
                "FILLED  oid=%d  %s %d × %s @ %.4f  slippage=$%.4f  "
                "chase_steps=%d",
                orderId, rec.action, rec.filled_qty, sym,
                rec.avg_fill_price, rec.slippage, rec.chase_step,
            )

        elif rec.status == OrderStatus.CANCELLED:
            self._stop_chase(orderId)
            logger.info("CANCELLED  oid=%d", orderId)

        elif rec.status == OrderStatus.REJECTED:
            self._stop_chase(orderId)
            logger.warning("REJECTED  oid=%d  reason=%s", orderId, rec.reject_reason)

    def openOrder(self, orderId, contract, order, orderState):
        logger.debug("openOrder  oid=%d  status=%s", orderId, orderState.status)

    def execDetails(self, reqId, contract, execution):
        logger.debug(
            "execDetails  oid=%d  qty=%d  price=%.4f  side=%s",
            execution.orderId, execution.shares,
            execution.price, execution.side,
        )

    def commissionReport(self, commissionReport):
        oid = commissionReport.execId  # not order_id, but close enough for logging
        comm = commissionReport.commission
        logger.debug("commissionReport  execId=%s  commission=%.4f", oid, comm)
        # Update the most recently filled order's commission
        with self._book_lock:
            for rec in reversed(list(self._book.values())):
                if rec.status == OrderStatus.FILLED and rec.commission == 0.0:
                    rec.commission = comm
                    break

    def openOrderEnd(self):
        logger.debug("openOrderEnd — all open orders received")

    # ════════════════════════════════════════════════════════════════
    # EWrapper callbacks — connection
    # ════════════════════════════════════════════════════════════════

    def nextValidId(self, orderId: int):
        with self._oid_lock:
            self._next_oid = max(self._next_oid, orderId)
        self._oid_ready_evt.set()
        logger.debug("nextValidId=%d", orderId)

    def connectAck(self):
        logger.info("ExecutionHandler connectAck  clientId=%d", self.client_id)

    def connectionClosed(self):
        logger.warning("ExecutionHandler connectionClosed")

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        msg = f"[EXEC] TWS error reqId={reqId} code={errorCode}: {errorString}"
        if errorCode == 201:   # order rejected
            with self._book_lock:
                rec = self._book.get(reqId)
                if rec:
                    rec.status = OrderStatus.REJECTED
                    rec.reject_reason = errorString
            self._stop_chase(reqId)
            logger.warning(msg + "  ← ORDER REJECTED")
        elif errorCode in (2104, 2106, 2107, 2158):
            logger.debug(msg)
        else:
            logger.error(msg)

    # ════════════════════════════════════════════════════════════════
    # Market quote injection
    # ════════════════════════════════════════════════════════════════

    def update_quote(self, bid: float, ask: float, spot: float = 0.0) -> None:
        """
        Called by ConnectionManager (or a test harness) to keep the
        quote cache fresh for mid-price calculations.
        """
        self._bid  = bid
        self._ask  = ask
        self._spot = spot

    # ════════════════════════════════════════════════════════════════
    # Cancellation
    # ════════════════════════════════════════════════════════════════

    def cancel(self, oid: int) -> None:
        """Cancel a specific open order."""
        self._stop_chase(oid)
        with self._book_lock:
            rec = self._book.get(oid)
            if rec and rec.status not in (OrderStatus.FILLED,
                                           OrderStatus.CANCELLED,
                                           OrderStatus.REJECTED):
                self.cancelOrder(oid)

    def _cancel_all_open(self) -> None:
        with self._book_lock:
            open_ids = [
                oid for oid, rec in self._book.items()
                if rec.status not in (OrderStatus.FILLED,
                                       OrderStatus.CANCELLED,
                                       OrderStatus.REJECTED)
            ]
        for oid in open_ids:
            self.cancel(oid)

    # ════════════════════════════════════════════════════════════════
    # Status & diagnostics
    # ════════════════════════════════════════════════════════════════

    def order_book_summary(self) -> list[dict]:
        with self._book_lock:
            return [
                {
                    "oid":        rec.order_id,
                    "action":     rec.action,
                    "qty":        rec.quantity,
                    "status":     rec.status.name,
                    "filled":     rec.filled_qty,
                    "avg_fill":   rec.avg_fill_price,
                    "limit":      rec.limit_price,
                    "slippage":   rec.slippage,
                    "chase_step": rec.chase_step,
                }
                for rec in self._book.values()
            ]

    def session_summary(self) -> str:
        return (
            f"ExecutionHandler Session Summary\n"
            f"  Orders placed    : {self.orders_placed}\n"
            f"  Orders filled    : {self.orders_filled}\n"
            f"  Orders cancelled : {self.orders_cancelled}\n"
            f"  Fill rate        : {self._fill_rate():.1%}\n"
            f"  Avg slippage     : ${self._avg_slippage():.4f}\n"
            f"  Total slippage   : ${self.total_slippage:.4f}"
        )

    # ════════════════════════════════════════════════════════════════
    # Private helpers
    # ════════════════════════════════════════════════════════════════

    def _run_loop(self) -> None:
        """Daemon reader thread — delegates to EClient.run()."""
        while not self._stop.is_set():
            try:
                self.run()
            except Exception as exc:
                logger.error("exec reader loop: %s", exc)
            if not self._stop.is_set():
                time.sleep(1.0)

    def _next_order_id(self) -> int:
        with self._oid_lock:
            oid = self._next_oid
            self._next_oid += 1
            return oid

    def _stop_chase(self, oid: int) -> None:
        evt = self._chase_stop.pop(oid, None)
        if evt:
            evt.set()
        self._chase_threads.pop(oid, None)

    def _mid_price(self) -> float:
        if self._bid > 0 and self._ask > 0:
            return (self._bid + self._ask) / 2.0
        return 0.0

    def _log_fill(self, rec: OrderRecord) -> None:
        fill = FillRecord(
            timestamp=datetime.now().isoformat(),
            order_id=rec.order_id,
            symbol=getattr(rec.contract, "symbol", "?"),
            action=rec.action,
            filled_qty=rec.filled_qty,
            avg_fill_price=rec.avg_fill_price,
            commission=rec.commission,
            slippage=rec.slippage,
            status=rec.status.name,
            chase_step=rec.chase_step,
            signal_action=rec.signal_action,
        )
        self._logger.log(fill)

    def _fill_rate(self) -> float:
        if self.orders_placed == 0:
            return 0.0
        return self.orders_filled / self.orders_placed

    def _avg_slippage(self) -> float:
        if self.orders_filled == 0:
            return 0.0
        return self.total_slippage / self.orders_filled


# ════════════════════════════════════════════════════════════════════
# Default contract helper
# ════════════════════════════════════════════════════════════════════

def _default_spy_option() -> Contract:
    """Placeholder ATM SPY straddle leg (replace with live contract)."""
    c = Contract()
    c.symbol     = "SPY"
    c.secType    = "OPT"
    c.exchange   = "SMART"
    c.currency   = "USD"
    c.right      = "C"
    c.multiplier = "100"
    return c
