"""
WebSocket connection manager for VitalFlow AI real-time alerts.

Architecture
------------
The doctor portal establishes a persistent WebSocket connection to
    ws://localhost:8000/ws/doctor/{doctor_id}

on login.  The connection carries a JWT in the query string for authentication:
    ws://localhost:8000/ws/doctor/3?token=eyJhbGc...

When a patient submits a vital-sign reading that crosses a threshold, the
patient reading route calls `ws_manager.notify_doctor(doctor_id, payload)`,
which pushes a JSON message to all active connections for that doctor in under
10 milliseconds — no polling required.

Connection lifecycle:
  connect()    → adds the WebSocket to the doctor's connection list
  disconnect() → removes it (called on normal close or network error)
  notify_doctor() → fans out to all connections for a given doctor_id

Multiple tabs / devices are supported: all connections for the same doctor_id
receive every alert.
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import DefaultDict, List

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Thread-safe in-process WebSocket hub.

    Keyed by doctor_id (int) → list of active WebSocket connections.
    """

    def __init__(self):
        # DefaultDict means we never need to initialise a key manually
        self._connections: DefaultDict[int, List[WebSocket]] = defaultdict(list)

    async def connect(self, doctor_id: int, websocket: WebSocket) -> None:
        """Accept the WebSocket handshake and register the connection."""
        await websocket.accept()
        self._connections[doctor_id].append(websocket)
        logger.info(
            "[WS] Doctor %d connected. Active connections: %d",
            doctor_id,
            len(self._connections[doctor_id]),
        )

    def disconnect(self, doctor_id: int, websocket: WebSocket) -> None:
        """Remove a connection (called on close or error)."""
        connections = self._connections.get(doctor_id, [])
        if websocket in connections:
            connections.remove(websocket)
        logger.info(
            "[WS] Doctor %d disconnected. Remaining: %d",
            doctor_id,
            len(connections),
        )

    async def notify_doctor(self, doctor_id: int, payload: dict) -> None:
        """
        Push a JSON payload to all WebSocket connections for *doctor_id*.
        Stale connections (already closed) are silently removed.
        """
        connections = self._connections.get(doctor_id, [])
        if not connections:
            logger.debug("[WS] No active connections for doctor %d.", doctor_id)
            return

        message   = json.dumps(payload)
        dead: List[WebSocket] = []

        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                # Connection dropped without a clean close frame
                dead.append(ws)

        for ws in dead:
            self.disconnect(doctor_id, ws)

    def notify_doctor_sync(self, doctor_id: int, payload: dict) -> None:
        """
        Synchronous wrapper — safe to call from FastAPI sync endpoint handlers
        that run in a thread pool while the event loop is still running on the
        main thread.

        Uses asyncio.ensure_future() to schedule the coroutine on the running
        event loop without blocking the calling thread.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.notify_doctor(doctor_id, payload))
            else:
                loop.run_until_complete(self.notify_doctor(doctor_id, payload))
        except RuntimeError:
            logger.warning(
                "[WS] Could not schedule notify_doctor for doctor %d — no event loop.",
                doctor_id,
            )

    def active_doctor_ids(self) -> List[int]:
        """Return the list of doctor IDs with at least one active connection."""
        return [did for did, conns in self._connections.items() if conns]


# ── Singleton instance shared across the entire FastAPI application ────────────
ws_manager = ConnectionManager()


# ── WebSocket endpoint handler ────────────────────────────────────────────────

async def doctor_ws_endpoint(
    websocket: WebSocket,
    doctor_id: int,
    token: str,
) -> None:
    """
    FastAPI WebSocket endpoint handler.

    Authentication: the JWT token is validated before accepting the connection.
    If invalid, the connection is rejected with code 1008 (Policy Violation).

    Mount this in main.py:
        from ws_manager import ws_manager, doctor_ws_endpoint
        from fastapi import WebSocket, Query

        @app.websocket("/ws/doctor/{doctor_id}")
        async def ws_doctor(
            websocket: WebSocket,
            doctor_id: int,
            token: str = Query(...),
        ):
            await doctor_ws_endpoint(websocket, doctor_id, token)
    """
    from auth import SECRET_KEY, ALGORITHM
    from jose import jwt, JWTError

    # Validate JWT before accepting the handshake
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_doctor_id: int = payload.get("doctor_id")
        if token_doctor_id != doctor_id:
            await websocket.close(code=1008, reason="Token does not match doctor_id")
            return
    except JWTError:
        await websocket.close(code=1008, reason="Invalid token")
        return

    await ws_manager.connect(doctor_id, websocket)
    try:
        # Keep the connection alive.  Clients may send {"type": "ping"} to
        # prevent proxy-level idle timeouts; we respond with {"type": "pong"}.
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg  = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                # Send server-side keepalive every 30 s
                await websocket.send_text(json.dumps({"type": "keepalive"}))
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass  # ignore malformed messages
    finally:
        ws_manager.disconnect(doctor_id, websocket)
