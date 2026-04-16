from __future__ import annotations

import asyncio
import json
import os
import ssl
import threading
import time
from collections import deque
from typing import Any, Optional

import pandas as pd
import websockets
from dotenv import load_dotenv

CORTEX_URL = "wss://127.0.0.1:6868"
SSL_CONTEXT = ssl._create_unverified_context()

load_dotenv()
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
BANDS = ["theta", "alpha", "betaL", "betaH", "gamma"]
POW_COLUMNS = [f"{channel}_{band}" for channel in CHANNELS for band in BANDS]
ALL_COLUMNS = [*POW_COLUMNS, "timestamp"]


def create_payload(method: str, params: dict[str, Any], request_id: int) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}


async def send_json_rpc(
    websocket: websockets.WebSocketClientProtocol,
    method: str,
    params: dict[str, Any],
    request_id: int,
) -> dict[str, Any]:
    payload = create_payload(method, params, request_id)
    await websocket.send(json.dumps(payload))
    response = await websocket.recv()
    return json.loads(response)


async def request_access(websocket: websockets.WebSocketClientProtocol, client_id: str, client_secret: str) -> dict[str, Any]:
    return await send_json_rpc(websocket, "requestAccess", {"clientId": client_id, "clientSecret": client_secret}, 1)


async def control_device(
    websocket: websockets.WebSocketClientProtocol,
    command: str,
    request_id: int,
    headset_id: Optional[str] = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"command": command}
    if headset_id:
        params["headset"] = headset_id
    return await send_json_rpc(websocket, "controlDevice", params, request_id)


async def query_headsets(websocket: websockets.WebSocketClientProtocol) -> dict[str, Any]:
    return await send_json_rpc(websocket, "queryHeadsets", {}, 3)


async def authorize(websocket: websockets.WebSocketClientProtocol, client_id: str, client_secret: str) -> dict[str, Any]:
    return await send_json_rpc(websocket, "authorize", {"clientId": client_id, "clientSecret": client_secret, "debit": 1}, 5)


async def create_session(websocket: websockets.WebSocketClientProtocol, cortex_token: str, headset_id: str) -> dict[str, Any]:
    return await send_json_rpc(websocket, "createSession", {"cortexToken": cortex_token, "headset": headset_id, "status": "open"}, 6)


async def subscribe_to_streams(websocket: websockets.WebSocketClientProtocol, cortex_token: str, session_id: str) -> dict[str, Any]:
    return await send_json_rpc(websocket, "subscribe", {"cortexToken": cortex_token, "session": session_id, "streams": ["pow", "dev"]}, 7)


class EpocXStream:
    def __init__(self, verbose: bool = True, max_rows: int = 25000) -> None:
        self.verbose = verbose
        self.max_rows = max_rows
        self.status_message = "idle"
        self.sensor_contact_quality: Optional[float] = None
        self.is_connected = False
        self.is_streaming = False
        self.headset_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.cortex_token: Optional[str] = None
        self.last_error: Optional[str] = None

        self._rows: deque[dict[str, float]] = deque(maxlen=max_rows)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.status_message = "stopped"
        self.is_connected = False
        self.is_streaming = False

    def wait_until_ready(self, timeout: float = 20.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def require_ready(self, timeout: float = 20.0) -> None:
        if self.wait_until_ready(timeout=timeout):
            return
        status = self.status_message
        if self.last_error:
            status = f"{status}; last_error={self.last_error}"
        raise RuntimeError(f"EPOC X stream did not become ready within {timeout:.1f}s. Last status: {status}")

    def dataframe_between(self, start_epoch: float, end_epoch: float) -> pd.DataFrame:
        with self._lock:
            rows = [row.copy() for row in self._rows if start_epoch <= float(row.get("timestamp", -1.0)) <= end_epoch]
        if not rows:
            return pd.DataFrame(columns=ALL_COLUMNS)
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    def rows_between(self, start_epoch: float, end_epoch: float) -> list[dict[str, float]]:
        frame = self.dataframe_between(start_epoch, end_epoch)
        if frame.empty:
            return []
        return frame.to_dict(orient="records")

    def samples_between(self, start_epoch: float, end_epoch: float, fallback_latest: bool = False) -> list[dict[str, float]]:
        rows = self.rows_between(start_epoch, end_epoch)
        if rows or not fallback_latest:
            return rows
        with self._lock:
            if not self._rows:
                return []
            return [dict(self._rows[-1])]

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as exc:
            self.last_error = str(exc)
            self.status_message = f"error: {exc}"
            self._log(f"EPOC X stream error: {exc}")
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    async def _run(self) -> None:
        if not CLIENT_ID or not CLIENT_SECRET:
            self.status_message = "missing CLIENT_ID/CLIENT_SECRET in .env"
            return

        self.status_message = "connecting to Cortex"
        async with websockets.connect(CORTEX_URL, ssl=SSL_CONTEXT, ping_interval=20, ping_timeout=20) as websocket:
            await request_access(websocket, CLIENT_ID, CLIENT_SECRET)
            await control_device(websocket, "refresh", 2)
            await asyncio.sleep(1.0)

            resp = await query_headsets(websocket)
            headsets = resp.get("result", [])
            if not headsets:
                self.status_message = "no headset found"
                return

            self.headset_id = str(headsets[0]["id"])
            await control_device(websocket, "connect", 4, headset_id=self.headset_id)
            self.is_connected = True
            self.status_message = f"connected to headset {self.headset_id}, waiting for pow stream"
            self._log(f"Connected to EPOC X: {self.headset_id}")

            auth_resp = await authorize(websocket, CLIENT_ID, CLIENT_SECRET)
            self.cortex_token = auth_resp["result"]["cortexToken"]
            session_resp = await create_session(websocket, self.cortex_token, self.headset_id)
            self.session_id = session_resp["result"]["id"]
            await subscribe_to_streams(websocket, self.cortex_token, self.session_id)

            while not self._stop_event.is_set():
                try:
                    data_msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except TimeoutError:
                    continue
                except asyncio.TimeoutError:
                    continue
                data = json.loads(data_msg)
                self._handle_message(data)

    def _handle_message(self, data: dict[str, Any]) -> None:
        if "pow" in data and isinstance(data["pow"], list):
            values = list(data["pow"])
            if len(values) >= len(POW_COLUMNS):
                row = {column: float(values[idx]) for idx, column in enumerate(POW_COLUMNS)}
                row["timestamp"] = float(data.get("time", time.time()))
                with self._lock:
                    self._rows.append(row)
                if not self.is_streaming:
                    self._log("EPOC X pow stream is live.")
                self.is_streaming = True
                self.status_message = f"streaming pow ({len(self._rows)} buffered rows)"
                self._ready_event.set()

        if "dev" in data:
            try:
                dev = list(data["dev"][2])
                if dev:
                    self.sensor_contact_quality = float(sum(dev) / len(dev))
            except Exception:
                pass


if __name__ == "__main__":
    async def _main() -> None:
        stream = EpocXStream(verbose=True)
        stream.start()
        try:
            while True:
                await asyncio.sleep(1.0)
                print(stream.status_message, flush=True)
        except KeyboardInterrupt:
            stream.stop()

    asyncio.run(_main())
