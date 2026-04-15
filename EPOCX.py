from __future__ import annotations

import asyncio
import json
import os
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import pandas as pd
import websockets
from dotenv import load_dotenv


CORTEX_URL = "wss://127.0.0.1:6868"

load_dotenv()

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
SSL_CONTEXT = ssl._create_unverified_context()

CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
BANDS = ["theta", "alpha", "betaL", "betaH", "gamma"]
POW_COLUMNS = [f"{channel}_{band}" for channel in CHANNELS for band in BANDS]


@dataclass(frozen=True)
class PowerSample:
    timestamp: float
    values: dict[str, float]
    sensor_contact_quality: Optional[float]


class EpocXStream:
    """
    Background Cortex listener for the EPOC X power-band stream.

    `connected` means the headset session is open.
    `streaming` means at least one pow sample has been received.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        cortex_url: str = CORTEX_URL,
        max_buffer_samples: int = 4096,
        verbose: bool = False,
    ) -> None:
        self.client_id = client_id or CLIENT_ID
        self.client_secret = client_secret or CLIENT_SECRET
        self.cortex_url = cortex_url
        self.verbose = verbose
        self.max_buffer_samples = max_buffer_samples

        self._buffer: Deque[PowerSample] = deque(maxlen=max_buffer_samples)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._streaming_event = threading.Event()

        self._connected = False
        self._streaming = False
        self._last_error: Optional[str] = None
        self._sensor_contact_quality: Optional[float] = None
        self._status_message = "idle"
        self._headset_id: Optional[str] = None
        self._session_id: Optional[str] = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def streaming(self) -> bool:
        return self._streaming

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def sensor_contact_quality(self) -> Optional[float]:
        return self._sensor_contact_quality

    @property
    def headset_id(self) -> Optional[str]:
        return self._headset_id

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._connected_event.clear()
        self._streaming_event.clear()
        self._connected = False
        self._streaming = False
        self._last_error = None
        self._status_message = "starting"
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="epocx-stream")
        self._thread.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

        self._connected = False
        self._streaming = False
        self._status_message = "stopped"

    def wait_until_connected(self, timeout: float = 10.0) -> bool:
        return self._connected_event.wait(timeout=timeout)

    def wait_until_streaming(self, timeout: float = 15.0) -> bool:
        return self._streaming_event.wait(timeout=timeout)

    def latest_sample(self) -> Optional[PowerSample]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def latest_psd_dict(self) -> Optional[dict[str, float]]:
        latest = self.latest_sample()
        if latest is None:
            return None
        return dict(latest.values)

    def samples_between(self, start_ts: float, end_ts: float, *, fallback_latest: bool = False) -> list[PowerSample]:
        with self._lock:
            samples = [sample for sample in self._buffer if start_ts <= sample.timestamp <= end_ts]
            if samples:
                return samples
            if fallback_latest and self._buffer:
                return [self._buffer[-1]]
            return []

    def dataframe_between(self, start_ts: float, end_ts: float, *, fallback_latest: bool = False) -> pd.DataFrame:
        samples = self.samples_between(start_ts, end_ts, fallback_latest=fallback_latest)
        rows: list[dict[str, float | None]] = []
        for sample in samples:
            row: dict[str, float | None] = {column: sample.values.get(column) for column in POW_COLUMNS}
            row["timestamp"] = sample.timestamp
            row["sensor_contact_quality"] = sample.sensor_contact_quality
            rows.append(row)
        return pd.DataFrame(rows)

    def mean_psd_between(self, start_ts: float, end_ts: float) -> Optional[dict[str, float]]:
        with self._lock:
            rows = [sample.values for sample in self._buffer if start_ts <= sample.timestamp <= end_ts]

        if not rows:
            return self.latest_psd_dict()

        frame = pd.DataFrame(rows, columns=POW_COLUMNS)
        means = frame.mean(axis=0, numeric_only=True)
        return {column: float(means.get(column, float("nan"))) for column in POW_COLUMNS}

    def mean_psd_from_samples(self, samples: list[PowerSample]) -> Optional[dict[str, float]]:
        if not samples:
            return self.latest_psd_dict()
        frame = pd.DataFrame([sample.values for sample in samples], columns=POW_COLUMNS)
        means = frame.mean(axis=0, numeric_only=True)
        return {column: float(means.get(column, float("nan"))) for column in POW_COLUMNS}

    def buffer_dataframe(self) -> pd.DataFrame:
        with self._lock:
            rows = []
            for sample in self._buffer:
                row = dict(sample.values)
                row["timestamp"] = sample.timestamp
                row["sensor_contact_quality"] = sample.sensor_contact_quality
                rows.append(row)
        return pd.DataFrame(rows)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._stream_forever())
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            self._loop.close()

    async def _stream_forever(self) -> None:
        try:
            if not self.client_id or not self.client_secret:
                raise RuntimeError("CLIENT_ID and CLIENT_SECRET must be set in the environment or .env file.")

            self._status_message = "connecting to Cortex"
            self._log("[EPOC X] Connecting to Cortex...")

            async with websockets.connect(self.cortex_url, ssl=SSL_CONTEXT, ping_interval=20, ping_timeout=20) as ws:
                await self._initialize_session(ws)
                self._connected = True
                self._connected_event.set()
                self._status_message = "connected; waiting for pow samples"
                self._log(f"[EPOC X] Connected to headset {self._headset_id}.")
                self._log("[EPOC X] Waiting for pow stream...")
                await self._listen(ws)
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            self._streaming = False
            self._last_error = str(exc)
            self._status_message = f"error: {exc}"
            self._connected_event.set()
            self._streaming_event.set()
            self._log(f"[EPOC X] {exc}")

    async def _listen(self, websocket) -> None:
        while not self._stop_event.is_set():
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            payload = json.loads(message)
            if "pow" in payload:
                self._handle_pow(payload)
            elif "dev" in payload:
                self._handle_dev(payload)

        try:
            await websocket.close()
        except Exception:
            pass

    def _handle_pow(self, payload: dict) -> None:
        values = payload.get("pow", [])
        timestamp = float(payload.get("time", time.time()))
        if len(values) != len(POW_COLUMNS):
            return

        sample = PowerSample(
            timestamp=timestamp,
            values={column: float(value) for column, value in zip(POW_COLUMNS, values)},
            sensor_contact_quality=self._sensor_contact_quality,
        )
        with self._lock:
            self._buffer.append(sample)

        if not self._streaming:
            self._streaming = True
            self._status_message = "streaming"
            self._streaming_event.set()
            self._log("[EPOC X] pow stream received. EEG is streaming.")

    def _handle_dev(self, payload: dict) -> None:
        try:
            dev_values = payload["dev"][2]
            values = list(dev_values)
            if values:
                self._sensor_contact_quality = float(sum(values) / len(values))
        except Exception:
            return

    async def _initialize_session(self, websocket) -> None:
        await self._request_access(websocket)
        await self._control_device(websocket, "refresh", request_id=2)
        await asyncio.sleep(1.0)

        headsets_resp = await self._query_headsets(websocket)
        headsets = headsets_resp.get("result", [])
        if not headsets:
            raise RuntimeError("No EPOC X headset found. Make sure Emotiv Launcher is running and the headset is connected.")

        self._headset_id = headsets[0]["id"]
        await self._control_device(websocket, "connect", request_id=4, headset_id=self._headset_id)

        auth_resp = await self._authorize(websocket)
        cortex_token = auth_resp["result"]["cortexToken"]

        session_resp = await self._create_session(websocket, cortex_token, self._headset_id)
        self._session_id = session_resp["result"]["id"]

        await self._subscribe_to_streams(websocket, cortex_token, self._session_id)

    async def _send_json_rpc(self, websocket, method: str, params: dict, request_id: int) -> dict:
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
        await websocket.send(json.dumps(payload))
        response = await websocket.recv()
        decoded = json.loads(response)
        if "error" in decoded:
            raise RuntimeError(f"Cortex {method} failed: {decoded['error']}")
        return decoded

    async def _request_access(self, websocket) -> dict:
        return await self._send_json_rpc(
            websocket,
            "requestAccess",
            {"clientId": self.client_id, "clientSecret": self.client_secret},
            request_id=1,
        )

    async def _control_device(self, websocket, command: str, request_id: int, headset_id: Optional[str] = None) -> dict:
        params = {"command": command}
        if headset_id:
            params["headset"] = headset_id
        return await self._send_json_rpc(websocket, "controlDevice", params, request_id)

    async def _query_headsets(self, websocket) -> dict:
        return await self._send_json_rpc(websocket, "queryHeadsets", {}, request_id=3)

    async def _authorize(self, websocket) -> dict:
        return await self._send_json_rpc(
            websocket,
            "authorize",
            {"clientId": self.client_id, "clientSecret": self.client_secret, "debit": 1},
            request_id=5,
        )

    async def _create_session(self, websocket, cortex_token: str, headset_id: str) -> dict:
        return await self._send_json_rpc(
            websocket,
            "createSession",
            {"cortexToken": cortex_token, "headset": headset_id, "status": "open"},
            request_id=6,
        )

    async def _subscribe_to_streams(self, websocket, cortex_token: str, session_id: str) -> dict:
        return await self._send_json_rpc(
            websocket,
            "subscribe",
            {"cortexToken": cortex_token, "session": session_id, "streams": ["pow", "dev"]},
            request_id=7,
        )


if __name__ == "__main__":
    stream = EpocXStream(verbose=True)
    stream.start()
    connected = stream.wait_until_connected(timeout=10)
    streaming = stream.wait_until_streaming(timeout=15)
    print(f"Connected: {connected and stream.connected}")
    print(f"Streaming: {stream.streaming and not stream.last_error}")
    print(f"Status: {stream.status_message}")
    try:
        while True:
            time.sleep(1)
            latest = stream.latest_psd_dict()
            print("Latest sample available:", latest is not None)
            print("Sensor contact quality:", stream.sensor_contact_quality)
    except KeyboardInterrupt:
        stream.stop()
