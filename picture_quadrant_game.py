from __future__ import annotations

import csv
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

from EPOCX import EpocXStream, POW_COLUMNS, PowerSample
from model import QuadrantModel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
AUTO_ADVANCE_MS = 2500
IMAGE_MAX_SIZE = (900, 650)
MAP_SIZE = 240
BG_COLOR = "#2f2f2f"
PANEL_COLOR = "#2f2f2f"
FG_COLOR = "#ffffff"
MUTED_TEXT_COLOR = "#9c9c9c"
MARKER_COLOR = "#4a90ff"
TRACK_COLOR = "#555555"
ANIMATION_STEPS = 24
ANIMATION_DELAY_MS = 18
MODEL_DELAY_MS = 280
IMAGE_CORNER_RADIUS = 28
COMPACT_LAYOUT_WIDTH = 1180
EEG_CONNECT_TIMEOUT_SECONDS = 12.0
EEG_STREAM_TIMEOUT_SECONDS = 12.0


@dataclass(frozen=True)
class Quadrant:
    key: str
    code: str
    label: str
    valence: str
    arousal: str
    grid_position: tuple[int, int]


QUADRANTS: dict[str, Quadrant] = {
    "1": Quadrant("1", "LALV", "Low Arousal / Low Valence", "Low", "Low", (0, 1)),
    "2": Quadrant("2", "LAHV", "Low Arousal / High Valence", "High", "Low", (1, 1)),
    "3": Quadrant("3", "HAHV", "High Arousal / High Valence", "High", "High", (1, 0)),
    "4": Quadrant("4", "HALV", "Low Valence / High Arousal", "Low", "High", (0, 0)),
}
QUADRANT_BY_CODE = {quadrant.code: quadrant for quadrant in QUADRANTS.values()}

QUADRANT_DESCRIPTIONS = {
    "1": "1 = sadness / boredom",
    "2": "2 = calmness / relaxation",
    "3": "3 = excitement / enjoyment",
    "4": "4 = frustration / anxiety",
}


class SessionRecorder:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.current_session_path = self.base_dir / "current_session.csv"
        self.full_dataset_path = self.base_dir / "datasets/full_dataset.csv"
        self.full_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        self._aggregated = False
        self._fieldnames = self._build_fieldnames()
        self.current_session_path.write_text("", encoding="utf-8")
        print(f"[SessionRecorder] Cleared {self.current_session_path}", flush=True)

    def _build_fieldnames(self) -> list[str]:
        return [
            "session_id",
            "trial_index",
            "image_name",
            "trial_started_at",
            "trial_ended_at",
            "time_elapsed_seconds",
            "sample_timestamp",
            "user_predicted_key",
            "user_predicted_code",
            "user_predicted_label",
            "model_predicted_key",
            "model_predicted_code",
            "model_predicted_label",
            "model_match",
            "sensor_contact_quality",
            *POW_COLUMNS,
        ]

    def append_rows(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        file_exists = self.current_session_path.exists() and self.current_session_path.stat().st_size > 0
        with self.current_session_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in self._fieldnames})

    def aggregate_to_full_dataset(self) -> Optional[Path]:
        if self._aggregated:
            return self.full_dataset_path
        if not self.current_session_path.exists() or self.current_session_path.stat().st_size == 0:
            self._aggregated = True
            return None

        target_exists = self.full_dataset_path.exists() and self.full_dataset_path.stat().st_size > 0
        with self.current_session_path.open("r", newline="", encoding="utf-8") as source, self.full_dataset_path.open(
            "a", newline="", encoding="utf-8"
        ) as target:
            reader = csv.reader(source)
            writer = csv.writer(target)
            try:
                header = next(reader)
            except StopIteration:
                self._aggregated = True
                return None

            if not target_exists:
                writer.writerow(header)
            for row in reader:
                writer.writerow(row)

        self._aggregated = True
        print(f"[SessionRecorder] Appended session rows into {self.full_dataset_path}", flush=True)
        return self.full_dataset_path


class AccuracyBar(tk.Frame):
    def __init__(self, master: tk.Misc, width: int = 320, **kwargs) -> None:
        super().__init__(master, bg=BG_COLOR, **kwargs)
        self.current_value = 0.0
        self.target_value = 0.0
        self.animation_job: Optional[str] = None
        self.width = width
        self.height = 28

        self.title_label = tk.Label(
            self,
            text="Model Accuracy",
            font=("Helvetica", 15, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.title_label.pack(anchor="e", pady=(0, 6))

        self.canvas = tk.Canvas(
            self,
            width=self.width,
            height=self.height,
            bg=BG_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(anchor="e", fill="x")

        self.value_label = tk.Label(
            self,
            text="0.0%",
            font=("Helvetica", 13),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.value_label.pack(anchor="e", pady=(6, 0))

        self._draw_bar(self.current_value)

    def resize(self, width: int) -> None:
        self.width = max(180, width)
        self.canvas.configure(width=self.width)
        self._draw_bar(self.current_value)

    def _draw_bar(self, percent: float) -> None:
        self.canvas.delete("all")
        x1 = 14
        x2 = max(x1 + 10, self.width - 14)
        y = self.height / 2
        line_width = 16

        self.canvas.create_line(
            x1,
            y,
            x2,
            y,
            fill=TRACK_COLOR,
            width=line_width,
            capstyle=tk.ROUND,
        )

        fill_end = x1 + (x2 - x1) * max(0.0, min(100.0, percent)) / 100.0
        self.canvas.create_line(
            x1,
            y,
            fill_end,
            y,
            fill=MARKER_COLOR,
            width=line_width,
            capstyle=tk.ROUND,
        )
        self.value_label.configure(text=f"{percent:.1f}%")

    def animate_to(self, new_value: float) -> None:
        new_value = max(0.0, min(100.0, new_value))
        self.target_value = new_value
        if self.animation_job is not None:
            self.after_cancel(self.animation_job)
            self.animation_job = None

        start = self.current_value
        delta = self.target_value - start
        steps = 26

        def step(i: int) -> None:
            progress = i / steps
            eased = 1 - (1 - progress) ** 3
            value = start + delta * eased
            self.current_value = value
            self._draw_bar(value)
            if i < steps:
                self.animation_job = self.after(18, lambda: step(i + 1))
            else:
                self.current_value = self.target_value
                self._draw_bar(self.current_value)
                self.animation_job = None

        step(1)


class QuadrantMap(tk.Frame):
    def __init__(self, master: tk.Misc, title: str, **kwargs) -> None:
        super().__init__(master, bg=PANEL_COLOR, **kwargs)
        self.map_size = MAP_SIZE
        self.current_quadrant: Optional[Quadrant] = None
        self.current_marker_text = "●"
        self.marker_id: Optional[int] = None
        self.marker_text_id: Optional[int] = None
        self.animation_job: Optional[str] = None

        self.title_label = tk.Label(
            self,
            text=title,
            font=("Helvetica", 16, "bold"),
            bg=PANEL_COLOR,
            fg=FG_COLOR,
        )
        self.title_label.pack(anchor="center", pady=(0, 10))

        self.canvas = tk.Canvas(
            self,
            width=self.map_size,
            height=self.map_size,
            bg=PANEL_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()
        self._draw_base_map()

    def resize(self, new_size: int) -> None:
        new_size = max(150, min(260, int(new_size)))
        if new_size == self.map_size:
            return
        self.map_size = new_size
        self.canvas.configure(width=self.map_size, height=self.map_size)
        self._draw_base_map()
        if self.current_quadrant is not None:
            x, y = self._quadrant_point(self.current_quadrant)
            self._draw_marker(x, y, self.current_marker_text)

    def _draw_base_map(self) -> None:
        c = self.canvas
        c.delete("all")
        self.marker_id = None
        self.marker_text_id = None

        mid = self.map_size / 2
        margin = max(14, int(self.map_size * 0.08))
        text_w = max(72, int(self.map_size * 0.38))
        label_font = ("Helvetica", max(10, int(self.map_size * 0.05)))

        c.create_line(mid, margin, mid, self.map_size - margin, width=2.2, fill=FG_COLOR)
        c.create_line(margin, mid, self.map_size - margin, mid, width=2.2, fill=FG_COLOR)

        c.create_text(self.map_size * 0.26, self.map_size * 0.26, text=QUADRANT_DESCRIPTIONS["4"], font=label_font, fill=MUTED_TEXT_COLOR, width=text_w, justify="center")
        c.create_text(self.map_size * 0.74, self.map_size * 0.26, text=QUADRANT_DESCRIPTIONS["3"], font=label_font, fill=MUTED_TEXT_COLOR, width=text_w, justify="center")
        c.create_text(self.map_size * 0.26, self.map_size * 0.74, text=QUADRANT_DESCRIPTIONS["1"], font=label_font, fill=MUTED_TEXT_COLOR, width=text_w, justify="center")
        c.create_text(self.map_size * 0.74, self.map_size * 0.74, text=QUADRANT_DESCRIPTIONS["2"], font=label_font, fill=MUTED_TEXT_COLOR, width=text_w, justify="center")

    def _center_point(self) -> tuple[float, float]:
        mid = self.map_size / 2
        return mid, mid

    def _quadrant_point(self, quadrant: Quadrant) -> tuple[float, float]:
        col, row = quadrant.grid_position
        mid = self.map_size / 2
        offset = self.map_size * 0.27
        x = mid - offset if col == 0 else mid + offset
        y = mid - offset if row == 0 else mid + offset
        return x, y

    def clear_selection(self) -> None:
        if self.animation_job is not None:
            self.after_cancel(self.animation_job)
            self.animation_job = None
        self.current_quadrant = None
        self._draw_base_map()

    def _draw_marker(self, x: float, y: float, marker_text: str) -> None:
        radius = max(10, int(self.map_size * 0.20))
        font_size = max(10, int(self.map_size * 0.05))
        if self.marker_id is None:
            self.marker_id = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=MARKER_COLOR, width=2, fill=MARKER_COLOR)
            self.marker_text_id = self.canvas.create_text(x, y, text=marker_text, font=("Helvetica", font_size, "bold"), fill=FG_COLOR)
        else:
            self.canvas.coords(self.marker_id, x - radius, y - radius, x + radius, y + radius)
            self.canvas.coords(self.marker_text_id, x, y)
            self.canvas.itemconfigure(self.marker_text_id, text=marker_text, font=("Helvetica", font_size, "bold"))

    def animate_to_selection(self, quadrant: Quadrant, marker_text: str = "●") -> None:
        if self.animation_job is not None:
            self.after_cancel(self.animation_job)
            self.animation_job = None

        self.current_quadrant = quadrant
        self.current_marker_text = marker_text
        self._draw_base_map()
        start_x, start_y = self._center_point()
        end_x, end_y = self._quadrant_point(quadrant)
        self._draw_marker(start_x, start_y, marker_text)

        def step(i: int) -> None:
            progress = i / ANIMATION_STEPS
            eased = 1 - (1 - progress) ** 3
            x = start_x + (end_x - start_x) * eased
            y = start_y + (end_y - start_y) * eased
            self._draw_marker(x, y, marker_text)
            if i < ANIMATION_STEPS:
                self.animation_job = self.after(ANIMATION_DELAY_MS, lambda: step(i + 1))
            else:
                self.animation_job = None

        step(1)


class PictureQuadrantGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Valence-Arousal Picture Game")
        self.root.geometry("1380x900")
        self.root.minsize(860, 620)
        self.root.configure(bg=BG_COLOR, padx=14, pady=14)

        self.base_dir = Path(__file__).resolve().parent
        self.pictures_dir = self.base_dir / "pictures"
        self.recorder = SessionRecorder(self.base_dir)
        self.eeg_stream = self._initialize_eeg_stream()
        self.model = QuadrantModel()

        self.image_paths = self._load_picture_paths()
        self.current_image_path: Optional[Path] = None
        self.current_pil_image: Optional[Image.Image] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.last_choice: Optional[str] = None
        self.auto_advance_job: Optional[str] = None
        self.model_reveal_job: Optional[str] = None
        self.layout_job: Optional[str] = None
        self.compact_layout = False
        self.trial_index = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.trial_started_perf = time.perf_counter()
        self.trial_started_epoch = time.time()
        self.pending_model_choice: Optional[Quadrant] = None
        self.pending_model_available = False

        self._build_layout()
        self._bind_keys()
        self.root.bind("<Configure>", self._schedule_layout_refresh)
        self.root.protocol("WM_DELETE_WINDOW", self._quit_and_aggregate)
        self.show_next_picture()
        self.root.after(50, self._refresh_layout)
        self.root.after(500, self._refresh_status_text)

    def _initialize_eeg_stream(self) -> EpocXStream:
        stream = EpocXStream(verbose=True)
        print("[Game] Starting EPOC X stream...", flush=True)
        stream.start()

        connected = stream.wait_until_connected(timeout=EEG_CONNECT_TIMEOUT_SECONDS)
        if not connected or not stream.connected:
            stream.stop()
            raise RuntimeError(
                "EPOC X failed to connect before game start. "
                f"Status: {stream.status_message}. "
                f"Last error: {stream.last_error or 'none'}"
            )
        print(f"[Game] EPOC X connected: headset={stream.headset_id}", flush=True)

        streaming = stream.wait_until_streaming(timeout=EEG_STREAM_TIMEOUT_SECONDS)
        if not streaming or not stream.streaming:
            stream.stop()
            raise RuntimeError(
                "EPOC X connected but pow stream did not start before game start. "
                f"Status: {stream.status_message}. "
                f"Last error: {stream.last_error or 'none'}"
            )
        print("[Game] EPOC X is streaming.", flush=True)
        return stream

    def _load_picture_paths(self) -> list[Path]:
        if not self.pictures_dir.exists():
            raise FileNotFoundError(
                f"Could not find a 'pictures' folder at: {self.pictures_dir}\n"
                "Create that folder next to this script and add image files to it."
            )

        image_paths = sorted([p for p in self.pictures_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        if not image_paths:
            raise FileNotFoundError(f"The folder {self.pictures_dir} exists but contains no supported image files.")
        return image_paths

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=5)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        self.left_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.left_frame.rowconfigure(1, weight=1)
        self.left_frame.columnconfigure(0, weight=1)

        self.title = tk.Label(self.left_frame, text="1 = LALV   2 = LAHV   3 = HAHV   4 = HALV", font=("Helvetica", 24, "bold"), bg=BG_COLOR, fg=FG_COLOR)
        self.title.grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.image_panel = tk.Frame(self.left_frame, bg=PANEL_COLOR)
        self.image_panel.grid(row=1, column=0, sticky="nsew")
        self.image_panel.rowconfigure(0, weight=1)
        self.image_panel.columnconfigure(0, weight=1)

        self.image_label = tk.Label(self.image_panel, bg=PANEL_COLOR, fg=FG_COLOR, bd=0, highlightthickness=0)
        self.image_label.grid(row=0, column=0)

        self.right_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.columnconfigure(0, weight=1)

        self.info_panel = tk.Frame(self.right_frame, bg=PANEL_COLOR, padx=16, pady=14)
        self.info_panel.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        self.info_label = tk.Label(
            self.info_panel,
            text=(
                "Look at the image, then press 1-4 to classify your emotion. "
                "Press Enter for the next image. Press Q to append current_session.csv into full_dataset.csv and quit."
            ),
            justify="left",
            anchor="w",
            font=("Helvetica", 13),
            bg=PANEL_COLOR,
            fg=FG_COLOR,
            wraplength=340,
        )
        self.info_label.pack(anchor="w", fill="x")

        self.status_label = tk.Label(
            self.info_panel,
            text="EEG connected and streaming.",
            justify="left",
            anchor="w",
            font=("Helvetica", 11),
            bg=PANEL_COLOR,
            fg=MUTED_TEXT_COLOR,
            wraplength=340,
        )
        self.status_label.pack(anchor="w", fill="x", pady=(10, 0))

        self.user_panel = tk.Frame(self.right_frame, bg=PANEL_COLOR, padx=16, pady=14)
        self.user_panel.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        self.user_map = QuadrantMap(self.user_panel, title="User Selection")
        self.user_map.pack(anchor="center")

        self.model_panel = tk.Frame(self.right_frame, bg=PANEL_COLOR, padx=16, pady=14)
        self.model_panel.grid(row=2, column=0, sticky="ew")
        self.model_map = QuadrantMap(self.model_panel, title="Model Prediction")
        self.model_map.pack(anchor="center")

        self.bottom_row = tk.Frame(self.right_frame, bg=BG_COLOR)
        self.bottom_row.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        self.bottom_row.columnconfigure(0, weight=1)

        self.accuracy_bar = AccuracyBar(self.bottom_row)
        self.accuracy_bar.grid(row=0, column=1, sticky="e")

    def _bind_keys(self) -> None:
        for key in QUADRANTS:
            self.root.bind(key, self._handle_quadrant_key)
            self.root.bind(f"<KP_{key}>", self._handle_quadrant_key)

        self.root.bind("<Return>", lambda _event: self.show_next_picture())
        self.root.bind("q", lambda _event: self._quit_and_aggregate())
        self.root.bind("Q", lambda _event: self._quit_and_aggregate())
        self.root.bind("<Escape>", lambda _event: self._quit_and_aggregate())

    def _choose_random_image(self) -> Path:
        if len(self.image_paths) == 1:
            return self.image_paths[0]
        options = [p for p in self.image_paths if p != self.current_image_path]
        return random.choice(options)

    def _cancel_pending_jobs(self) -> None:
        if self.auto_advance_job is not None:
            self.root.after_cancel(self.auto_advance_job)
            self.auto_advance_job = None
        if self.model_reveal_job is not None:
            self.root.after_cancel(self.model_reveal_job)
            self.model_reveal_job = None

    def _rounded_photo_image(self, image: Image.Image) -> ImageTk.PhotoImage:
        rounded = image.convert("RGBA")
        mask = Image.new("L", rounded.size, 0)
        draw = ImageDraw.Draw(mask)
        radius = max(0, min(IMAGE_CORNER_RADIUS, min(rounded.size) // 2))
        draw.rounded_rectangle((0, 0, rounded.size[0], rounded.size[1]), radius=radius, fill=255)
        rounded.putalpha(mask)
        return ImageTk.PhotoImage(rounded)

    def _display_current_image(self) -> None:
        if self.current_pil_image is None:
            return

        self.root.update_idletasks()
        panel_w = max(320, self.image_panel.winfo_width() - 24)
        panel_h = max(220, self.image_panel.winfo_height() - 24)
        max_w = min(IMAGE_MAX_SIZE[0], panel_w)
        max_h = min(IMAGE_MAX_SIZE[1], panel_h)

        image = self.current_pil_image.copy()
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.current_photo = self._rounded_photo_image(image)
        self.image_label.configure(image=self.current_photo)

    def _schedule_layout_refresh(self, _event: tk.Event | None = None) -> None:
        if self.layout_job is not None:
            self.root.after_cancel(self.layout_job)
        self.layout_job = self.root.after(40, self._refresh_layout)

    def _apply_layout_mode(self, compact: bool) -> None:
        if compact == self.compact_layout:
            return
        self.compact_layout = compact

        self.left_frame.grid_forget()
        self.right_frame.grid_forget()

        if compact:
            self.root.columnconfigure(0, weight=1)
            self.root.columnconfigure(1, weight=0)
            self.root.rowconfigure(0, weight=3)
            self.root.rowconfigure(1, weight=2)
            self.left_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 12))
            self.right_frame.grid(row=1, column=0, sticky="nsew")
        else:
            self.root.columnconfigure(0, weight=5)
            self.root.columnconfigure(1, weight=2)
            self.root.rowconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=0)
            self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 18), pady=0)
            self.right_frame.grid(row=0, column=1, sticky="nsew")

    def _refresh_layout(self) -> None:
        self.layout_job = None
        self.root.update_idletasks()

        root_w = max(1, self.root.winfo_width())
        root_h = max(1, self.root.winfo_height())
        compact = root_w < COMPACT_LAYOUT_WIDTH
        self._apply_layout_mode(compact)

        outer_pad = max(8, min(18, root_w // 90))
        gap_pad = max(8, min(18, root_w // 80))
        self.root.configure(padx=outer_pad, pady=outer_pad)
        self.left_frame.grid_configure(padx=(0, gap_pad if not compact else 0), pady=(0, gap_pad if compact else 0))
        self.right_frame.grid_configure(padx=0, pady=0)

        title_size = 24 if root_w >= 1280 else 22 if root_w >= 1100 else 20
        info_size = 13 if root_w >= 1050 else 12
        self.title.configure(font=("Helvetica", title_size, "bold"))
        self.info_label.configure(font=("Helvetica", info_size))
        self.info_label.configure(wraplength=max(240, self.right_frame.winfo_width() - 36))
        self.status_label.configure(wraplength=max(240, self.right_frame.winfo_width() - 36))

        panel_pad_x = max(10, min(16, root_w // 100))
        panel_pad_y = max(8, min(14, root_h // 80))
        inner_gap = max(8, min(14, root_h // 90))

        for panel in (self.info_panel, self.user_panel, self.model_panel):
            panel.configure(padx=panel_pad_x, pady=panel_pad_y)

        self.info_panel.grid_configure(pady=(0, inner_gap))
        self.user_panel.grid_configure(pady=(0, inner_gap))
        self.model_panel.grid_configure(pady=(0, 0))
        self.bottom_row.grid_configure(pady=(inner_gap, 0))

        right_w = max(240, self.right_frame.winfo_width())
        right_h = max(280, self.right_frame.winfo_height())
        width_based_size = right_w - (52 if compact else 44)
        height_budget = right_h - 140
        height_based_size = height_budget // 2
        map_size = min(240, max(150, width_based_size, 0), max(150, height_based_size, 0))
        self.user_map.resize(map_size)
        self.model_map.resize(map_size)

        acc_width = min(360, max(210, right_w - 20))
        self.accuracy_bar.resize(acc_width)

        self._display_current_image()

    def _refresh_status_text(self) -> None:
        eeg_status = self.eeg_stream.status_message
        contact = self.eeg_stream.sensor_contact_quality
        contact_text = "n/a" if contact is None else f"{contact:.2f}"
        model_status = self.model.status_text()
        self.status_label.configure(
            text=(
                f"EEG: {eeg_status}\n"
                f"Sensor contact quality: {contact_text}\n"
                f"Model: {model_status}\n"
                f"Session file: {self.recorder.current_session_path.name}"
            )
        )
        self.root.after(1000, self._refresh_status_text)

    def show_next_picture(self) -> None:
        self._cancel_pending_jobs()
        self.last_choice = None
        self.pending_model_choice = None
        self.pending_model_available = False
        self.trial_index += 1
        self.current_image_path = self._choose_random_image()
        self.current_pil_image = Image.open(self.current_image_path).copy()
        self.trial_started_perf = time.perf_counter()
        self.trial_started_epoch = time.time()

        self.user_map.clear_selection()
        self.model_map.clear_selection()
        self._display_current_image()

    def _update_accuracy(self, is_correct: bool) -> float:
        self.total_predictions += 1
        if is_correct:
            self.correct_predictions += 1
        accuracy = (self.correct_predictions / self.total_predictions) * 100.0
        self.accuracy_bar.animate_to(accuracy)
        return accuracy

    def _reveal_model_choice(self) -> None:
        if self.pending_model_choice is not None:
            self.model_map.animate_to_selection(self.pending_model_choice, marker_text="M")
            if self.last_choice is not None and self.pending_model_available:
                user_choice = QUADRANTS[self.last_choice]
                is_correct = self.pending_model_choice.key == user_choice.key
                self._update_accuracy(is_correct)
        self.auto_advance_job = self.root.after(AUTO_ADVANCE_MS, self.show_next_picture)

    def _predict_model_choice(self, samples: list[PowerSample]) -> Optional[Quadrant]:
        features = self.eeg_stream.mean_psd_from_samples(samples)
        if features is None:
            return None
        code = self.model.predict_code(features)
        if code is None:
            return None
        return QUADRANT_BY_CODE.get(code)

    def _build_rows_for_trial(
        self,
        user_choice: Quadrant,
        model_choice: Optional[Quadrant],
        elapsed_seconds: float,
        trial_ended_epoch: float,
        samples: list[PowerSample],
    ) -> list[dict[str, object]]:
        trial_started_at = datetime.fromtimestamp(self.trial_started_epoch, tz=timezone.utc).isoformat()
        trial_ended_at = datetime.fromtimestamp(trial_ended_epoch, tz=timezone.utc).isoformat()
        model_match = int(model_choice.key == user_choice.key) if model_choice else ""

        rows: list[dict[str, object]] = []
        for sample in samples:
            row: dict[str, object] = {
                "session_id": self.recorder.session_id,
                "trial_index": self.trial_index,
                "image_name": self.current_image_path.name if self.current_image_path else "",
                "trial_started_at": trial_started_at,
                "trial_ended_at": trial_ended_at,
                "time_elapsed_seconds": round(elapsed_seconds, 6),
                "sample_timestamp": datetime.fromtimestamp(sample.timestamp, tz=timezone.utc).isoformat(),
                "user_predicted_key": user_choice.key,
                "user_predicted_code": user_choice.code,
                "user_predicted_label": user_choice.label,
                "model_predicted_key": model_choice.key if model_choice else "",
                "model_predicted_code": model_choice.code if model_choice else "",
                "model_predicted_label": model_choice.label if model_choice else "",
                "model_match": model_match,
                "sensor_contact_quality": "" if sample.sensor_contact_quality is None else round(sample.sensor_contact_quality, 6),
            }
            for column in POW_COLUMNS:
                value = sample.values.get(column)
                row[column] = "" if value is None else round(float(value), 8)
            rows.append(row)

        if rows:
            return rows

        blank_row: dict[str, object] = {
            "session_id": self.recorder.session_id,
            "trial_index": self.trial_index,
            "image_name": self.current_image_path.name if self.current_image_path else "",
            "trial_started_at": trial_started_at,
            "trial_ended_at": trial_ended_at,
            "time_elapsed_seconds": round(elapsed_seconds, 6),
            "sample_timestamp": "",
            "user_predicted_key": user_choice.key,
            "user_predicted_code": user_choice.code,
            "user_predicted_label": user_choice.label,
            "model_predicted_key": model_choice.key if model_choice else "",
            "model_predicted_code": model_choice.code if model_choice else "",
            "model_predicted_label": model_choice.label if model_choice else "",
            "model_match": model_match,
            "sensor_contact_quality": "" if self.eeg_stream.sensor_contact_quality is None else round(self.eeg_stream.sensor_contact_quality, 6),
        }
        for column in POW_COLUMNS:
            blank_row[column] = ""
        return [blank_row]

    def _handle_quadrant_key(self, event: tk.Event) -> None:
        key = str(event.keysym).replace("KP_", "")
        if key not in QUADRANTS:
            return
        if self.last_choice is not None:
            return

        self._cancel_pending_jobs()
        self.last_choice = key
        user_choice = QUADRANTS[key]
        trial_ended_epoch = time.time()
        elapsed_seconds = time.perf_counter() - self.trial_started_perf
        samples = self.eeg_stream.samples_between(self.trial_started_epoch, trial_ended_epoch, fallback_latest=False)
        model_choice = self._predict_model_choice(samples)

        rows = self._build_rows_for_trial(user_choice, model_choice, elapsed_seconds, trial_ended_epoch, samples)
        self.recorder.append_rows(rows)
        print(f"[Game] Saved {len(rows)} EEG row(s) for trial {self.trial_index}.", flush=True)

        self.pending_model_choice = model_choice
        self.pending_model_available = model_choice is not None
        self.user_map.animate_to_selection(user_choice, marker_text="U")
        self.model_reveal_job = self.root.after(MODEL_DELAY_MS, self._reveal_model_choice)

    def _quit_and_aggregate(self) -> None:
        self._cancel_pending_jobs()
        self.recorder.aggregate_to_full_dataset()
        self.eeg_stream.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PictureQuadrantGame(root)
    root.mainloop()
