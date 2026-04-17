from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

from EPOCX import EpocXStream, POW_COLUMNS
from model import QuadrantModel

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
AUTO_ADVANCE_MS = 2500
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
STREAM_READY_TIMEOUT_SECONDS = 20.0
IMAGE_PANEL_PADDING = 6
EEG_POLL_INTERVAL_MS = 200


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
        self.full_dataset_path = self.base_dir / "datasets" / "full_dataset.csv"
        self.full_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        self._aggregated = False
        self._fieldnames = [
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
        self._initialize_current_session_file()

    @property
    def fieldnames(self) -> list[str]:
        return list(self._fieldnames)

    def _initialize_current_session_file(self) -> None:
        with self.current_session_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def append_rows(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with self.current_session_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in self._fieldnames})

    def finalize_trial_rows(
        self,
        trial_index: int,
        sample_timestamps: set[str],
        finalized_values: dict[str, object],
    ) -> int:
        """
        Rewrite current_session.csv and update only the rows that belong to the
        given trial and whose sample_timestamp is in sample_timestamps.
        """
        if not self.current_session_path.exists():
            return 0

        updated_count = 0
        with self.current_session_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)

        for row in rows:
            if (
                row.get("session_id") == self.session_id
                and str(row.get("trial_index", "")) == str(trial_index)
                and row.get("sample_timestamp", "") in sample_timestamps
            ):
                for key, value in finalized_values.items():
                    if key in self._fieldnames:
                        row[key] = value
                updated_count += 1

        with self.current_session_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in self._fieldnames})

        return updated_count

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
        self.canvas.create_line(x1, y, x2, y, fill=TRACK_COLOR, width=16, capstyle=tk.ROUND)
        fill_end = x1 + (x2 - x1) * max(0.0, min(100.0, percent)) / 100.0
        self.canvas.create_line(x1, y, fill_end, y, fill=MARKER_COLOR, width=16, capstyle=tk.ROUND)
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

        c.create_text(
            self.map_size * 0.26,
            self.map_size * 0.26,
            text=QUADRANT_DESCRIPTIONS["4"],
            font=label_font,
            fill=MUTED_TEXT_COLOR,
            width=text_w,
            justify="center",
        )
        c.create_text(
            self.map_size * 0.74,
            self.map_size * 0.26,
            text=QUADRANT_DESCRIPTIONS["3"],
            font=label_font,
            fill=MUTED_TEXT_COLOR,
            width=text_w,
            justify="center",
        )
        c.create_text(
            self.map_size * 0.26,
            self.map_size * 0.74,
            text=QUADRANT_DESCRIPTIONS["1"],
            font=label_font,
            fill=MUTED_TEXT_COLOR,
            width=text_w,
            justify="center",
        )
        c.create_text(
            self.map_size * 0.74,
            self.map_size * 0.74,
            text=QUADRANT_DESCRIPTIONS["2"],
            font=label_font,
            fill=MUTED_TEXT_COLOR,
            width=text_w,
            justify="center",
        )

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
            self.marker_id = self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                outline=MARKER_COLOR,
                width=2,
                fill=MARKER_COLOR,
            )
            self.marker_text_id = self.canvas.create_text(
                x,
                y,
                text=marker_text,
                font=("Helvetica", font_size, "bold"),
                fill=FG_COLOR,
            )
        else:
            self.canvas.coords(self.marker_id, x - radius, y - radius, x + radius, y + radius)
            self.canvas.coords(self.marker_text_id, x, y)
            self.canvas.itemconfigure(
                self.marker_text_id,
                text=marker_text,
                font=("Helvetica", font_size, "bold"),
            )

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
        self.online_state_dir = self.base_dir / "online_state"
        self.online_state_dir.mkdir(parents=True, exist_ok=True)
        self.training_status_path = self.online_state_dir / "training_status.json"

        self.recorder = SessionRecorder(self.base_dir)
        self.eeg_stream = EpocXStream(verbose=True)
        self.eeg_available = False
        self.finetune_process: Optional[subprocess.Popen[str]] = None
        self.finetune_status_text = "idle"

        try:
            self.eeg_stream.start()
            self.eeg_stream.require_ready(timeout=STREAM_READY_TIMEOUT_SECONDS)
            self.eeg_available = True
            print("EPOC X stream is ready. Loading model and starting game.", flush=True)
        except Exception as exc:
            self.eeg_available = False
            print(
                f"No EPOC X stream detected after {STREAM_READY_TIMEOUT_SECONDS:.0f} seconds. "
                f"Starting game without headset. ({exc})",
                flush=True,
            )

        self.model = QuadrantModel()

        self.image_paths = self._load_picture_paths()
        self.current_image_path: Optional[Path] = None
        self.current_pil_image: Optional[Image.Image] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self.last_choice: Optional[str] = None
        self.auto_advance_job: Optional[str] = None
        self.model_reveal_job: Optional[str] = None
        self.layout_job: Optional[str] = None
        self.eeg_poll_job: Optional[str] = None

        self.compact_layout = False
        self.trial_index = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.trial_started_perf = time.perf_counter()
        self.trial_started_epoch = time.time()
        self.pending_model_choice: Optional[Quadrant] = None
        self.pending_model_available = False

        self.current_trial_samples: list[dict[str, float]] = []
        self.current_trial_sample_timestamp_strings: set[str] = set()
        self.last_eeg_poll_epoch: float = 0.0

        self._build_layout()
        self._bind_keys()

        self.root.bind("<Configure>", self._schedule_layout_refresh)
        self.root.protocol("WM_DELETE_WINDOW", self._quit_and_aggregate)

        self.show_next_picture()
        self.root.after(50, self._refresh_layout)
        self.root.after(500, self._refresh_status_text)

    def _load_picture_paths(self) -> list[Path]:
        if not self.pictures_dir.exists():
            raise FileNotFoundError(f"Could not find a 'pictures' folder at: {self.pictures_dir}")
        image_paths = sorted(
            [p for p in self.pictures_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not image_paths:
            raise FileNotFoundError(f"The folder {self.pictures_dir} contains no supported image files.")
        return image_paths

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=5)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        self.left_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.left_frame.rowconfigure(0, weight=0)
        self.left_frame.rowconfigure(1, weight=1)
        self.left_frame.columnconfigure(0, weight=1)

        self.title = tk.Label(
            self.left_frame,
            text="1 = LALV   2 = LAHV   3 = HAHV   4 = HALV",
            font=("Helvetica", 24, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.title.grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.image_panel = tk.Frame(self.left_frame, bg=PANEL_COLOR)
        self.image_panel.grid(row=1, column=0, sticky="nsew")
        self.image_panel.rowconfigure(0, weight=1)
        self.image_panel.columnconfigure(0, weight=1)

        self.image_label = tk.Label(
            self.image_panel,
            bg=PANEL_COLOR,
            fg=FG_COLOR,
            bd=0,
            highlightthickness=0,
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=IMAGE_PANEL_PADDING, pady=IMAGE_PANEL_PADDING)

        self.right_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.columnconfigure(0, weight=1)

        self.info_panel = tk.Frame(self.right_frame, bg=PANEL_COLOR, padx=16, pady=14)
        self.info_panel.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        self.info_label = tk.Label(
            self.info_panel,
            text=(
                "Look at the image, then press 1-4 to classify your emotion. "
                "Press Enter for the next image. Press Q to append current_session.csv "
                "into datasets/full_dataset.csv and quit."
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
            text="EEG stream ready.",
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
        if self.eeg_poll_job is not None:
            self.root.after_cancel(self.eeg_poll_job)
            self.eeg_poll_job = None

    def _rounded_photo_image(self, image: Image.Image) -> ImageTk.PhotoImage:
        rounded = image.convert("RGBA")
        mask = Image.new("L", rounded.size, 0)
        draw = ImageDraw.Draw(mask)
        radius = max(0, min(IMAGE_CORNER_RADIUS, min(rounded.size) // 2))
        draw.rounded_rectangle((0, 0, rounded.size[0], rounded.size[1]), radius=radius, fill=255)
        rounded.putalpha(mask)
        return ImageTk.PhotoImage(rounded)

    def _resize_image_to_fit(self, image: Image.Image, max_w: int, max_h: int) -> Image.Image:
        max_w = max(1, int(max_w))
        max_h = max(1, int(max_h))

        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            return image.copy()

        scale = min(max_w / src_w, max_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        if (new_w, new_h) == image.size:
            return image.copy()

        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _display_current_image(self) -> None:
        if self.current_pil_image is None:
            return

        self.root.update_idletasks()

        panel_w = self.image_panel.winfo_width()
        panel_h = self.image_panel.winfo_height()

        available_w = max(1, panel_w - 2 * IMAGE_PANEL_PADDING)
        available_h = max(1, panel_h - 2 * IMAGE_PANEL_PADDING)

        if available_w < 10 or available_h < 10:
            self.root.after(20, self._display_current_image)
            return

        image = self._resize_image_to_fit(self.current_pil_image, available_w, available_h)
        self.current_photo = self._rounded_photo_image(image)
        self.image_label.configure(image=self.current_photo)

    def _schedule_layout_refresh(self, _event: tk.Event | None = None) -> None:
        if self.layout_job is not None:
            self.root.after_cancel(self.layout_job)
        self.layout_job = self.root.after(30, self._refresh_layout)

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
        self.root.configure(padx=outer_pad, pady=outer_pad)

        right_w = max(240, self.right_frame.winfo_width())
        right_h = max(280, self.right_frame.winfo_height())

        self.info_label.configure(wraplength=max(240, right_w - 36))
        self.status_label.configure(wraplength=max(240, right_w - 36))

        map_size = min(240, max(150, right_w - 44), max(150, (right_h - 140) // 2))
        self.user_map.resize(map_size)
        self.model_map.resize(map_size)

        self.accuracy_bar.resize(min(360, max(210, right_w - 20)))

        title_size = max(16, min(24, root_w // 55))
        self.title.configure(font=("Helvetica", title_size, "bold"))

        self._display_current_image()

    def _poll_finetune_process(self) -> None:
        if self.training_status_path.exists():
            try:
                payload = json.loads(self.training_status_path.read_text(encoding="utf-8"))
                self.finetune_status_text = str(payload.get("message") or payload.get("status") or "idle")
            except Exception:
                self.finetune_status_text = "status file unreadable"

        if self.finetune_process is not None:
            return_code = self.finetune_process.poll()
            if return_code is not None:
                self.finetune_process = None
                if return_code == 0:
                    self.model = QuadrantModel()
                    self.finetune_status_text = "fine-tune finished; reloaded model"
                else:
                    self.finetune_status_text = f"fine-tune failed (exit {return_code})"

    def _refresh_status_text(self) -> None:
        self._poll_finetune_process()
        if self.eeg_available:
            status_prefix = "EEG stream ready."
        else:
            status_prefix = "No headset detected. Running without EEG."
        self.status_label.configure(
            text=f"{status_prefix} Online fine-tune status: {self.finetune_status_text}"
        )
        self.root.after(1000, self._refresh_status_text)

    def _safe_samples_between(
        self,
        start_epoch: float,
        end_epoch: float,
        fallback_latest: bool,
    ) -> list[dict[str, float]]:
        if not self.eeg_available:
            return []
        try:
            return self.eeg_stream.samples_between(start_epoch, end_epoch, fallback_latest=fallback_latest)
        except TypeError:
            return self.eeg_stream.samples_between(start_epoch, end_epoch)
        except Exception:
            return []

    def _sample_timestamp_to_iso(self, sample_ts: float) -> str:
        return datetime.fromtimestamp(sample_ts, tz=timezone.utc).isoformat()

    def _build_stream_rows(
        self,
        samples: list[dict[str, float]],
    ) -> list[dict[str, object]]:
        started_at = datetime.fromtimestamp(self.trial_started_epoch, tz=timezone.utc).isoformat()
        rows: list[dict[str, object]] = []

        for sample in samples:
            sample_ts = float(sample.get("timestamp", time.time()))
            row: dict[str, object] = {
                "session_id": self.recorder.session_id,
                "trial_index": self.trial_index,
                "image_name": self.current_image_path.name if self.current_image_path else "",
                "trial_started_at": started_at,
                "trial_ended_at": "",
                "time_elapsed_seconds": "",
                "sample_timestamp": self._sample_timestamp_to_iso(sample_ts),
                "user_predicted_key": "",
                "user_predicted_code": "",
                "user_predicted_label": "",
                "model_predicted_key": "",
                "model_predicted_code": "",
                "model_predicted_label": "",
                "model_match": "",
                "sensor_contact_quality": "",
            }

            for column in POW_COLUMNS:
                value = sample.get(column, "")
                row[column] = round(float(value), 8) if value not in ("", None) else ""

            rows.append(row)

        return rows

    def _start_trial_capture(self) -> None:
        self.current_trial_samples = []
        self.current_trial_sample_timestamp_strings = set()
        self.last_eeg_poll_epoch = self.trial_started_epoch
        if self.eeg_available:
            self._schedule_eeg_poll()

    def _schedule_eeg_poll(self) -> None:
        if not self.eeg_available:
            return
        if self.eeg_poll_job is not None:
            self.root.after_cancel(self.eeg_poll_job)
        self.eeg_poll_job = self.root.after(EEG_POLL_INTERVAL_MS, self._poll_eeg_during_trial)

    def _poll_eeg_during_trial(self) -> None:
        self.eeg_poll_job = None

        if not self.eeg_available or self.last_choice is not None:
            return

        now_epoch = time.time()
        new_samples = self._safe_samples_between(self.last_eeg_poll_epoch, now_epoch, fallback_latest=False)
        self.last_eeg_poll_epoch = now_epoch

        deduped_samples: list[dict[str, float]] = []
        for sample in new_samples:
            try:
                sample_ts = float(sample.get("timestamp", now_epoch))
            except Exception:
                sample_ts = now_epoch

            sample_ts_iso = self._sample_timestamp_to_iso(sample_ts)
            if sample_ts_iso in self.current_trial_sample_timestamp_strings:
                continue

            self.current_trial_sample_timestamp_strings.add(sample_ts_iso)
            self.current_trial_samples.append(sample)
            deduped_samples.append(sample)

        if deduped_samples:
            stream_rows = self._build_stream_rows(deduped_samples)
            self.recorder.append_rows(stream_rows)

        self._schedule_eeg_poll()

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
        self._start_trial_capture()

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
                self._update_accuracy(self.pending_model_choice.key == user_choice.key)

        self.auto_advance_job = self.root.after(AUTO_ADVANCE_MS, self.show_next_picture)

    def _predict_model_choice(self, samples: list[dict[str, float]]) -> Optional[Quadrant]:
        if not samples:
            return None

        model_samples: list[dict[str, object]] = []
        for sample in samples:
            row = dict(sample)
            row.setdefault("trial_index", self.trial_index)
            row.setdefault("image_name", self.current_image_path.name if self.current_image_path else "")
            row.setdefault("session_id", self.recorder.session_id)
            model_samples.append(row)

        code = self.model.predict_code(model_samples)
        return QUADRANT_BY_CODE.get(code) if code else None

    def _maybe_start_finetune(self) -> None:
        if not self.eeg_available:
            return
        if self.trial_index < 5 or self.trial_index % 5 != 0:
            return
        if self.finetune_process is not None and self.finetune_process.poll() is None:
            self.finetune_status_text = "training already running"
            return

        script_path = self.base_dir / "online_finetune.py"
        if not script_path.exists():
            self.finetune_status_text = "online_finetune.py not found"
            return

        cmd = [
            sys.executable,
            str(script_path),
            "--round-count",
            str(self.trial_index),
            "--current-session",
            str(self.recorder.current_session_path),
        ]
        self.finetune_process = subprocess.Popen(
            cmd,
            cwd=str(self.base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.finetune_status_text = f"started after round {self.trial_index}"

    def _finalize_current_trial_rows(
        self,
        user_choice: Quadrant,
        model_choice: Optional[Quadrant],
        elapsed_seconds: float,
        trial_ended_epoch: float,
    ) -> int:
        sensor_contact_quality = self.eeg_stream.sensor_contact_quality if self.eeg_available else None
        ended_at = datetime.fromtimestamp(trial_ended_epoch, tz=timezone.utc).isoformat()

        finalized_values: dict[str, object] = {
            "trial_ended_at": ended_at,
            "time_elapsed_seconds": round(elapsed_seconds, 6),
            "user_predicted_key": user_choice.key,
            "user_predicted_code": user_choice.code,
            "user_predicted_label": user_choice.label,
            "model_predicted_key": model_choice.key if model_choice else "",
            "model_predicted_code": model_choice.code if model_choice else "",
            "model_predicted_label": model_choice.label if model_choice else "",
            "model_match": int(model_choice.key == user_choice.key) if model_choice else "",
            "sensor_contact_quality": "" if sensor_contact_quality is None else round(sensor_contact_quality, 6),
        }

        updated_count = self.recorder.finalize_trial_rows(
            trial_index=self.trial_index,
            sample_timestamps=self.current_trial_sample_timestamp_strings,
            finalized_values=finalized_values,
        )

        if updated_count == 0:
            fallback_row: dict[str, object] = {
                "session_id": self.recorder.session_id,
                "trial_index": self.trial_index,
                "image_name": self.current_image_path.name if self.current_image_path else "",
                "trial_started_at": datetime.fromtimestamp(self.trial_started_epoch, tz=timezone.utc).isoformat(),
                "trial_ended_at": ended_at,
                "time_elapsed_seconds": round(elapsed_seconds, 6),
                "sample_timestamp": ended_at,
                "user_predicted_key": user_choice.key,
                "user_predicted_code": user_choice.code,
                "user_predicted_label": user_choice.label,
                "model_predicted_key": model_choice.key if model_choice else "",
                "model_predicted_code": model_choice.code if model_choice else "",
                "model_predicted_label": model_choice.label if model_choice else "",
                "model_match": int(model_choice.key == user_choice.key) if model_choice else "",
                "sensor_contact_quality": "" if sensor_contact_quality is None else round(sensor_contact_quality, 6),
            }
            for column in POW_COLUMNS:
                fallback_row[column] = ""
            self.recorder.append_rows([fallback_row])
            updated_count = 1

        return updated_count

    def _handle_quadrant_key(self, event: tk.Event) -> None:
        key = str(event.keysym).replace("KP_", "")
        if key not in QUADRANTS or self.last_choice is not None:
            return

        trial_ended_epoch = time.time()
        elapsed_seconds = time.perf_counter() - self.trial_started_perf

        self._cancel_pending_jobs()
        self.last_choice = key

        final_samples = self._safe_samples_between(self.last_eeg_poll_epoch, trial_ended_epoch, fallback_latest=True)
        deduped_final_samples: list[dict[str, float]] = []
        for sample in final_samples:
            try:
                sample_ts = float(sample.get("timestamp", trial_ended_epoch))
            except Exception:
                sample_ts = trial_ended_epoch

            sample_ts_iso = self._sample_timestamp_to_iso(sample_ts)
            if sample_ts_iso in self.current_trial_sample_timestamp_strings:
                continue

            self.current_trial_sample_timestamp_strings.add(sample_ts_iso)
            self.current_trial_samples.append(sample)
            deduped_final_samples.append(sample)

        if deduped_final_samples:
            final_stream_rows = self._build_stream_rows(deduped_final_samples)
            self.recorder.append_rows(final_stream_rows)

        user_choice = QUADRANTS[key]
        if self.eeg_available and not self.current_trial_samples:
            self.finetune_status_text = "warning: no in-range EEG rows; saved metadata row only"

        model_choice = self._predict_model_choice(self.current_trial_samples) if self.current_trial_samples else None
        updated_count = self._finalize_current_trial_rows(
            user_choice=user_choice,
            model_choice=model_choice,
            elapsed_seconds=elapsed_seconds,
            trial_ended_epoch=trial_ended_epoch,
        )
        print(f"Saved/finalized {updated_count} row(s) in {self.recorder.current_session_path}", flush=True)

        self._maybe_start_finetune()

        self.pending_model_choice = model_choice
        self.pending_model_available = model_choice is not None

        self.user_map.animate_to_selection(user_choice, marker_text="U")
        self.model_reveal_job = self.root.after(MODEL_DELAY_MS, self._reveal_model_choice)

    def _quit_and_aggregate(self) -> None:
        self._cancel_pending_jobs()

        if self.finetune_process is not None and self.finetune_process.poll() is None:
            try:
                self.finetune_process.terminate()
            except Exception:
                pass

        self.recorder.aggregate_to_full_dataset()
        try:
            self.eeg_stream.stop()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    try:
        app = PictureQuadrantGame(root)
    except Exception:
        root.destroy()
        raise
    root.deiconify()
    root.mainloop()