import os
import time
import threading
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "face_landmarker.task"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sleepiness Detector UI")
        self.root.geometry("1180x720")
        self.root.minsize(1050, 650)

        self.cap = None
        self.landmarker = None
        self.running = False
        self.current_frame_bgr = None

        self.closed_start_time = None
        self.was_closed = False
        self.alert_active = False
        self.blink_count = 0
        self.alert_count = 0
        self.session_start = None

        self.threshold_var = tk.DoubleVar(value=0.21)
        self.sleepy_seconds_var = tk.DoubleVar(value=1.2)
        self.sound_enabled_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Camera stopped")
        self.ear_var = tk.StringVar(value="--")
        self.closed_time_var = tk.StringVar(value="0.00 s")
        self.blink_count_var = tk.StringVar(value="0")
        self.alert_count_var = tk.StringVar(value="0")
        self.session_time_var = tk.StringVar(value="00:00")
        self.threshold_text_var = tk.StringVar(value=f"{self.threshold_var.get():.2f}")
        self.sleepy_seconds_text_var = tk.StringVar(value=f"{self.sleepy_seconds_var.get():.1f} s")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        video_panel = ttk.Frame(self.root, padding=12)
        video_panel.grid(row=0, column=0, sticky="nsew")
        video_panel.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            video_panel,
            text="Press Start Camera",
            bg="#111111",
            fg="white",
            font=("Segoe UI", 22, "bold"),
            anchor="center",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side_panel = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        side_panel.grid(row=0, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)

        title = ttk.Label(side_panel, text="Sleepiness Monitor", font=("Segoe UI", 20, "bold"))
        title.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.status_card = tk.Label(
            side_panel,
            textvariable=self.status_var,
            bg="#555555",
            fg="white",
            font=("Segoe UI", 16, "bold"),
            padx=12,
            pady=16,
            wraplength=300,
        )
        self.status_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        metrics = ttk.LabelFrame(side_panel, text="Live Metrics", padding=10)
        metrics.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        metrics.columnconfigure(1, weight=1)

        self._add_metric(metrics, 0, "Eye openness / EAR", self.ear_var)
        self._add_metric(metrics, 1, "Eyes closed for", self.closed_time_var)
        self._add_metric(metrics, 2, "Blinks", self.blink_count_var)
        self._add_metric(metrics, 3, "Sleepy alerts", self.alert_count_var)
        self._add_metric(metrics, 4, "Session time", self.session_time_var)

        ttk.Label(metrics, text="Drowsiness progress").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        self.progress = ttk.Progressbar(metrics, orient="horizontal", mode="determinate", maximum=100)
        self.progress.grid(row=6, column=0, columnspan=2, sticky="ew")

        controls = ttk.LabelFrame(side_panel, text="Controls", padding=10)
        controls.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        ttk.Button(controls, text="Start Camera", command=self.start_detection).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(controls, text="Stop Camera", command=self.stop_detection).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )
        ttk.Button(controls, text="Reset Stats", command=self.reset_stats).grid(
            row=1, column=0, sticky="ew", padx=(0, 5), pady=(8, 0)
        )
        ttk.Button(controls, text="Save Snapshot", command=self.save_snapshot).grid(
            row=1, column=1, sticky="ew", padx=(5, 0), pady=(8, 0)
        )

        settings = ttk.LabelFrame(side_panel, text="Sensitivity Settings", padding=10)
        settings.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        settings.columnconfigure(0, weight=1)

        row = ttk.Frame(settings)
        row.grid(row=0, column=0, sticky="ew")
        row.columnconfigure(0, weight=1)
        ttk.Label(row, text="Eye closed threshold").grid(row=0, column=0, sticky="w")
        ttk.Label(row, textvariable=self.threshold_text_var).grid(row=0, column=1, sticky="e")

        ttk.Scale(
            settings,
            from_=0.15,
            to=0.30,
            variable=self.threshold_var,
            command=lambda _: self.update_setting_labels(),
        ).grid(row=1, column=0, sticky="ew", pady=(2, 10))

        row2 = ttk.Frame(settings)
        row2.grid(row=2, column=0, sticky="ew")
        row2.columnconfigure(0, weight=1)
        ttk.Label(row2, text="Seconds before sleepy alert").grid(row=0, column=0, sticky="w")
        ttk.Label(row2, textvariable=self.sleepy_seconds_text_var).grid(row=0, column=1, sticky="e")

        ttk.Scale(
            settings,
            from_=0.5,
            to=3.0,
            variable=self.sleepy_seconds_var,
            command=lambda _: self.update_setting_labels(),
        ).grid(row=3, column=0, sticky="ew", pady=(2, 10))

        ttk.Checkbutton(settings, text="Play alert sound", variable=self.sound_enabled_var).grid(
            row=4, column=0, sticky="w"
        )

        log_frame = ttk.LabelFrame(side_panel, text="Event Log", padding=10)
        log_frame.grid(row=5, column=0, sticky="nsew")
        side_panel.rowconfigure(5, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_box = tk.Text(log_frame, height=7, wrap="word", state="disabled")
        self.log_box.grid(row=0, column=0, sticky="nsew")

    def _add_metric(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=variable, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=1, sticky="e", pady=2
        )

    def update_setting_labels(self):
        self.threshold_text_var.set(f"{self.threshold_var.get():.2f}")
        self.sleepy_seconds_text_var.set(f"{self.sleepy_seconds_var.get():.1f} s")

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{timestamp}] {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def create_landmarker(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Missing model file",
                "face_landmarker.task is missing. Put it in the same folder as this Python file.",
            )
            return None

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        return vision.FaceLandmarker.create_from_options(options)

    def start_detection(self):
        if self.running:
            return

        self.landmarker = self.create_landmarker()

        if self.landmarker is None:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open your webcam.")
            self.landmarker = None
            return

        self.running = True
        self.session_start = time.time()
        self.closed_start_time = None
        self.was_closed = False
        self.alert_active = False

        self.log_message("Camera started")
        self.update_frame()

    def stop_detection(self):
        self.running = False
        self.closed_start_time = None
        self.was_closed = False
        self.alert_active = False

        self.progress["value"] = 0
        self.closed_time_var.set("0.00 s")
        self.status_var.set("Camera stopped")
        self.update_status_card("Camera stopped")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass

            self.landmarker = None

        self.video_label.configure(image="", text="Press Start Camera")
        self.video_label.image = None

        self.log_message("Camera stopped")

    def reset_stats(self):
        self.blink_count = 0
        self.alert_count = 0

        self.blink_count_var.set("0")
        self.alert_count_var.set("0")

        self.closed_start_time = None
        self.was_closed = False
        self.alert_active = False
        self.progress["value"] = 0

        self.log_message("Stats reset")

    def save_snapshot(self):
        if self.current_frame_bgr is None:
            messagebox.showinfo("No frame", "Start the camera before saving a snapshot.")
            return

        filename = datetime.now().strftime("sleep_detector_snapshot_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(filename, self.current_frame_bgr)

        self.log_message(f"Snapshot saved: {filename}")
        messagebox.showinfo("Snapshot saved", f"Saved {filename}")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.status_var.set("Could not read webcam frame")
            self.update_status_card("Error")
            self.root.after(200, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        status = "No Face Detected"
        average_ear = None
        closed_duration = 0.0
        progress_value = 0.0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_eye_points = self.get_eye_points(landmarks, LEFT_EYE, width, height)
            right_eye_points = self.get_eye_points(landmarks, RIGHT_EYE, width, height)

            left_ear = self.eye_aspect_ratio(left_eye_points)
            right_ear = self.eye_aspect_ratio(right_eye_points)

            average_ear = (left_ear + right_ear) / 2.0

            for point in left_eye_points + right_eye_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            threshold = self.threshold_var.get()
            sleepy_seconds = self.sleepy_seconds_var.get()

            is_closed = average_ear < threshold
            now = time.time()

            if is_closed:
                if self.closed_start_time is None:
                    self.closed_start_time = now

                closed_duration = now - self.closed_start_time
                progress_value = min(100, (closed_duration / sleepy_seconds) * 100)

                if closed_duration >= sleepy_seconds:
                    status = "YOU'RE SLEEPY"

                    if not self.alert_active:
                        self.alert_count += 1
                        self.alert_count_var.set(str(self.alert_count))
                        self.alert_active = True
                        self.log_message("Sleepy alert triggered")
                        self.play_alert_sound()
                else:
                    status = "Eyes Closing..."

            else:
                if self.was_closed and self.closed_start_time is not None:
                    blink_duration = now - self.closed_start_time

                    if 0.08 <= blink_duration < sleepy_seconds:
                        self.blink_count += 1
                        self.blink_count_var.set(str(self.blink_count))

                self.closed_start_time = None
                self.alert_active = False
                status = "Act Normal"
                closed_duration = 0.0
                progress_value = 0.0

            self.was_closed = is_closed

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                status,
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.status_bgr(status),
                3,
            )

        else:
            self.closed_start_time = None
            self.was_closed = False
            self.alert_active = False
            self.ear_var.set("--")

        self.current_frame_bgr = frame.copy()

        self.update_ui_values(status, average_ear, closed_duration, progress_value)
        self.show_frame_in_ui(frame)

        self.root.after(15, self.update_frame)

    def update_ui_values(self, status, average_ear, closed_duration, progress_value):
        self.status_var.set(status)
        self.update_status_card(status)

        if average_ear is not None:
            self.ear_var.set(f"{average_ear:.3f}")

        self.closed_time_var.set(f"{closed_duration:.2f} s")
        self.progress["value"] = progress_value

        if self.session_start is not None and self.running:
            elapsed = int(time.time() - self.session_start)
            minutes, seconds = divmod(elapsed, 60)
            self.session_time_var.set(f"{minutes:02d}:{seconds:02d}")

    def update_status_card(self, status):
        if status == "YOU'RE SLEEPY":
            self.status_card.configure(bg="#c62828", fg="white")
        elif status == "Act Normal":
            self.status_card.configure(bg="#2e7d32", fg="white")
        elif status == "Camera stopped":
            self.status_card.configure(bg="#555555", fg="white")
        else:
            self.status_card.configure(bg="#f9a825", fg="black")

    def show_frame_in_ui(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        max_width = 820
        max_height = 620

        image.thumbnail((max_width, max_height))

        try:
            resample_mode = Image.Resampling.LANCZOS
        except AttributeError:
            resample_mode = Image.LANCZOS

        image = image.resize(image.size, resample_mode)
        photo = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def play_alert_sound(self):
        if not self.sound_enabled_var.get():
            return

        def beep():
            try:
                import winsound

                winsound.Beep(900, 250)
                winsound.Beep(1200, 250)
            except Exception:
                self.root.bell()

        threading.Thread(target=beep, daemon=True).start()

    def status_bgr(self, status):
        if status == "YOU'RE SLEEPY":
            return (0, 0, 255)

        if status == "Act Normal":
            return (0, 255, 0)

        return (0, 255, 255)

    def get_eye_points(self, landmarks, eye_indices, width, height):
        points = []

        for index in eye_indices:
            landmark = landmarks[index]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        return points

    def eye_aspect_ratio(self, points):
        p1, p2, p3, p4, p5, p6 = points

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)
        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return 0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def on_close(self):
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SleepDetectorApp(root)
    root.mainloop()