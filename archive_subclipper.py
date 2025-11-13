#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont
from pathlib import Path

# ----- Drag & Drop (optionnel) -----
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False
    TkinterDnD = None
    DND_FILES = None

# ----- Chemin du logo -----
# ----- Chemin du logo (compatible script + app bundlée) -----
def resource_path(relative_path: str) -> str:
    """
    Retourne le chemin absolu vers une ressource, que l'on soit :
    - en script normal
    - ou empaqueté avec PyInstaller (sys._MEIPASS)
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent
    return str(base_path / relative_path)

LOGO_PATH = resource_path("archives_subclipper_logo.png")


# Dépendances externes
try:
    import cv2
except Exception as e:
    raise SystemExit("OpenCV (opencv-python) est requis : pip install opencv-python") from e

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit("Pillow est requis : pip install Pillow") from e

# pyscenedetect est facultatif
try:
    import scenedetect  # noqa
    from scenedetect.detectors import ContentDetector  # noqa
    HAS_SCENEDETECT = True
except Exception:
    HAS_SCENEDETECT = False


# ---------- Helpers Timecode ----------

def frames_to_tc(frames: int, fps: int) -> str:
    if frames < 0:
        frames = 0
    f = frames % fps
    total_seconds = frames // fps
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def tc_to_frames(tc: str, fps: int) -> int:
    h, m, s, f = map(int, tc.split(":"))
    return ((h * 3600 + m * 60 + s) * fps) + f


# ---------- ALE writer ----------

def write_ale_rows(rows, ale_path: str, fps=25, tracks_default="V"):
    """
    rows: list de dict avec clés:
      name, tape, start_tc, end_tc, source_file(optional), tracks(optional)
    """
    with open(ale_path, "w", encoding="utf-8") as f:
        f.write("Heading\n")
        f.write("FIELD_DELIM\tTABS\n")
        f.write("VIDEO_FORMAT\t1080\n")
        f.write("AUDIO_FORMAT\t48khz\n")
        f.write(f"FPS\t{fps}\n\n")
        columns = ["Name", "Tape", "Start", "End", "Tracks"]
        if any(r.get("source_file") for r in rows):
            columns.append("Source File")
        f.write("Column\n")
        f.write("\t".join(columns) + "\n\n")
        f.write("Data\n")
        for r in rows:
            name = r["name"]
            tape = r.get("tape", "")
            start_tc = r["start_tc"]
            end_tc = r["end_tc"]
            tracks = r.get("tracks") or tracks_default
            line = [name, tape, start_tc, end_tc, tracks]
            if "source_file" in r and r["source_file"]:
                line.append(r["source_file"])
            f.write("\t".join(line) + "\n")


# ---------- Détection de coupes (fallback OpenCV) ----------

def detect_scenes_opencv(
    video_path: str,
    fps: int,
    stride: int,
    sensitivity_pct: int,
    progress_callback=None,
    cancel_callback=None,
    start_frame=None,
    end_frame=None,
    full_total_frames=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo pour détection.")

    if full_total_frames is None:
        full_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    sf = start_frame if start_frame is not None else 0
    ef = end_frame if end_frame is not None else full_total_frames
    sf = max(0, min(sf, full_total_frames))
    ef = max(sf + 1, min(ef, full_total_frames))

    range_len = max(1, ef - sf)

    threshold = 0.6 - 0.5 * (sensitivity_pct / 100.0)
    threshold = max(0.1, min(0.6, threshold))

    scene_cuts = [sf]
    prev_hist = None
    frame_idx = sf

    def frame_to_hist(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    if progress_callback:
        progress_callback(0.0)

    while True:
        if cancel_callback and cancel_callback():
            cap.release()
            raise RuntimeError("__SCAN_CANCELLED__")

        if frame_idx >= ef:
            break

        ok = cap.grab()
        if not ok:
            break
        if frame_idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            hist = frame_to_hist(frame)
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                diff = 1.0 - float(corr)
                if diff >= threshold:
                    scene_cuts.append(frame_idx)
            prev_hist = hist

            if progress_callback:
                pct = 100.0 * (frame_idx - sf) / float(range_len)
                progress_callback(pct)

        frame_idx += 1

    cap.release()

    if progress_callback:
        progress_callback(100.0)

    cleaned = []
    last = -999999
    min_gap = max(1, int(fps * 0.5))
    for c in sorted(set(scene_cuts)):
        if c - last >= min_gap:
            cleaned.append(c)
            last = c

    if not cleaned or cleaned[0] != sf:
        cleaned = [sf] + cleaned
    return cleaned, full_total_frames


# ---------- Détection principale (PySceneDetect + modes + plage) ----------

def detect_scenes(
    video_path: str,
    sensitivity_pct: int,
    fps_hint: int,
    progress_callback=None,
    scan_mode: str = "fast",
    cancel_callback=None,
    start_frame=None,
    end_frame=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo.")

    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or fps_hint or 25
    total_frames_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    sf = start_frame if start_frame is not None else 0
    ef = end_frame if end_frame is not None else total_frames_clip
    sf = max(0, min(sf, total_frames_clip))
    ef = max(sf + 1, min(ef, total_frames_clip))

    range_len = max(1, ef - sf)

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    # PySceneDetect dispo
    if HAS_SCENEDETECT:
        from scenedetect.detectors import ContentDetector

        thr = 50 - 35 * (sensitivity_pct / 100.0)
        thr = max(5.0, min(50.0, float(thr)))

        detector = ContentDetector(threshold=thr)
        scene_cuts = []

        frame_idx = sf
        if progress_callback:
            progress_callback(0.0)

        if scan_mode == "precise":
            stride = 1
        elif scan_mode == "turbo":
            stride = 3
        elif scan_mode == "megaturbo":
            stride = 6
        else:  # fast
            stride = 2

        while True:
            if cancel_callback and cancel_callback():
                cap.release()
                raise RuntimeError("__SCAN_CANCELLED__")

            if frame_idx >= ef:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if stride > 1 and (frame_idx % stride) != 0:
                frame_idx += 1
                if progress_callback and scan_mode != "megaturbo":
                    if (frame_idx - sf) % 10 == 0:
                        pct = 100.0 * (frame_idx - sf) / float(range_len)
                        progress_callback(pct)
                continue

            cut_list = detector.process_frame(frame_idx, frame)
            if cut_list:
                scene_cuts.extend(cut_list)

            frame_idx += 1

            if progress_callback and scan_mode != "megaturbo":
                if (frame_idx - sf) % 10 == 0:
                    pct = 100.0 * (frame_idx - sf) / float(range_len)
                    progress_callback(pct)

        cap.release()

        if progress_callback and scan_mode != "megaturbo":
            progress_callback(100.0)

        if not scene_cuts:
            starts = [sf]
        else:
            starts = sorted(set(scene_cuts))
            if starts[0] != sf:
                starts = [sf] + starts

        return starts, total_frames_clip, fps

    # Fallback OpenCV
    cap.release()

    base_stride = max(1, fps // 2)
    if scan_mode == "precise":
        stride = base_stride
    elif scan_mode == "turbo":
        stride = max(1, base_stride * 4)
    elif scan_mode == "megaturbo":
        stride = max(1, base_stride * 8)
    else:
        stride = max(1, base_stride * 2)

    if scan_mode == "megaturbo":
        progress_callback = None

    starts, full_total = detect_scenes_opencv(
        video_path,
        fps,
        stride=stride,
        sensitivity_pct=sensitivity_pct,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        start_frame=sf,
        end_frame=ef,
        full_total_frames=total_frames_clip,
    )
    return starts, full_total, fps


# ---------- VideoPlayer ----------

class VideoPlayer(threading.Thread):
    """
    Boucle de lecture sur un Label Tkinter.
    J/K/L : vitesse -pause + (multi-pression = *2, *4, ...)
    """
    def __init__(self, app, label_widget):
        super().__init__(daemon=True)
        self.app = app
        self.label = label_widget
        self.cap = None
        self.path = None
        self.frame = None
        self.photo = None
        self.running = False
        self.playback_speed = 0.0
        self.fps = 25
        self.total_frames = 0
        self.cur_frame_idx = 0

    def load(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la vidéo.")
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS))) or 25
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.cur_frame_idx = 0
        self.seek(0)
        self.playback_speed = 0.0

    def seek(self, frame_index: int):
        if not self.cap:
            return
        frame_index = max(0, min(self.total_frames - 1, frame_index))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.cur_frame_idx = frame_index
            self.show_frame(frame)

    def show_frame(self, frame):
        self.frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        maxw, maxh = 960, 480
        scale = min(maxw / w, maxh / h, 1.0)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(img)
        # on enlève le texte "Drop here"
        self.label.configure(image=self.photo, text="")

        offset = getattr(self.app, "tc_offset_frames", 0)
        tc = frames_to_tc(self.cur_frame_idx + offset, self.fps)
        self.app.tc_var.set(tc)

    def run(self):
        self.running = True
        while self.running:
            try:
                if not self.cap or self.playback_speed == 0.0:
                    time.sleep(0.01)
                    continue
                step = int(self.playback_speed) if abs(self.playback_speed) >= 1 else (1 if self.playback_speed > 0 else -1)
                next_idx = self.cur_frame_idx + step
                if next_idx < 0 or next_idx >= self.total_frames:
                    self.playback_speed = 0.0
                    continue
                self.seek(next_idx)
                delay = 1.0 / (self.fps * max(1, abs(step)))
                time.sleep(delay)
            except Exception:
                time.sleep(0.02)

    def stop(self):
        self.running = False
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def key_L(self):
        if self.playback_speed <= 0:
            self.playback_speed = 1
        else:
            self.playback_speed = min(32, self.playback_speed * 2)

    def key_J(self):
        if self.playback_speed >= 0:
            self.playback_speed = -1
        else:
            self.playback_speed = max(-32, self.playback_speed * 2)

    def key_K(self):
        self.playback_speed = 0.0


# ---------- App Tk ----------

class App(TkinterDnD.Tk if HAS_DND else tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MXF → EDL → ALE (subclips Avid)")
        self.geometry("1230x1020")
        self.resizable(width=False, height=False)

        self.video_path = tk.StringVar()
        self.name_prefix = tk.StringVar(value="subclip_")
        self.start_index = tk.IntVar(value=1)
        self.padding = tk.IntVar(value=3)
        self.fps_override = tk.StringVar(value="")
        self.source_file = tk.StringVar(value="")
        self.track_v = tk.BooleanVar(value=True)
        self.track_a1 = tk.BooleanVar(value=False)
        self.track_a2 = tk.BooleanVar(value=False)
        self.entire_clip = tk.BooleanVar(value=True)
        self.sensitivity = tk.IntVar(value=100)
        self.start_tc_str = tk.StringVar(value="00:00:00:00")
        self.tc_var = tk.StringVar(value="00:00:00:00")
        self.scan_mode = tk.StringVar(value="fast")
        self.duration_var = tk.StringVar(value="--:--:--:--")

        self.in_frame = None
        self.out_frame = None
        self.cuts = []
        self.cuts_raw = []

        self.tc_offset_frames = 0

        self._scan_cancelled = False
        self._scan_thread = None
        self._scan_ok = None
        self._scan_err = None
        self._scan_result = None

        self._logo_img = None
        self._thumb_images = []

        self._build_ui()

        # cacher TC + vignettes tant qu'aucune vidéo
        self.tc_big_label.pack_forget()
        self.thumbs_frame.pack_forget()

        # scroll global
        self.bind_all("<MouseWheel>", self._global_scroll)
        self.bind_all("<Button-4>", self._global_scroll)
        self.bind_all("<Button-5>", self._global_scroll)

        self.player = VideoPlayer(self, self.preview_label)
        self.bind_all("<Key-j>", lambda e: self.player.key_J())
        self.bind_all("<Key-k>", lambda e: self.player.key_K())
        self.bind_all("<Key-l>", lambda e: self.player.key_L())
        self.bind_all("<Key-i>", lambda e: self.mark_in())
        self.bind_all("<Key-o>", lambda e: self.mark_out())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.player.start()

    # --- scroll global : toujours vers les vignettes ---
    def _global_scroll(self, event):
        if not hasattr(self, "thumbs_canvas"):
            return
        canvas = self.thumbs_canvas
        if sys.platform == "darwin":
            canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=6)

        ttk.Button(top, text="Ouvrir vidéo (MXF…)", command=self.open_video).pack(side="left")
        ttk.Entry(top, textvariable=self.video_path, width=60).pack(side="left", padx=6)
        ttk.Label(top, text="Durée:").pack(side="left")
        ttk.Label(top, textvariable=self.duration_var, width=12).pack(side="left", padx=6)
        ttk.Button(top, text="Nouveau (Reset)", command=self.reset_app).pack(side="left", padx=6)
        ttk.Button(top, text="Vider vignettes", command=self._clear_thumbs).pack(side="left")

        right = ttk.Frame(self)
        right.pack(side="right", fill="y", padx=8, pady=6)

        cfg = ttk.LabelFrame(right, text="Paramètres")
        cfg.pack(fill="x", pady=4)

        row = 0

        def add_row(lbl, widget):
            nonlocal row
            ttk.Label(cfg, text=lbl).grid(row=row, column=0, sticky="e", padx=4, pady=2)
            widget.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            row += 1

        add_row("Préfixe", ttk.Entry(cfg, textvariable=self.name_prefix, width=24))
        add_row("Index départ", ttk.Spinbox(cfg, from_=1, to=999999, textvariable=self.start_index, width=8))
        add_row("Padding", ttk.Spinbox(cfg, from_=1, to=6, textvariable=self.padding, width=6))

        tracks_row = ttk.Frame(cfg)

        def add_tracks_row():
            nonlocal row
            ttk.Label(cfg, text="Tracks").grid(row=row, column=0, sticky="e", padx=4, pady=2)
            tracks_row.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            row += 1

        add_tracks_row()
        ttk.Checkbutton(tracks_row, text="V", variable=self.track_v).pack(side="left", padx=2)
        ttk.Checkbutton(tracks_row, text="A1", variable=self.track_a1).pack(side="left", padx=2)
        ttk.Checkbutton(tracks_row, text="A2", variable=self.track_a2).pack(side="left", padx=2)

        add_row("Source File", ttk.Entry(cfg, textvariable=self.source_file, width=24))
        add_row("FPS (auto si vide)", ttk.Entry(cfg, textvariable=self.fps_override, width=8))

        start_tc_frame = ttk.Frame(cfg)
        start_tc_entry = ttk.Entry(start_tc_frame, textvariable=self.start_tc_str, width=12)
        start_tc_entry.pack(side="left")
        ttk.Button(start_tc_frame, text="Set", command=self.apply_start_tc).pack(side="left", padx=4)
        start_tc_entry.bind("<Return>", lambda e: self.apply_start_tc())
        add_row("Start TC", start_tc_frame)

        sens_frame = ttk.LabelFrame(right, text="Détection de coupes")
        sens_frame.pack(fill="x", pady=8)
        ttk.Scale(sens_frame, from_=0, to=100, orient="horizontal", variable=self.sensitivity).pack(
            fill="x", padx=6, pady=4
        )
        ttk.Label(sens_frame, text="← moins sensible | plus sensible →").pack()

        ttk.Label(sens_frame, text="Mode de scan :").pack(anchor="w", padx=6, pady=(6, 2))
        modes_row = ttk.Frame(sens_frame)
        modes_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Radiobutton(modes_row, text="Précis", value="precise", variable=self.scan_mode).pack(
            side="left", padx=(0, 4)
        )
        ttk.Radiobutton(modes_row, text="Rapide", value="fast", variable=self.scan_mode).pack(
            side="left", padx=4
        )
        ttk.Radiobutton(modes_row, text="Turbo", value="turbo", variable=self.scan_mode).pack(
            side="left", padx=4
        )
        ttk.Radiobutton(modes_row, text="MegaTurbo", value="megaturbo", variable=self.scan_mode).pack(
            side="left", padx=4
        )

        io_frame = ttk.LabelFrame(right, text="Plage")
        io_frame.pack(fill="x", pady=8)
        ttk.Checkbutton(
            io_frame,
            text="Entire Clip",
            variable=self.entire_clip,
            command=self.on_entire_clip_toggle
        ).pack(anchor="w", padx=6)
        b_row = ttk.Frame(io_frame)
        b_row.pack(fill="x", padx=6, pady=4)
        ttk.Button(b_row, text="Mark In (I)", command=self.mark_in).pack(side="left", padx=2)
        ttk.Button(b_row, text="Mark Out (O)", command=self.mark_out).pack(side="left", padx=2)
        self.in_label = ttk.Label(io_frame, text="IN: --")
        self.out_label = ttk.Label(io_frame, text="OUT: --")
        self.in_label.pack(anchor="w", padx=6)
        self.out_label.pack(anchor="w", padx=6)

        act = ttk.Frame(right)
        act.pack(fill="x", pady=8)
        self.scan_button = ttk.Button(act, text="Scanner les coupes", command=self.scan_cuts)
        self.scan_button.pack(fill="x", pady=2)
        ttk.Button(act, text="Générer ALE", command=self.generate_ale).pack(fill="x", pady=2)

        helpf = ttk.LabelFrame(right, text="Raccourcis")
        helpf.pack(fill="x", pady=8)
        ttk.Label(
            helpf,
            text="J = recule (multi-pression = plus vite)\nK = pause\nL = avance (multi-pression)\nI/O = Mark In/Out",
        ).pack(anchor="w", padx=6, pady=4)

        # Logo en bas, centré
        branding = ttk.Frame(right)
        branding.pack(side="bottom", fill="x", pady=(0, 4))
        branding.columnconfigure(0, weight=1)

        try:
            img = Image.open(LOGO_PATH)
            img = img.convert("RGBA")
            max_w, max_h = 500, 200
            w, h = img.size
            scale = min(max_w / w, max_h / h, 1.0)
            new_size = (int(w * scale), int(h * scale))
            if new_size != (w, h):
                img = img.resize(new_size, Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(img)
            lbl_logo = ttk.Label(branding, image=self._logo_img)
            lbl_logo.grid(row=0, column=0, sticky="n", padx=4, pady=(0, 2))
        except Exception:
            lbl_logo = ttk.Label(
                branding,
                text="Archives Subclipper\nby skeupon",
                justify="center"
            )
            lbl_logo.grid(row=0, column=0, sticky="n", padx=4, pady=(0, 2))

        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=6)

        self.preview_label = ttk.Label(left)
        self.preview_label.pack(anchor="n", fill="both", expand=True)

        # Texte "Drop here" par défaut
        self.drop_font = tkfont.Font(size=16, weight="bold")
        self.preview_label.configure(
            text="Drop video here\n(MXF, MOV, MP4…)",
            font=self.drop_font,
            anchor="center",
            justify="center"
        )

        # Activation du drag & drop si dispo
        if HAS_DND:
            self.preview_label.drop_target_register(DND_FILES)
            self.preview_label.dnd_bind("<<Drop>>", self.on_drop_files)

        big_tc_font = tkfont.Font(size=18, weight="bold")
        self.tc_big_label = ttk.Label(left, textvariable=self.tc_var, font=big_tc_font)
        self.tc_big_label.pack(pady=(8, 4))

        self.thumbs_frame = ttk.LabelFrame(left, text="0 coupes détectées")
        self.thumbs_frame.pack(fill="both", expand=True, pady=8)
        self.thumbs_canvas = tk.Canvas(self.thumbs_frame)
        self.thumbs_canvas.pack(side="left", fill="both", expand=True)
        self.thumbs_scroll = ttk.Scrollbar(
            self.thumbs_frame, orient="vertical", command=self.thumbs_canvas.yview
        )
        self.thumbs_scroll.pack(side="right", fill="y")
        self.thumbs_canvas.configure(yscrollcommand=self.thumbs_scroll.set)
        self.thumbs_inner = ttk.Frame(self.thumbs_canvas)
        self.thumbs_canvas.create_window((0, 0), window=self.thumbs_inner, anchor="nw")
        self.thumbs_inner.bind(
            "<Configure>",
            lambda e: self.thumbs_canvas.configure(scrollregion=self.thumbs_canvas.bbox("all")),
        )

    # --- Drag & Drop handler ---

    def on_drop_files(self, event):
        # Debug facultatif : voir la chaîne brute dans la console
        print("DROP RAW:", repr(event.data))

        data = (event.data or "").strip()
        if not data:
            return

        paths = []
        i = 0
        n = len(data)

        while i < n:
            if data[i] == "{":
                # chemin entre accolades { ... }
                j = data.find("}", i + 1)
                if j == -1:
                    # accolade fermante manquante -> on prend le reste
                    paths.append(data[i + 1:].strip())
                    break
                paths.append(data[i + 1: j])
                i = j + 1
            else:
                # chemin sans accolades, jusqu'au prochain espace
                j = data.find(" ", i)
                if j == -1:
                    paths.append(data[i:])
                    break
                paths.append(data[i:j])
                i = j + 1

            # skip espaces entre chemins
            while i < n and data[i].isspace():
                i += 1

        if not paths:
            return

        # on ne prend que le premier fichier dropé
        path = paths[0].strip()

        # certains systèmes peuvent donner des URLs file://
        if path.startswith("file://"):
            path = path[7:]

        # enlever d'éventuelles guillemets
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]

        from pathlib import Path as _Path
        p = _Path(path)

        if not p.is_file():
            messagebox.showerror("Erreur", f"Fichier non trouvé :\n{path}")
            return

        self.open_video_path(str(p))

    # ------------ Actions ------------

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier vidéo (MXF, MOV, MP4…)",
            filetypes=[
                ("Vidéo", "*.mxf *.mov *.mp4 *.mkv *.avi *.mxf"),
                ("Tous", "*.*"),
            ],
        )
        if not path:
            return
        self.open_video_path(path)

    def open_video_path(self, path: str):
        self.video_path.set(path)
        try:
            self.player.load(path)
            fps = self.player.fps or 25
            total = self.player.total_frames or 0
            self.duration_var.set(frames_to_tc(total, fps))

            if not self.source_file.get():
                self.source_file.set(Path(path).name)

            self.in_frame = None
            self.out_frame = None
            self.in_label.config(text="IN: --")
            self.out_label.config(text="OUT: --")
            self.cuts = []
            self.cuts_raw = []
            self._clear_thumbs()
            self.scan_button.config(text="Scanner les coupes")
            self.apply_start_tc(update_thumbs=False)

            # on cache le texte de drop, la vidéo va s'afficher
            self.tc_big_label.pack(pady=(8, 4))
            self.thumbs_frame.pack(fill="both", expand=True, pady=8)
        except Exception as e:
            # en cas de plantage, on remet le texte de drop
            self.preview_label.configure(
                image="",
                text="Drop video here\n(MXF, MOV, MP4…)",
                font=self.drop_font,
                anchor="center",
                justify="center"
            )
            messagebox.showerror("Erreur", str(e))

    def reset_app(self):
        try:
            if hasattr(self, "player") and self.player.cap:
                self.player.key_K()
                self.player.seek(0)
        except Exception:
            pass

        self._scan_cancelled = True
        self._scan_ok = None
        self._scan_err = None
        self._scan_result = None
        self._hide_busy()
        self.scan_button.config(text="Scanner les coupes")

        self.video_path.set("")
        self.name_prefix.set("subclip_")
        self.start_index.set(1)
        self.padding.set(3)
        self.fps_override.set("")
        self.source_file.set("")
        self.track_v.set(True)
        self.track_a1.set(False)
        self.track_a2.set(False)
        self.entire_clip.set(True)
        self.sensitivity.set(100)
        self.scan_mode.set("fast")
        self.tc_offset_frames = 0
        self.tc_var.set("00:00:00:00")
        self.duration_var.set("--:--:--:--")
        self.in_frame = None
        self.out_frame = None
        self.in_label.config(text="IN: --")
        self.out_label.config(text="OUT: --")
        self.cuts = []
        self.cuts_raw = []
        self._clear_thumbs()
        self._set_thumbs_count_label(0)

        # on remet le texte de drop
        self.preview_label.configure(
            image="",
            text="Drop video here\n(MXF, MOV, MP4…)",
            font=self.drop_font,
            anchor="center",
            justify="center"
        )

        self.tc_big_label.pack_forget()
        self.thumbs_frame.pack_forget()

    def _set_thumbs_count_label(self, n):
        if n == 1:
            label = "1 coupe détectée"
        else:
            label = f"{n} coupes détectées"
        self.thumbs_frame.config(text=label)

    def _clear_thumbs(self):
        for child in list(self.thumbs_inner.children.values()):
            child.destroy()
        self._thumb_images.clear()
        self.thumbs_canvas.configure(scrollregion=(0, 0, 0, 0))

    # --- Entire Clip toggle ---

    def on_entire_clip_toggle(self):
        if self.entire_clip.get():
            self.in_frame = None
            self.out_frame = None
            self.in_label.config(text="IN: --")
            self.out_label.config(text="OUT: --")

    def mark_in(self):
        if not self.player or not self.player.cap:
            return
        self.entire_clip.set(False)

        self.in_frame = self.player.cur_frame_idx
        fps = self.player.fps or 25
        tc = frames_to_tc(self.in_frame + self.tc_offset_frames, fps)
        self.in_label.config(text=f"IN: {tc}")
        if self.out_frame is not None and self.out_frame < self.in_frame:
            self.out_frame = None
            self.out_label.config(text="OUT: --")

    def mark_out(self):
        if not self.player or not self.player.cap:
            return
        self.entire_clip.set(False)

        self.out_frame = self.player.cur_frame_idx
        fps = self.player.fps or 25
        tc = frames_to_tc(self.out_frame + self.tc_offset_frames, fps)
        self.out_label.config(text=f"OUT: {tc}")
        if self.in_frame is not None and self.out_frame < self.in_frame:
            self.in_frame, self.out_frame = self.out_frame, self.in_frame
            tc_in = frames_to_tc(self.in_frame + self.tc_offset_frames, fps)
            tc_out = frames_to_tc(self.out_frame + self.tc_offset_frames, fps)
            self.in_label.config(text=f"IN: {tc_in}")
            self.out_label.config(text=f"OUT: {tc_out}")

    # --- Start TC / offset ---

    def apply_start_tc(self, update_thumbs=True):
        fps = None
        if self.fps_override.get().strip():
            try:
                fps = int(self.fps_override.get().strip())
            except Exception:
                fps = None
        if not fps:
            fps = getattr(self.player, "fps", 25) or 25

        tc_text = self.start_tc_str.get().strip() or "00:00:00:00"
        try:
            self.tc_offset_frames = tc_to_frames(tc_text, fps)
        except Exception:
            self.tc_offset_frames = 0

        if self.player and self.player.cap:
            cur = self.player.cur_frame_idx
            self.tc_var.set(frames_to_tc(cur + self.tc_offset_frames, fps))
        else:
            self.tc_var.set(frames_to_tc(self.tc_offset_frames, fps))

        if self.in_frame is not None:
            self.in_label.config(
                text=f"IN: {frames_to_tc(self.in_frame + self.tc_offset_frames, fps)}"
            )
        if self.out_frame is not None:
            self.out_label.config(
                text=f"OUT: {frames_to_tc(self.out_frame + self.tc_offset_frames, fps)}"
            )

        if update_thumbs and self.video_path.get().strip() and self.cuts:
            self._render_thumbs(self.video_path.get().strip(), fps, self.cuts)

    # --- Busy overlay + progression ---

    def _show_busy(self, message="Traitement...", allow_cancel=False, show_progress=True):
        try:
            if hasattr(self, "_busy_win") and self._busy_win and tk.Toplevel.winfo_exists(self._busy_win):
                return
        except Exception:
            pass
        self._busy_win = tk.Toplevel(self)
        self._busy_win.title("")
        self._busy_win.transient(self)
        self._busy_win.grab_set()
        self._busy_win.resizable(False, False)
        self._busy_win.geometry(
            "+%d+%d" % (self.winfo_rootx() + 80, self.winfo_rooty() + 80)
        )

        self._busy_msg_label = ttk.Label(self._busy_win, text=message)
        self._busy_msg_label.pack(padx=16, pady=(16, 8))

        self._busy_progress = None
        self._busy_pb = None
        self._busy_pct_label = None

        if show_progress:
            self._busy_progress = tk.IntVar(value=0)
            self._busy_pb = ttk.Progressbar(
                self._busy_win,
                mode="determinate",
                length=240,
                maximum=100,
                variable=self._busy_progress,
            )
            self._busy_pb.pack(padx=16, pady=(0, 4))
            self._busy_pct_label = ttk.Label(self._busy_win, text="0%")
            self._busy_pct_label.pack(padx=16, pady=(0, 8))
        else:
            ttk.Label(self._busy_win, text="").pack(pady=(0, 4))

        self._busy_cancel_btn = None
        if allow_cancel:
            self._busy_cancel_btn = ttk.Button(
                self._busy_win, text="Annuler", command=self._on_busy_cancel
            )
            self._busy_cancel_btn.pack(pady=(0, 12))

        self._busy_win.update_idletasks()

    def _on_busy_cancel(self):
        self._scan_cancelled = True

    def _set_busy_progress(self, pct, message=None):
        if not hasattr(self, "_busy_progress") or self._busy_progress is None:
            return

        def _update():
            if not hasattr(self, "_busy_win") or not self._busy_win:
                return
            try:
                pct_clamped = max(0, min(100, int(pct)))
            except Exception:
                pct_clamped = 0
            self._busy_progress.set(pct_clamped)
            if self._busy_pct_label is not None:
                self._busy_pct_label.config(text=f"{pct_clamped}%")
            if message is not None:
                self._busy_msg_label.config(text=message)
            try:
                self._busy_win.update_idletasks()
            except Exception:
                pass

        self.after(0, _update)

    def _hide_busy(self):
        try:
            if hasattr(self, "_busy_win") and self._busy_win:
                try:
                    self._busy_win.grab_release()
                except Exception:
                    pass
                self._busy_win.destroy()
                self._busy_win = None
        except Exception:
            pass

    # --- Scan des coupes ---

    def scan_cuts(self):
        path = self.video_path.get().strip()
        if not path:
            messagebox.showerror("Erreur", "Ouvre d'abord un fichier vidéo.")
            return

        if self._scan_thread is not None and self._scan_thread.is_alive():
            messagebox.showinfo("Scan en cours", "Un scan est déjà en cours.")
            return

        def compute_range(total_frames_clip):
            lo = 0
            hi = total_frames_clip
            if not self.entire_clip.get():
                if self.in_frame is not None and self.out_frame is not None:
                    lo, hi = sorted((self.in_frame, self.out_frame))
                elif self.in_frame is not None:
                    lo, hi = self.in_frame, total_frames_clip
                elif self.out_frame is not None:
                    lo, hi = 0, self.out_frame
            lo = max(0, lo)
            hi = max(lo + 1, min(hi, total_frames_clip))
            return lo, hi

        mode = self.scan_mode.get() or "fast"

        def worker():
            try:
                self._scan_cancelled = False
                sens = int(self.sensitivity.get())
                fps_hint = None
                if self.fps_override.get().strip():
                    try:
                        fps_hint = int(self.fps_override.get().strip())
                    except Exception:
                        fps_hint = None

                cap_tmp = cv2.VideoCapture(path)
                total_frames_clip = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                cap_tmp.release()

                lo, hi = compute_range(total_frames_clip)

                if mode == "megaturbo":
                    on_progress = None
                else:
                    def on_progress(p):
                        self._set_busy_progress(p, "Scan de la vidéo en cours…")

                def is_cancelled():
                    return self._scan_cancelled

                starts, total, fps = detect_scenes(
                    path,
                    sensitivity_pct=sens,
                    fps_hint=fps_hint or 25,
                    progress_callback=on_progress,
                    scan_mode=mode,
                    cancel_callback=is_cancelled,
                    start_frame=lo,
                    end_frame=hi,
                )
                if mode != "megaturbo" and on_progress is not None:
                    self._set_busy_progress(100, "Scan terminé")
                self._scan_ok = True
                self._scan_result = (starts, total, fps, lo, hi)
                self._scan_err = None
            except Exception as e:
                self._scan_ok = False
                if isinstance(e, RuntimeError) and str(e) == "__SCAN_CANCELLED__":
                    self._scan_err = "__SCAN_CANCELLED__"
                else:
                    self._scan_err = e

        msg = "Scan de la vidéo en cours…"
        if mode == "megaturbo":
            msg = "Scan MegaTurbo en cours…"
            self._show_busy(msg, allow_cancel=True, show_progress=False)
        else:
            self._show_busy(msg, allow_cancel=True, show_progress=True)
            self._set_busy_progress(0)

        self._scan_ok = None
        self._scan_result = None
        self._scan_err = None
        self._scan_thread = threading.Thread(target=worker, daemon=True)
        self._scan_thread.start()

        def poll():
            if self._scan_ok is None:
                self.after(100, poll)
                return
            self._hide_busy()
            if not self._scan_ok:
                if self._scan_err == "__SCAN_CANCELLED__":
                    return
                messagebox.showerror("Erreur (scan)", str(self._scan_err))
                return

            starts, total, fps, lo_used, hi_used = self._scan_result
            self.player.fps = fps
            self.player.total_frames = total

            cuts = []
            for i, s in enumerate(starts):
                e = starts[i + 1] if i + 1 < len(starts) else hi_used
                if e > s:
                    cuts.append((s, e))
            self.cuts_raw = cuts[:]
            self.cuts = cuts[:]

            self._render_thumbs(path, fps, self.cuts)
            self.scan_button.config(text="Re-scanner les coupes")
            self._set_thumbs_count_label(len(self.cuts))
            messagebox.showinfo("OK", f"{len(self.cuts)} segments détectés.")

        self.after(100, poll)

    # --- Rendu vignettes ---

    def _render_thumbs(self, path, fps, cuts):
        self._clear_thumbs()
        self._set_thumbs_count_label(len(cuts))
        if not cuts:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        offset = self.tc_offset_frames

        cols = 5
        thumb_w = 140

        for idx, (s, e) in enumerate(cuts, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = thumb_w / float(w)
            rgb = cv2.resize(rgb, (thumb_w, int(h * scale)))
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)
            self._thumb_images.append(photo)

            row = (idx - 1) // cols
            col = (idx - 1) % cols

            pane = ttk.Frame(self.thumbs_inner, relief="groove", borderwidth=1)
            pane.grid(row=row, column=col, padx=4, pady=4, sticky="nw")

            lbl = ttk.Label(pane, image=photo)
            lbl.image = photo
            lbl.pack(side="top", padx=4, pady=(4, 2))

            info = ttk.Frame(pane)
            info.pack(side="top", fill="x", expand=True, padx=4, pady=(0, 4))

            dur_frames = e - s
            dur_tc = frames_to_tc(dur_frames, fps)

            ttk.Label(info, text=f"{idx:03d}").pack(anchor="w")
            ttk.Label(info, text=f"IN: {frames_to_tc(s + offset, fps)}").pack(anchor="w")
            ttk.Label(info, text=f"OUT: {frames_to_tc(e + offset, fps)}").pack(anchor="w")
            ttk.Label(info, text=f"Durée: {dur_tc}").pack(anchor="w")

            def make_seek(frame_index):
                return lambda ev=None: self.player.seek(frame_index)

            lbl.bind("<Button-1>", make_seek(s))

        cap.release()

    def _get_tracks_str(self) -> str:
        parts = []
        if self.track_v.get():
            parts.append("V")
        if self.track_a1.get():
            parts.append("A1")
        if self.track_a2.get():
            parts.append("A2")
        return "".join(parts) if parts else "V"

    def generate_ale(self):
        if not hasattr(self, "cuts_raw") or not self.cuts_raw:
            messagebox.showerror(
                "Erreur", "Aucune coupe détectée. Clique 'Scanner les coupes' d'abord."
            )
            return
        try:
            fps = None
            if self.fps_override.get().strip():
                try:
                    fps = int(self.fps_override.get().strip())
                except Exception:
                    fps = None
            fps = fps or self.player.fps or 25

            lo = 0
            hi = self.player.total_frames or 0
            if self.in_frame is not None and self.out_frame is not None:
                lo, hi = sorted((self.in_frame, self.out_frame))

            selected = []
            for (s, e) in self.cuts_raw:
                if e <= lo or s >= hi:
                    continue
                s2 = max(s, lo)
                e2 = min(e, hi)
                if e2 > s2:
                    selected.append((s2, e2))

            if not selected:
                messagebox.showerror("Erreur", "Aucun segment ne tombe dans la plage IN/OUT.")
                return

            ale_path = filedialog.asksaveasfilename(
                title="Enregistrer ALE",
                defaultextension=".ale",
                filetypes=[("ALE", "*.ale"), ("Tous", "*.*")],
            )
            if not ale_path:
                return

            name_prefix = self.name_prefix.get() or "subclip_"
            index = int(self.start_index.get() or 1)
            padding = int(self.padding.get() or 3)
            source_file = self.source_file.get().strip() or Path(self.video_path.get()).name

            tracks = self._get_tracks_str()
            tc_offset_frames = self.tc_offset_frames

            rows = []
            for (s, e) in selected:
                start_tc = frames_to_tc(s + tc_offset_frames, fps)
                end_tc = frames_to_tc(e + tc_offset_frames, fps)
                name = f"{name_prefix}{index:0{padding}d}"
                index += 1
                rows.append(
                    {
                        "name": name,
                        "tape": "",
                        "start_tc": start_tc,
                        "end_tc": end_tc,
                        "tracks": tracks,
                        "source_file": source_file,
                    }
                )
            write_ale_rows(rows, ale_path, fps=fps, tracks_default=tracks)
            messagebox.showinfo("Terminé", f"ALE généré :\n{ale_path}")
        except Exception as e:
            messagebox.showerror("Erreur (ALE)", str(e))

    def on_close(self):
        try:
            self._scan_cancelled = True
            self.player.stop()
        except Exception:
            pass
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
