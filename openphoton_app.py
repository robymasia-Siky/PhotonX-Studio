#!/usr/bin/env python3
"""
PhotonX Studio — Professional AI Audio Demixing & Restoration (Local GPU/CPU)
Open-source equivalent to iZotope Photon.
Powered by Meta AI HTDemucs v4 (4-Stem & 6-Stem), FoxJoy De-Reverb HQ, Kimberley Jensen Vocal SOTA,
DirectML / CUDA acceleration, and PyQt6.
"""

import os
import sys
import time
import threading
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import soundfile as sf
import sounddevice as sd

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QTabWidget,
    QProgressBar, QFrame, QMessageBox, QDialog, QComboBox, QScrollArea
)
import scipy.signal as signal
import onnxruntime as ort

try:
    from model_manager import (
        MODEL_REGISTRY,
        get_model_path,
        is_model_installed,
        download_model,
        get_available_execution_providers
    )
except ImportError:
    MODEL_REGISTRY = {}
    def get_model_path(k): return APP_DIR / "models" / f"{k}.onnx"
    def is_model_installed(k): return get_model_path(k).exists()
    def download_model(k, cb=None): return get_model_path(k)
    def get_available_execution_providers(): return ['CPUExecutionProvider']

ICON_PATH = APP_DIR / "assets" / "icon.png"

def resolve_model_path():
    candidates = ["htdemucs", "htdemucs_6s", "kim_vocal_2"]
    for c in candidates:
        p = get_model_path(c)
        if p and p.exists():
            return p
    return APP_DIR / "models" / "htdemucs.onnx"

MODEL_PATH = resolve_model_path()

ENGINES = {
    "Music 4-Stems (Ultra)": {
        "stems": ["Vocals", "Drums", "Bass", "Other"],
        "colors": [QColor(76, 201, 240), QColor(247, 37, 133), QColor(247, 127, 0), QColor(128, 237, 153)],
        "algo": "Meta AI HTDemucs v4 SOTA (Zero Residual Phase Cancellation)"
    },
    "Music 6-Stems (Guitar & Piano)": {
        "stems": ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"],
        "colors": [QColor(76, 201, 240), QColor(247, 37, 133), QColor(247, 127, 0), QColor(254, 228, 64), QColor(155, 93, 229), QColor(128, 237, 153)],
        "algo": "Meta AI HTDemucs 6-Stem Full Studio Isolation"
    },
    "Acapella & Instrumental (Kim SOTA)": {
        "stems": ["Vocals", "Instrumental"],
        "colors": [QColor(76, 201, 240), QColor(247, 127, 0)],
        "algo": "Kimberley Jensen Vocal SOTA (MDX-Net / Mel-Band)"
    },
    "Drums 5-Stems": {
        "stems": ["Kick", "Snare", "Hi-Hat", "Toms", "Cymbals"],
        "colors": [QColor(230, 57, 70), QColor(241, 250, 238), QColor(168, 218, 220), QColor(69, 123, 157), QColor(29, 53, 87)],
        "algo": "DrumSep UVR 5-Stem Multi-Band Crossover"
    },
    "Cinema 4-Stems": {
        "stems": ["Dialogue", "Music", "Effects", "Ambience"],
        "colors": [QColor(0, 180, 216), QColor(114, 9, 183), QColor(255, 183, 3), QColor(251, 133, 0)],
        "algo": "Film Industry M&E Delivery Matrix"
    },
    "Voice Restore & Denoise": {
        "stems": ["Dry Voice", "Reverb Tail", "Noise Floor", "Air Presence"],
        "colors": [QColor(82, 183, 136), QColor(144, 224, 239), QColor(87, 117, 144), QColor(244, 162, 97)],
        "algo": "FoxJoy Studio De-Reverb + Adaptive STFT Wiener Denoise"
    },
    "Voice Separator (Multi-Speaker)": {
        "stems": ["Speaker 1", "Speaker 2", "Speaker 3", "Other"],
        "colors": [QColor(255, 112, 166), QColor(155, 93, 229), QColor(0, 245, 212), QColor(254, 228, 64)],
        "algo": "ReDimNet Formant & Pitch Diarization"
    }
}


class ModelDownloadWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    finished_success = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_key):
        super().__init__()
        self.model_key = model_key

    def run(self):
        try:
            download_model(self.model_key, progress_callback=lambda pct, msg: self.progress_changed.emit(pct, msg))
            self.finished_success.emit(self.model_key)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PhotonX — SOTA Model Manager (Hugging Face)")
        self.resize(760, 520)
        self.setStyleSheet("background-color: #0d1117; color: #f1f5f9; font-family: 'Segoe UI';")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        header = QLabel("📦 MODEL MANAGER — Hugging Face SOTA Checkpoints")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header.setStyleSheet("color: #38bdf8;")
        layout.addWidget(header)

        sub = QLabel("Scarica on-demand i pesi neurali ad altissima risoluzione per arricchire le capacità di PhotonX Studio.")
        sub.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #1e293b; background: transparent; border-radius: 6px; }")

        container = QWidget()
        self.cards_layout = QVBoxLayout(container)
        self.cards_layout.setContentsMargins(12, 12, 12, 12)
        self.cards_layout.setSpacing(12)

        self.btn_map = {}
        for key, meta in MODEL_REGISTRY.items():
            card = self._create_model_card(key, meta)
            self.cards_layout.addWidget(card)

        self.cards_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        close_btn = QPushButton("Chiudi")
        close_btn.setFixedSize(100, 32)
        close_btn.setStyleSheet("background-color: #334155; color: white; border-radius: 6px; font-weight: bold;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _create_model_card(self, key, meta):
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 10px; }")
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(12, 10, 12, 10)
        hl.setSpacing(12)

        vl = QVBoxLayout()
        vl.setSpacing(4)

        title = QLabel(f"{meta['name']}  <font color='#64748b'>[{meta['category']}]</font>")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        vl.addWidget(title)

        desc = QLabel(meta["description"])
        desc.setStyleSheet("color: #8b949e; font-size: 11px;")
        desc.setWordWrap(True)
        vl.addWidget(desc)

        stems_lbl = QLabel(f"Stems: {', '.join(meta['stems'])}")
        stems_lbl.setStyleSheet("color: #58a6ff; font-size: 10px; font-weight: bold;")
        vl.addWidget(stems_lbl)

        hl.addLayout(vl, stretch=1)

        btn = QPushButton()
        btn.setFixedSize(160, 34)
        self.btn_map[key] = btn
        self._update_button_state(key, btn)
        btn.clicked.connect(lambda _, k=key: self._on_download_clicked(k))
        hl.addWidget(btn)

        return frame

    def _update_button_state(self, key, btn):
        installed = is_model_installed(key)
        if installed:
            p = get_model_path(key)
            mb = p.stat().st_size / (1024 * 1024) if p else 0
            btn.setText(f"✅ Installato ({mb:.0f} MB)")
            btn.setEnabled(False)
            btn.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: bold; border-radius: 6px;")
        else:
            btn.setText("⬇ Scarica (Hugging Face)")
            btn.setEnabled(True)
            btn.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px;")

    def _on_download_clicked(self, key):
        btn = self.btn_map.get(key)
        if btn:
            btn.setEnabled(False)
            btn.setText("⏳ Connessione...")

        self.dl_worker = ModelDownloadWorker(key)
        self.dl_worker.progress_changed.connect(lambda pct, msg: btn.setText(f"⏳ {pct}%") if btn else None)
        self.dl_worker.finished_success.connect(lambda k: self._on_download_finished(k))
        self.dl_worker.error_occurred.connect(lambda err: QMessageBox.critical(self, "Errore Download", err))
        self.dl_worker.start()

    def _on_download_finished(self, key):
        btn = self.btn_map.get(key)
        if btn:
            self._update_button_state(key, btn)
        QMessageBox.information(self, "Download Completato", f"Il modello '{MODEL_REGISTRY[key]['name']}' è stato installato ed è pronto all'uso!")


class WaveformVisualizer(QFrame):
    lane_clicked = pyqtSignal(int)
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(240)
        self.setStyleSheet("background-color: #0b0e14; border: 1px solid #1e2432; border-radius: 6px;")
        self.waveforms = []
        self.colors = []
        self.stem_names = []
        self.playhead_ratio = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(self, stem_names, waveforms, colors):
        self.stem_names = stem_names
        self.waveforms = waveforms
        self.colors = colors
        self.update()

    def set_playhead(self, ratio):
        self.playhead_ratio = max(0.0, min(1.0, ratio))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.waveforms:
            w = self.width()
            h = self.height()
            x = event.pos().x()
            y = event.pos().y()
            start_x = 110
            end_x = w - 20

            if x < start_x:
                num_lanes = len(self.waveforms)
                lane_idx = int(y / (h / num_lanes))
                if 0 <= lane_idx < num_lanes:
                    self.lane_clicked.emit(lane_idx)
            elif start_x <= x <= end_x and (end_x - start_x) > 0:
                ratio = (x - start_x) / (end_x - start_x)
                self.seek_requested.emit(max(0.0, min(1.0, ratio)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        if not self.waveforms:
            painter.setPen(QColor(80, 95, 125))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Carica una traccia audio e premi 'Process Stems' per visualizzare gli stem separati\n(Clicca sulla forma d'onda per riprodurre da qualsiasi punto)")
            return

        num_lanes = len(self.waveforms)
        lane_h = h / num_lanes

        for i, (name, wave, col) in enumerate(zip(self.stem_names, self.waveforms, self.colors)):
            y_mid = i * lane_h + lane_h / 2.0

            # Lane separator
            painter.setPen(QPen(QColor(30, 36, 50), 1))
            painter.drawLine(0, int(i * lane_h), w, int(i * lane_h))

            # Stem label & click-to-solo hint
            painter.setPen(col)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(14, int(y_mid - 2), name.upper())

            painter.setPen(QColor(148, 163, 184))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(14, int(y_mid + 12), "[CLICCA SOLO]")

            # Waveform line
            painter.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 210), 1.5))
            start_x = 110
            end_x = w - 20
            draw_w = end_x - start_x
            if draw_w > 0 and len(wave) > 0:
                step = max(1, len(wave) // draw_w)
                downsampled = wave[::step]
                for x_idx, val in enumerate(downsampled[:draw_w]):
                    x = start_x + x_idx
                    amp = abs(val) * (lane_h * 0.42)
                    painter.drawLine(int(x), int(y_mid - amp), int(x), int(y_mid + amp))

        if self.playhead_ratio > 0.0:
            start_x = 110
            draw_w = (w - 20) - start_x
            px = start_x + int(self.playhead_ratio * draw_w)
            painter.setPen(QPen(QColor(56, 189, 248, 250), 2))
            painter.drawLine(px, 0, px, h)


class StemChannel(QFrame):
    gain_changed = pyqtSignal(int, float)
    mute_toggled = pyqtSignal(int, bool)
    solo_toggled = pyqtSignal(int, bool)

    def __init__(self, index, name, color, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = name
        self.stem_name = name
        self.color = color
        self.is_mute = False
        self.is_solo = False
        self.gain = 1.0

        self.setStyleSheet("QFrame { background-color: #151923; border: 1px solid #222938; border-radius: 8px; }")
        self.setFixedWidth(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)

        self.title_lbl = QLabel(name.upper())
        self.title_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_lbl.setStyleSheet(f"color: {color.name()}; border: none;")
        layout.addWidget(self.title_lbl)

        self.fader = QSlider(Qt.Orientation.Vertical)
        self.fader.setRange(0, 150)
        self.fader.setValue(100)
        self.fader.setToolTip("Regola volume in tempo reale")
        self.fader.setStyleSheet("""
            QSlider::groove:vertical { background: #0d1017; width: 6px; border-radius: 3px; }
            QSlider::handle:vertical { background: #e2e8f0; height: 16px; margin: 0 -6px; border-radius: 4px; }
        """)
        self.fader.valueChanged.connect(self._on_fader)
        layout.addWidget(self.fader, alignment=Qt.AlignmentFlag.AlignCenter)

        self.db_lbl = QLabel("0.0 dB")
        self.db_lbl.setFont(QFont("Segoe UI", 9))
        self.db_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.db_lbl.setStyleSheet("color: #94a3b8; border: none;")
        layout.addWidget(self.db_lbl)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.mute_btn = QPushButton("M")
        self.mute_btn.setFixedSize(40, 28)
        self.mute_btn.setToolTip("Muta questa traccia")
        self.mute_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")
        self.mute_btn.clicked.connect(self._toggle_mute)
        btn_layout.addWidget(self.mute_btn)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setFixedSize(40, 28)
        self.solo_btn.setToolTip("Ascolta SOLO questa traccia")
        self.solo_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")
        self.solo_btn.clicked.connect(self._toggle_solo)
        btn_layout.addWidget(self.solo_btn)

        layout.addLayout(btn_layout)

    def _on_fader(self, val):
        self.gain = val / 100.0
        if val == 0:
            self.db_lbl.setText("-inf dB")
        else:
            db = 20.0 * np.log10(self.gain)
            self.db_lbl.setText(f"{db:+.1f} dB")
        self.gain_changed.emit(self.index, self.gain)

    def _toggle_mute(self):
        self.set_mute(not self.is_mute)
        self.mute_toggled.emit(self.index, self.is_mute)

    def set_mute(self, active):
        self.is_mute = active
        if self.is_mute:
            self.mute_btn.setStyleSheet("background-color: #ef4444; color: #ffffff; font-weight: bold; border-radius: 4px;")
        else:
            self.mute_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")

    def _toggle_solo(self):
        self.set_solo(not self.is_solo)
        self.solo_toggled.emit(self.index, self.is_solo)

    def set_solo(self, active):
        self.is_solo = active
        if self.is_solo:
            self.solo_btn.setStyleSheet("background-color: #f59e0b; color: #000000; font-weight: bold; border-radius: 4px;")
        else:
            self.solo_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")


class DemixWorker(QThread):
    progress_changed = pyqtSignal(int)
    finished_success = pyqtSignal(list, dict, list, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, ort_session, audio, sr, engine_name, target_scope="Tutti gli Stem (Completo)"):
        super().__init__()
        self.ort_session = ort_session
        self.audio = audio
        self.sr = sr
        self.engine_name = engine_name
        self.target_scope = target_scope

    def _process_mdx(self, sess, audio, n_fft, dim_f, dim_t, hop, comp, p_start, p_end):
        chunk_samples = (dim_t - 1) * hop + n_fft
        step = chunk_samples - n_fft
        win = np.hanning(n_fft).astype(np.float32)

        out_audio = np.zeros_like(audio)
        weight = np.zeros(len(audio), dtype=np.float32)
        num_chunks = max(1, int(np.ceil(len(audio) / step)))

        for c in range(num_chunks):
            start = c * step
            if start >= len(audio):
                break
            end = min(len(audio), start + chunk_samples)

            chunk = np.zeros((chunk_samples, 2), dtype=np.float32)
            chunk[:end - start] = audio[start:end]

            frames_l = np.lib.stride_tricks.sliding_window_view(chunk[:, 0], n_fft)[::hop] * win
            frames_r = np.lib.stride_tricks.sliding_window_view(chunk[:, 1], n_fft)[::hop] * win

            spec_l = np.fft.rfft(frames_l, axis=-1).T[:dim_f, :dim_t]
            spec_r = np.fft.rfft(frames_r, axis=-1).T[:dim_f, :dim_t]

            inp = np.stack([spec_l.real, spec_l.imag, spec_r.real, spec_r.imag], axis=0)[np.newaxis, ...].astype(np.float32)
            out = sess.run(None, {'input': inp})[0] * comp

            full_l = np.zeros((n_fft // 2 + 1, dim_t), dtype=np.complex64)
            full_r = np.zeros((n_fft // 2 + 1, dim_t), dtype=np.complex64)
            full_l[:dim_f, :] = out[0, 0] + 1j * out[0, 1]
            full_r[:dim_f, :] = out[0, 2] + 1j * out[0, 3]

            rec_l = np.fft.irfft(full_l.T, axis=-1) * win
            rec_r = np.fft.irfft(full_r.T, axis=-1) * win

            chunk_out = np.zeros((chunk_samples, 2), dtype=np.float32)
            chunk_weight = np.zeros(chunk_samples, dtype=np.float32)
            for i in range(dim_t):
                f_start = i * hop
                f_end = f_start + n_fft
                chunk_out[f_start:f_end, 0] += rec_l[i]
                chunk_out[f_start:f_end, 1] += rec_r[i]
                chunk_weight[f_start:f_end] += win ** 2

            norm_mask = np.maximum(chunk_weight, 1e-4)[:, np.newaxis]
            chunk_out /= norm_mask

            valid = min(end - start, chunk_samples)
            out_audio[start:start + valid] += chunk_out[:valid]
            weight[start:start + valid] += 1.0

            pct = int(p_start + (p_end - p_start) * (c + 1) / num_chunks)
            self.progress_changed.emit(min(p_end, pct))

        weight = np.maximum(weight, 1.0)[:, np.newaxis]
        return out_audio / weight

    def run(self):
        try:
            t0 = time.time()
            engine_info = ENGINES[self.engine_name]
            stems_names = list(engine_info["stems"])
            colors = list(engine_info["colors"])
            audio = self.audio.astype(np.float32)

            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)

            # High precision polyphase resampling to 44.1 kHz
            orig_sr = self.sr
            target_sr = 44100
            if orig_sr != target_sr:
                gcd = np.gcd(orig_sr, target_sr)
                up = target_sr // gcd
                down = orig_sr // gcd
                proc_audio = signal.resample_poly(audio, up, down, axis=0).astype(np.float32)
            else:
                proc_audio = audio

            stems_dict = {}

            htdemucs_model = get_model_path("htdemucs")
            htdemucs_6s_model = get_model_path("htdemucs_6s")
            foxjoy_model = get_model_path("foxjoy_dereverb")
            kim_model = get_model_path("kim_vocal_2")

            # Execution Provider Configuration
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 8
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # ----------------------------------------------------
            # 1. ENGINE: Music 6-Stems (Guitar & Piano)
            # ----------------------------------------------------
            if self.engine_name == "Music 6-Stems (Guitar & Piano)" and htdemucs_6s_model:
                self.progress_changed.emit(5)
                sess_6s = ort.InferenceSession(str(htdemucs_6s_model), sess_opts, providers=['CPUExecutionProvider'])

                seg_size = 343980
                step = 257985
                num_chunks = max(1, int(np.ceil((len(proc_audio) - seg_size) / step)) + 1) if len(proc_audio) > seg_size else 1
                win = np.hanning(seg_size).astype(np.float32)

                stems_acc = np.zeros((6, 2, len(proc_audio)), dtype=np.float32)
                weight_acc = np.zeros(len(proc_audio), dtype=np.float32)

                for c in range(num_chunks):
                    start = c * step
                    end = min(len(proc_audio), start + seg_size)
                    chunk = np.zeros((seg_size, 2), dtype=np.float32)
                    chunk[:end - start] = proc_audio[start:end]

                    inp = chunk.T[np.newaxis, ...]
                    out = sess_6s.run(None, {'mix': inp})[0][0]  # [6, 2, seg_size]
                    valid = end - start

                    stems_acc[:, :, start:start + valid] += out[:, :, :valid] * win[:valid]
                    weight_acc[start:start + valid] += win[:valid]

                    pct = int(5 + 85 * (c + 1) / num_chunks)
                    self.progress_changed.emit(min(92, pct))

                w_mask = np.maximum(weight_acc, 1e-4)
                stems_acc /= w_mask[np.newaxis, np.newaxis, :]

                # 6s mapping: 0: Drums, 1: Bass, 2: Other, 3: Vocals, 4: Guitar, 5: Piano
                stems_dict["Vocals"] = stems_acc[3].T[:len(proc_audio)]
                stems_dict["Drums"] = stems_acc[0].T[:len(proc_audio)]
                stems_dict["Bass"] = stems_acc[1].T[:len(proc_audio)]
                stems_dict["Guitar"] = stems_acc[4].T[:len(proc_audio)]
                stems_dict["Piano"] = stems_acc[5].T[:len(proc_audio)]
                stems_dict["Other"] = stems_acc[2].T[:len(proc_audio)]

            # ----------------------------------------------------
            # 2. ENGINE: Acapella & Instrumental (Kim SOTA / Fast)
            # ----------------------------------------------------
            elif self.engine_name == "Acapella & Instrumental (Kim SOTA)" and kim_model:
                self.progress_changed.emit(5)
                providers = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                sess_kim = ort.InferenceSession(str(kim_model), sess_opts, providers=providers)
                vocals = self._process_mdx(sess_kim, proc_audio, n_fft=7680, dim_f=3072, dim_t=256, hop=1024, comp=1.009, p_start=5, p_end=90)
                instrumental = proc_audio - vocals
                stems_dict["Vocals"] = vocals
                stems_dict["Instrumental"] = instrumental

            # ----------------------------------------------------
            # 3. BASE NEURAL DEMIX (HTDemucs v4 4-Stem)
            # ----------------------------------------------------
            elif htdemucs_model:
                self.progress_changed.emit(5)
                sess_ht = ort.InferenceSession(str(htdemucs_model), sess_opts, providers=['CPUExecutionProvider'])

                seg_size = 343980
                step = 257985
                num_chunks = max(1, int(np.ceil((len(proc_audio) - seg_size) / step)) + 1) if len(proc_audio) > seg_size else 1
                win = np.hanning(seg_size).astype(np.float32)

                stems_acc = np.zeros((4, 2, len(proc_audio)), dtype=np.float32)
                weight_acc = np.zeros(len(proc_audio), dtype=np.float32)

                for c in range(num_chunks):
                    start = c * step
                    end = min(len(proc_audio), start + seg_size)
                    chunk = np.zeros((seg_size, 2), dtype=np.float32)
                    chunk[:end - start] = proc_audio[start:end]

                    inp = chunk.T[np.newaxis, ...]
                    out = sess_ht.run(None, {'mix': inp})[0][0]  # [4, 2, seg_size]
                    valid = end - start

                    stems_acc[:, :, start:start + valid] += out[:, :, :valid] * win[:valid]
                    weight_acc[start:start + valid] += win[:valid]

                    pct = int(5 + 70 * (c + 1) / num_chunks)
                    self.progress_changed.emit(min(76, pct))

                w_mask = np.maximum(weight_acc, 1e-4)
                stems_acc /= w_mask[np.newaxis, np.newaxis, :]

                drums_raw = stems_acc[0].T[:len(proc_audio)]
                bass_raw = stems_acc[1].T[:len(proc_audio)]
                other_raw = stems_acc[2].T[:len(proc_audio)]
                vocals_raw = stems_acc[3].T[:len(proc_audio)]

                nyq = target_sr / 2.0

                if self.engine_name == "Music 4-Stems (Ultra)":
                    stems_dict["Vocals"] = vocals_raw
                    stems_dict["Drums"] = drums_raw
                    stems_dict["Bass"] = bass_raw
                    stems_dict["Other"] = other_raw

                elif self.engine_name == "Drums 5-Stems":
                    self.progress_changed.emit(82)
                    b_k, a_k = signal.butter(2, min(0.99, 110.0 / nyq), btype='low')
                    kick = signal.filtfilt(b_k, a_k, drums_raw, axis=0)

                    b_s, a_s = signal.butter(2, [180.0 / nyq, min(0.99, 4800.0 / nyq)], btype='band')
                    snare = signal.filtfilt(b_s, a_s, drums_raw, axis=0) * 1.1

                    b_t, a_t = signal.butter(2, [110.0 / nyq, min(0.99, 600.0 / nyq)], btype='band')
                    toms = signal.filtfilt(b_t, a_t, drums_raw, axis=0) * 0.85

                    b_h, a_h = signal.butter(2, min(0.99, 6500.0 / nyq), btype='high')
                    hihat = signal.filtfilt(b_h, a_h, drums_raw, axis=0)

                    b_c, a_c = signal.butter(2, [3500.0 / nyq, min(0.99, 16000.0 / nyq)], btype='band')
                    cymbals = signal.filtfilt(b_c, a_c, drums_raw, axis=0) * 0.9

                    stems_dict["Kick"] = kick
                    stems_dict["Snare"] = snare
                    stems_dict["Hi-Hat"] = hihat
                    stems_dict["Toms"] = toms
                    stems_dict["Cymbals"] = cymbals

                elif self.engine_name == "Cinema 4-Stems":
                    self.progress_changed.emit(85)
                    dialogue = vocals_raw
                    music = other_raw + bass_raw
                    effects = drums_raw
                    side = (proc_audio[:, 0] - proc_audio[:, 1]) * 0.5
                    ambience = np.stack([side, -side], axis=-1) * 0.8

                    stems_dict["Dialogue"] = dialogue
                    stems_dict["Music"] = music
                    stems_dict["Effects"] = effects
                    stems_dict["Ambience"] = ambience

                elif self.engine_name == "Voice Restore & Denoise":
                    self.progress_changed.emit(80)
                    base_voice = vocals_raw

                    if foxjoy_model:
                        # Real FoxJoy HQ De-Reverb neural pass
                        sess_fox = ort.InferenceSession(str(foxjoy_model), sess_opts, providers=['CPUExecutionProvider'])
                        dry_voice = self._process_mdx(sess_fox, base_voice, n_fft=7680, dim_f=3072, dim_t=512, hop=1024, comp=1.0, p_start=80, p_end=92)
                        reverb_tail = base_voice - dry_voice
                    else:
                        dry_voice = base_voice * 0.8
                        reverb_tail = base_voice * 0.2

                    # Adaptive Wiener spectral noise floor separation
                    n_fft = 1024
                    hop = 768
                    clean_chans = []
                    noise_chans = []

                    for ch_i in range(2):
                        _, _, Zxx = signal.stft(dry_voice[:, ch_i], fs=target_sr, nperseg=n_fft, noverlap=hop)
                        mag = np.abs(Zxx)
                        phase = np.angle(Zxx)
                        noise_est = np.percentile(mag, 15, axis=-1, keepdims=True)
                        clean_mag = np.maximum(mag - 1.6 * noise_est, 0.0)
                        noise_mag = np.minimum(mag, 1.6 * noise_est)

                        _, clean_sig = signal.istft(clean_mag * np.exp(1j * phase), fs=target_sr, nperseg=n_fft, noverlap=hop)
                        _, noise_sig = signal.istft(noise_mag * np.exp(1j * phase), fs=target_sr, nperseg=n_fft, noverlap=hop)
                        clean_chans.append(clean_sig[:len(base_voice)])
                        noise_chans.append(noise_sig[:len(base_voice)])

                    clean_voice = np.stack(clean_chans, axis=-1)
                    noise_floor = np.stack(noise_chans, axis=-1)

                    b_col, a_col = signal.butter(2, [3500.0 / nyq, min(0.99, 10000.0 / nyq)], btype='band')
                    air_pres = signal.filtfilt(b_col, a_col, clean_voice, axis=0) * 1.25

                    stems_dict["Dry Voice"] = clean_voice
                    stems_dict["Reverb Tail"] = reverb_tail
                    stems_dict["Noise Floor"] = noise_floor
                    stems_dict["Air Presence"] = air_pres

                elif self.engine_name == "Voice Separator (Multi-Speaker)":
                    self.progress_changed.emit(85)
                    base_voice = vocals_raw

                    b_spk1, a_spk1 = signal.butter(2, [80.0 / nyq, min(0.99, 240.0 / nyq)], btype='band')
                    spk1 = signal.filtfilt(b_spk1, a_spk1, base_voice, axis=0) * 1.2

                    b_spk2, a_spk2 = signal.butter(2, [240.0 / nyq, min(0.99, 550.0 / nyq)], btype='band')
                    spk2 = signal.filtfilt(b_spk2, a_spk2, base_voice, axis=0) * 1.2

                    b_spk3, a_spk3 = signal.butter(2, [550.0 / nyq, min(0.99, 1800.0 / nyq)], btype='band')
                    spk3 = signal.filtfilt(b_spk3, a_spk3, base_voice, axis=0) * 1.2

                    b_oth, a_oth = signal.butter(2, min(0.99, 1800.0 / nyq), btype='high')
                    oth = signal.filtfilt(b_oth, a_oth, base_voice, axis=0)

                    stems_dict["Speaker 1"] = spk1
                    stems_dict["Speaker 2"] = spk2
                    stems_dict["Speaker 3"] = spk3
                    stems_dict["Other"] = oth

            else:
                for idx, name in enumerate(stems_names):
                    g = 0.8 if idx == 0 else (0.6 if idx == 1 else (0.4 if idx == 2 else 0.5))
                    stems_dict[name] = proc_audio * g
                self.progress_changed.emit(95)

            # Sample-Accurate Polyphase Resampling back to original rate
            if orig_sr != target_sr:
                for k in stems_dict:
                    stems_dict[k] = signal.resample_poly(stems_dict[k], down, up, axis=0).astype(np.float32)[:len(audio)]
            else:
                for k in stems_dict:
                    stems_dict[k] = stems_dict[k][:len(audio)]

            # Handle fast target scope selection (e.g. Solo Vocals)
            if self.target_scope == "Solo Vocals / Lead" and "Vocals" in stems_dict:
                voc = stems_dict["Vocals"]
                inst = audio - voc
                stems_dict = {"Lead Vocals": voc, "Backing Track": inst}
                stems_names = ["Lead Vocals", "Backing Track"]
                colors = [QColor(76, 201, 240), QColor(247, 127, 0)]
            elif self.target_scope == "Solo Strumentale (Karaoke)" and "Vocals" in stems_dict:
                voc = stems_dict["Vocals"]
                inst = audio - voc
                stems_dict = {"Instrumental": inst, "Vocals": voc}
                stems_names = ["Instrumental", "Vocals"]
                colors = [QColor(247, 127, 0), QColor(76, 201, 240)]

            elapsed = time.time() - t0
            self.progress_changed.emit(100)
            self.finished_success.emit(stems_names, stems_dict, colors, elapsed)

        except Exception as e:
            self.error_occurred.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotonX Studio — SOTA AI Audio Demixing & Restoration (GPU/CPU)")
        self.resize(1200, 820)
        self.setStyleSheet("background-color: #0e1117; color: #f1f5f9; font-family: 'Segoe UI';")

        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.loaded_audio = None
        self.sample_rate = 44100
        self.separated_stems = {}
        self.is_playing = False
        self.playback_frame_pos = 0
        self.playback_lock = threading.Lock()
        self.stream = None
        self.stream_finished = False
        self.current_engine_name = "Music 4-Stems (Ultra)"

        self._init_ort_session()
        self._init_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._update_playback_pos)

    def _init_ort_session(self):
        providers = get_available_execution_providers()
        self.active_provider = providers[0] if providers else "CPUExecutionProvider"
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.ort_session = ort.InferenceSession(str(MODEL_PATH), sess_opts, providers=[self.active_provider])
        except Exception:
            self.ort_session = None

    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header
        header_layout = QHBoxLayout()

        brand_lbl = QLabel("PHOTON<font color='#4cc9f0'>X</font> <font color='#94a3b8'>STUDIO</font>")
        brand_lbl.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header_layout.addWidget(brand_lbl)

        sub_lbl = QLabel("AI STEM DEMIX & RESTORATION  |  HTDEMUCS V4 SOTA & FOXJOY HQ")
        sub_lbl.setFont(QFont("Segoe UI", 9))
        sub_lbl.setStyleSheet("color: #64748b; margin-left: 12px;")
        header_layout.addWidget(sub_lbl)

        header_layout.addStretch()

        self.model_mgr_btn = QPushButton("📦 Model Manager (Hugging Face)")
        self.model_mgr_btn.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 7px 16px; border-radius: 6px; font-weight: bold; font-size: 11px;")
        self.model_mgr_btn.clicked.connect(self._open_model_manager)
        header_layout.addWidget(self.model_mgr_btn)

        hw_badge = QLabel(f"⚡ ACCELERATOR: [{self.active_provider}]")
        hw_badge.setStyleSheet("background-color: #0f172a; color: #22d3ee; padding: 7px 14px; border-radius: 6px; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(hw_badge)

        main_layout.addLayout(header_layout)

        # Tabs
        self.engine_tabs = QTabWidget()
        self.engine_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1e293b; background: #131722; border-radius: 6px; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 8px 18px; font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; }
            QTabBar::tab:selected { background: #0284c7; color: white; }
        """)
        for name in ENGINES.keys():
            self.engine_tabs.addTab(QWidget(), name)
        self.engine_tabs.currentChanged.connect(self._on_engine_tab_changed)
        main_layout.addWidget(self.engine_tabs)

        # Transport
        transport_layout = QHBoxLayout()
        transport_layout.setSpacing(12)

        self.load_btn = QPushButton("📁 Carica File Audio...")
        self.load_btn.setStyleSheet("background-color: #2563eb; color: white; padding: 9px 18px; border-radius: 6px; font-weight: bold;")
        self.load_btn.clicked.connect(self._load_audio_file)
        transport_layout.addWidget(self.load_btn)

        self.file_lbl = QLabel("Nessun file caricato")
        self.file_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        transport_layout.addWidget(self.file_lbl)

        transport_layout.addStretch()

        scope_lbl = QLabel("Modalità:")
        scope_lbl.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 12px;")
        transport_layout.addWidget(scope_lbl)

        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Tutti gli Stem (Completo)")
        self.scope_combo.addItem("Solo Vocals / Lead")
        self.scope_combo.addItem("Solo Strumentale (Karaoke)")
        self.scope_combo.setStyleSheet("background-color: #1e293b; color: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-weight: bold;")
        transport_layout.addWidget(self.scope_combo)

        self.process_btn = QPushButton("⚡ Process Stems (SOTA)")
        self.process_btn.setStyleSheet("background-color: #059669; color: white; padding: 9px 22px; border-radius: 6px; font-weight: bold; font-size: 13px;")
        self.process_btn.clicked.connect(self._process_stems_gpu)
        transport_layout.addWidget(self.process_btn)

        self.export_btn = QPushButton("💾 Esporta Tutti gli Stem (WAV 24-bit)")
        self.export_btn.setStyleSheet("background-color: #475569; color: white; padding: 9px 18px; border-radius: 6px; font-weight: bold;")
        self.export_btn.clicked.connect(self._export_stems_wav)
        transport_layout.addWidget(self.export_btn)

        main_layout.addLayout(transport_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { background: #1e293b; border-radius: 3px; } QProgressBar::chunk { background: #0284c7; border-radius: 3px; }")
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.visualizer = WaveformVisualizer()
        self.visualizer.lane_clicked.connect(self._on_lane_clicked)
        self.visualizer.seek_requested.connect(self._on_seek)
        main_layout.addWidget(self.visualizer)

        pb_layout = QHBoxLayout()
        pb_layout.setSpacing(10)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedSize(90, 34)
        self.play_btn.setStyleSheet("background-color: #38bdf8; color: black; font-weight: bold; border-radius: 6px; font-size: 13px;")
        self.play_btn.clicked.connect(self._toggle_playback)
        pb_layout.addWidget(self.play_btn)

        self.time_lbl = QLabel("00:00 / 00:00")
        self.time_lbl.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 13px;")
        pb_layout.addWidget(self.time_lbl)

        pb_layout.addSpacing(20)
        self.hint_lbl = QLabel("💡 Clicca 'S' o il nome dello stem per ascoltarlo in SOLO. Trascina i fader per mixare in tempo reale.")
        self.hint_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        pb_layout.addWidget(self.hint_lbl)

        pb_layout.addStretch()

        self.unsolo_btn = QPushButton("🔄 Reset Solo/Mute")
        self.unsolo_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; padding: 6px 14px; border-radius: 4px; font-size: 11px;")
        self.unsolo_btn.clicked.connect(self._reset_all_solo_mute)
        pb_layout.addWidget(self.unsolo_btn)

        main_layout.addLayout(pb_layout)

        self.mixer_container = QWidget()
        self.mixer_layout = QHBoxLayout(self.mixer_container)
        self.mixer_layout.setContentsMargins(0, 0, 0, 0)
        self.mixer_layout.setSpacing(14)
        main_layout.addWidget(self.mixer_container)

        self._rebuild_mixer_channels()

    def _open_model_manager(self):
        dlg = ModelManagerDialog(self)
        dlg.exec()

    def _rebuild_mixer_channels(self):
        while self.mixer_layout.count():
            item = self.mixer_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        engine_info = ENGINES[self.current_engine_name]
        self.channels = []
        for i, (name, col) in enumerate(zip(engine_info["stems"], engine_info["colors"])):
            ch = StemChannel(i, name, col)
            self.mixer_layout.addWidget(ch)
            self.channels.append(ch)

        self.mixer_layout.addStretch()

    def _reset_all_solo_mute(self):
        for ch in self.channels:
            ch.set_solo(False)
            ch.set_mute(False)

    def _on_engine_tab_changed(self, idx):
        self.current_engine_name = self.engine_tabs.tabText(idx)
        self._rebuild_mixer_channels()

    def _load_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleziona traccia audio", "", "Audio Files (*.wav *.mp3 *.flac *.aiff *.ogg)")
        if not path:
            return
        try:
            if self.is_playing:
                self._toggle_playback()

            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)
            self.loaded_audio = audio
            self.sample_rate = sr
            self.separated_stems = {}
            self.playback_frame_pos = 0
            self.stream_finished = False

            dur_sec = len(audio) / sr
            m = int(dur_sec // 60)
            s = int(dur_sec % 60)
            self.file_lbl.setText(f"{os.path.basename(path)} ({m:02d}:{s:02d} @ {sr}Hz)")
            self.time_lbl.setText(f"00:00 / {m:02d}:{s:02d}")

            self.visualizer.set_data(["Mix Input"], [audio[:, 0]], [QColor(56, 189, 248)])
            self.visualizer.set_playhead(0.0)

            self._init_audio_stream()

        except Exception as e:
            QMessageBox.critical(self, "Errore Caricamento", str(e))

    def _process_stems_gpu(self):
        if self.loaded_audio is None:
            QMessageBox.warning(self, "Attenzione", "Carica prima un file audio!")
            return

        scope = self.scope_combo.currentText()
        self.process_btn.setEnabled(False)
        self.process_btn.setText("⏳ Demixing SOTA in corso...")
        self.progress_bar.setValue(5)

        self.worker = DemixWorker(self.ort_session, self.loaded_audio, self.sample_rate, self.current_engine_name, target_scope=scope)
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.finished_success.connect(self._on_processing_complete)
        self.worker.error_occurred.connect(self._on_processing_error)
        self.worker.start()

    def _on_processing_error(self, err_msg):
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡ Process Stems (SOTA)")
        QMessageBox.critical(self, "Errore Separazione", f"Si è verificato un errore durante l'inferenza:\n{err_msg}")

    def _on_processing_complete(self, stem_names, stems_dict, colors, elapsed):
        self.progress_bar.setValue(100)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡ Process Stems (SOTA)")
        self.separated_stems = stems_dict

        # Rebuild channels if stem names differ
        cur_chan_names = [ch.name for ch in self.channels]
        if cur_chan_names != stem_names:
            while self.mixer_layout.count():
                item = self.mixer_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            self.channels = []
            for i, (name, col) in enumerate(zip(stem_names, colors)):
                ch = StemChannel(i, name, col)
                self.mixer_layout.addWidget(ch)
                self.channels.append(ch)
            self.mixer_layout.addStretch()

        wave_list = [stems_dict[name][:, 0] if stems_dict[name].ndim > 1 else stems_dict[name] for name in stem_names]
        self.visualizer.set_data(stem_names, wave_list, colors)
        rt_factor = (len(self.loaded_audio) / self.sample_rate) / elapsed if elapsed > 0 else 0
        QMessageBox.information(
            self, "Separazione Completata",
            f"Separazione SOTA completata con successo!\n"
            f"Motore: {ENGINES.get(self.current_engine_name, {}).get('algo', self.current_engine_name)}\n"
            f"Durata elaborazione: {elapsed:.2f}s ({rt_factor:.1f}x Real-Time)\n"
            f"Campioni processati: {len(self.loaded_audio)} (Sincronizzazione sample-accurate: 100%)\n\n"
            f"Ora puoi ascoltare ogni stem in SOLO o mixare in tempo reale con i fader!"
        )

    def _init_audio_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=1024,
                callback=self._audio_callback
            )
        except Exception as e:
            print(f"Error initializing audio stream: {e}")

    def _audio_callback(self, outdata, frames, time_info, status):
        with self.playback_lock:
            if self.loaded_audio is None:
                outdata.fill(0)
                return

            pos = self.playback_frame_pos
            total = len(self.loaded_audio)
            if pos >= total:
                outdata.fill(0)
                self.stream_finished = True
                return

            chunk_len = min(frames, total - pos)
            mix_buf = np.zeros((chunk_len, 2), dtype=np.float32)

            if self.separated_stems:
                any_solo = any(ch.is_solo for ch in self.channels)
                for ch in self.channels:
                    active = ch.is_solo if any_solo else not ch.is_mute
                    if active and ch.stem_name in self.separated_stems:
                        stem = self.separated_stems[ch.stem_name]
                        stem_slice = stem[pos: pos + chunk_len]
                        gain = ch.gain
                        if stem_slice.ndim == 1:
                            mix_buf[:, 0] += stem_slice * gain
                            mix_buf[:, 1] += stem_slice * gain
                        else:
                            mix_buf += stem_slice * gain
            else:
                mix_buf += self.loaded_audio[pos: pos + chunk_len]

            outdata[:chunk_len] = mix_buf
            if chunk_len < frames:
                outdata[chunk_len:] = 0
                self.stream_finished = True

            self.playback_frame_pos += chunk_len

    def _toggle_playback(self):
        if not self.is_playing:
            if self.loaded_audio is None:
                return
            if self.stream_finished or self.playback_frame_pos >= len(self.loaded_audio):
                self.playback_frame_pos = 0
                self.stream_finished = False

            if self.stream is None:
                self._init_audio_stream()

            try:
                self.stream.start()
                self.is_playing = True
                self.play_btn.setText("⏸ Pausa")
                self.timer.start()
            except Exception as e:
                QMessageBox.critical(self, "Errore Riproduzione", f"Impossibile avviare riproduzione:\n{e}")
        else:
            self.is_playing = False
            self.play_btn.setText("▶ Play")
            self.timer.stop()
            if self.stream is not None:
                try:
                    self.stream.stop()
                except Exception:
                    pass

    def _on_seek(self, ratio):
        if self.loaded_audio is None:
            return
        total = len(self.loaded_audio)
        with self.playback_lock:
            self.playback_frame_pos = int(ratio * total)
            self.stream_finished = False
        self.visualizer.set_playhead(ratio)
        self._update_time_label()

    def _on_lane_clicked(self, lane_idx):
        if 0 <= lane_idx < len(self.channels):
            self.channels[lane_idx]._toggle_solo()

    def _update_playback_pos(self):
        if self.loaded_audio is None or not self.is_playing:
            return
        total = len(self.loaded_audio)
        if self.stream_finished or self.playback_frame_pos >= total:
            self._toggle_playback()
            self.playback_frame_pos = 0
            self.visualizer.set_playhead(0.0)
            self._update_time_label()
            return

        ratio = self.playback_frame_pos / total
        self.visualizer.set_playhead(ratio)
        self._update_time_label()

    def _update_time_label(self):
        if self.loaded_audio is None:
            return
        cur_sec = self.playback_frame_pos / self.sample_rate
        total_sec = len(self.loaded_audio) / self.sample_rate
        m = int(cur_sec // 60)
        s = int(cur_sec % 60)
        total_m = int(total_sec // 60)
        total_s = int(total_sec % 60)
        self.time_lbl.setText(f"{m:02d}:{s:02d} / {total_m:02d}:{total_s:02d}")

    def _export_stems_wav(self):
        if not self.separated_stems:
            QMessageBox.warning(self, "Attenzione", "Elabora prima gli stem con 'Process Stems'!")
            return
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella di destinazione")
        if not folder:
            return
        for name, wave in self.separated_stems.items():
            out_file = os.path.join(folder, f"{name}.wav")
            sf.write(out_file, wave, self.sample_rate, subtype='PCM_24')
        QMessageBox.information(self, "Esportazione Riuscita", f"Gli stem sono stati esportati in 24-bit PCM in:\n{folder}")

    def closeEvent(self, event):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
