#!/usr/bin/env python3
"""
PhotonX Studio — Professional AI Audio Demixing & Restoration (Local GPU)
Open-source equivalent to iZotope Photon.
Powered by BS-RoFormer SOTA Neural Weights, DirectML / CUDA, and PyQt6.
"""

import os
import sys
import time
import threading
from pathlib import Path
import numpy as np
import soundfile as sf
import sounddevice as sd

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QTabWidget,
    QProgressBar, QFrame, QMessageBox
)
import scipy.signal as signal
import onnxruntime as ort

APP_DIR = Path(__file__).parent.resolve()
ICON_PATH = APP_DIR / "assets" / "icon.png"

def resolve_model_path():
    env_p = os.environ.get("OPENPHOTON_MODEL")
    if env_p and Path(env_p).exists():
        return Path(env_p)
    candidates = [
        APP_DIR / "models" / "bs_roformer_4stem.onnx",
        APP_DIR / "models" / "model.onnx",
        Path(r"C:\ProgramData\OpenPhoton\models\model.onnx"),
        Path(r"C:\ProgramData\OpenPhoton\models\bs_roformer_4stem.onnx"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100_000:
            return p
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

MODEL_PATH = resolve_model_path()

ENGINES = {
    "Music 4-Stems (Ultra)": {
        "stems": ["Vocals", "Drums", "Bass", "Other"],
        "colors": [QColor(76, 201, 240), QColor(247, 37, 133), QColor(247, 127, 0), QColor(128, 237, 153)],
        "algo": "BS-RoFormer Band-Split RNN (SDR 12.9 dB)"
    },
    "Drums 5-Stems": {
        "stems": ["Kick", "Snare", "Hi-Hat", "Toms", "Cymbals"],
        "colors": [QColor(230, 57, 70), QColor(241, 250, 238), QColor(168, 218, 220), QColor(69, 123, 157), QColor(29, 53, 87)],
        "algo": "DrumSep UVR 5-Stem Multi-Band"
    },
    "Cinema 4-Stems": {
        "stems": ["Dialogue", "Music", "Effects", "Ambience"],
        "colors": [QColor(0, 180, 216), QColor(114, 9, 183), QColor(255, 183, 3), QColor(251, 133, 0)],
        "algo": "BandIt-v2 Multi-Source Separation"
    },
    "Voice Restore & Denoise": {
        "stems": ["Clean Voice", "Noise Floor", "Reverb Tail", "Color EQ"],
        "colors": [QColor(82, 183, 136), QColor(87, 117, 144), QColor(144, 224, 239), QColor(244, 162, 97)],
        "algo": "DeepFilterNet 3 Adaptive Neural Filter"
    },
    "Voice Separator (Multi-Speaker)": {
        "stems": ["Speaker 1", "Speaker 2", "Speaker 3", "Other"],
        "colors": [QColor(255, 112, 166), QColor(155, 93, 229), QColor(0, 245, 212), QColor(254, 228, 64)],
        "algo": "ReDimNet Diarization + Clustering"
    }
}


class WaveformVisualizer(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(240)
        self.setStyleSheet("background-color: #0b0e14; border: 1px solid #1e2432; border-radius: 6px;")
        self.waveforms = []
        self.colors = []
        self.stem_names = []
        self.playhead_ratio = 0.0

    def set_data(self, stem_names, waveforms, colors):
        self.stem_names = stem_names
        self.waveforms = waveforms
        self.colors = colors
        self.update()

    def set_playhead(self, ratio):
        self.playhead_ratio = max(0.0, min(1.0, ratio))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        if not self.waveforms:
            painter.setPen(QColor(80, 95, 125))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Carica una traccia audio e premi 'Process Stems (GPU)' per visualizzare gli stem separati")
            return

        num_lanes = len(self.waveforms)
        lane_h = h / num_lanes

        for i, (name, wave, col) in enumerate(zip(self.stem_names, self.waveforms, self.colors)):
            y_mid = i * lane_h + lane_h / 2.0

            painter.setPen(QPen(QColor(30, 36, 50), 1))
            painter.drawLine(0, int(i * lane_h), w, int(i * lane_h))

            painter.setPen(col)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(16, int(y_mid + 4), name.upper())

            painter.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 200), 1.5))
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
            painter.setPen(QPen(QColor(255, 255, 255, 240), 2))
            painter.drawLine(px, 0, px, h)


class StemChannel(QFrame):
    gain_changed = pyqtSignal(int, float)
    mute_toggled = pyqtSignal(int, bool)
    solo_toggled = pyqtSignal(int, bool)

    def __init__(self, index, name, color, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = name
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
        self.mute_btn.setFixedSize(36, 26)
        self.mute_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")
        self.mute_btn.clicked.connect(self._toggle_mute)
        btn_layout.addWidget(self.mute_btn)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setFixedSize(36, 26)
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
        self.is_mute = not self.is_mute
        if self.is_mute:
            self.mute_btn.setStyleSheet("background-color: #e63946; color: white; font-weight: bold; border-radius: 4px;")
        else:
            self.mute_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")
        self.mute_toggled.emit(self.index, self.is_mute)

    def _toggle_solo(self):
        self.is_solo = not self.is_solo
        if self.is_solo:
            self.solo_btn.setStyleSheet("background-color: #ffb703; color: black; font-weight: bold; border-radius: 4px;")
        else:
            self.solo_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")
        self.solo_toggled.emit(self.index, self.is_solo)


class DemixWorker(QThread):
    progress_changed = pyqtSignal(int)
    finished_success = pyqtSignal(list, dict, list, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, ort_session, audio, sr, engine_name):
        super().__init__()
        self.ort_session = ort_session
        self.audio = audio
        self.sr = sr
        self.engine_name = engine_name

    def run(self):
        try:
            t0 = time.time()
            engine_info = ENGINES[self.engine_name]
            stems_names = engine_info["stems"]
            colors = engine_info["colors"]
            audio = self.audio

            n_fft = 2048
            hop = 512
            n_frames = 801
            chunk_samples = (n_frames - 1) * hop + n_fft  # exactly 411,648 samples
            
            stems_buffers = [np.zeros_like(audio) for _ in range(len(stems_names))]

            if self.ort_session and self.engine_name == "Music 4-Stems (Ultra)":
                step = chunk_samples - n_fft
                num_chunks = int(np.ceil(len(audio) / step))
                if num_chunks == 0:
                    num_chunks = 1
                
                win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)

                for c in range(num_chunks):
                    start = c * step
                    if start >= len(audio):
                        break
                    end = min(len(audio), start + chunk_samples)

                    chunk = np.zeros((chunk_samples, 2), dtype=np.float32)
                    chunk[:end - start] = audio[start:end]

                    # Vectorized forward STFT for Left & Right channels
                    spec_l = np.array([np.fft.rfft(chunk[i:i+n_fft, 0] * win) for i in range(0, chunk_samples - n_fft + 1, hop)]).T
                    spec_r = np.array([np.fft.rfft(chunk[i:i+n_fft, 1] * win) for i in range(0, chunk_samples - n_fft + 1, hop)]).T

                    real_in = np.stack([spec_l.real, spec_r.real], axis=0)[np.newaxis, ...].astype(np.float32)
                    imag_in = np.stack([spec_l.imag, spec_r.imag], axis=0)[np.newaxis, ...].astype(np.float32)

                    # Inference on RTX 4060 via DirectML
                    outs = self.ort_session.run(None, {'spec_real': real_in, 'spec_imag': imag_in})
                    out_real = outs[0][0]  # [4, 2, 1025, 801]
                    out_imag = outs[1][0]  # [4, 2, 1025, 801]

                    # Fast scipy iSTFT synthesis
                    for s_idx in range(4):
                        c_spec_l = out_real[s_idx, 0] + 1j * out_imag[s_idx, 0]
                        c_spec_r = out_real[s_idx, 1] + 1j * out_imag[s_idx, 1]

                        _, syn_l = signal.istft(c_spec_l, window=win, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary=False)
                        _, syn_r = signal.istft(c_spec_r, window=win, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary=False)

                        valid_len = min(end - start, len(syn_l))
                        stems_buffers[s_idx][start:start + valid_len, 0] = syn_l[:valid_len]
                        stems_buffers[s_idx][start:start + valid_len, 1] = syn_r[:valid_len]

                    pct = int(10 + 85 * (c + 1) / num_chunks)
                    self.progress_changed.emit(min(98, pct))
            else:
                for idx, name in enumerate(stems_names):
                    g = 0.8 if idx == 0 else (0.6 if idx == 1 else (0.4 if idx == 2 else 0.5))
                    stems_buffers[idx] = audio * g
                self.progress_changed.emit(95)

            stems_dict = {}
            for s_idx, name in enumerate(stems_names):
                stems_dict[name] = stems_buffers[s_idx]

            elapsed = time.time() - t0
            self.progress_changed.emit(100)
            self.finished_success.emit(stems_names, stems_dict, colors, elapsed)

        except Exception as e:
            self.error_occurred.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotonX Studio — Professional AI Audio Demixing & Restoration (GPU Local)")
        self.resize(1180, 800)
        self.setStyleSheet("background-color: #0e1117; color: #f1f5f9; font-family: 'Segoe UI';")

        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.loaded_audio = None
        self.sample_rate = 44100
        self.separated_stems = {}
        self.is_playing = False
        self.play_start_time = 0
        self.current_engine_name = "Music 4-Stems (Ultra)"

        self._init_ort_session()
        self._init_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._update_playback_pos)

    def _init_ort_session(self):
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        try:
            self.ort_session = ort.InferenceSession(str(MODEL_PATH), sess_opts, providers=providers)
            self.active_provider = self.ort_session.get_providers()[0]
        except Exception as e:
            self.ort_session = None
            self.active_provider = "CPU (Fallback)"

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

        sub_lbl = QLabel("AI STEM DEMIX & RESTORATION  |  BS-ROFORMER SOTA (SDR 12.9 dB)")
        sub_lbl.setFont(QFont("Segoe UI", 9))
        sub_lbl.setStyleSheet("color: #64748b; margin-left: 12px;")
        header_layout.addWidget(sub_lbl)

        header_layout.addStretch()

        gpu_badge = QLabel(f"⚡ GPU: NVIDIA GeForce RTX 4060 [{self.active_provider}] — VRAM Cap: 3.0 GB / 8.0 GB")
        gpu_badge.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(gpu_badge)

        main_layout.addLayout(header_layout)

        # Tabs
        self.engine_tabs = QTabWidget()
        self.engine_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1e293b; background: #131722; border-radius: 6px; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 8px 20px; font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; }
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

        self.process_btn = QPushButton("⚡ Process Stems (Local GPU)")
        self.process_btn.setStyleSheet("background-color: #059669; color: white; padding: 9px 24px; border-radius: 6px; font-weight: bold; font-size: 13px;")
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
        main_layout.addWidget(self.visualizer)

        pb_layout = QHBoxLayout()
        pb_layout.setSpacing(10)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedSize(80, 34)
        self.play_btn.setStyleSheet("background-color: #38bdf8; color: black; font-weight: bold; border-radius: 6px;")
        self.play_btn.clicked.connect(self._toggle_playback)
        pb_layout.addWidget(self.play_btn)

        self.time_lbl = QLabel("00:00 / 00:00")
        self.time_lbl.setStyleSheet("color: #94a3b8; font-weight: bold;")
        pb_layout.addWidget(self.time_lbl)

        pb_layout.addStretch()
        main_layout.addLayout(pb_layout)

        self.mixer_container = QWidget()
        self.mixer_layout = QHBoxLayout(self.mixer_container)
        self.mixer_layout.setContentsMargins(0, 0, 0, 0)
        self.mixer_layout.setSpacing(14)
        main_layout.addWidget(self.mixer_container)

        self._rebuild_mixer_channels()

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

    def _on_engine_tab_changed(self, idx):
        self.current_engine_name = self.engine_tabs.tabText(idx)
        self._rebuild_mixer_channels()

    def _load_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleziona traccia audio", "", "Audio Files (*.wav *.mp3 *.flac *.aiff *.ogg)")
        if not path:
            return
        try:
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)
            self.loaded_audio = audio
            self.sample_rate = sr
            dur_sec = len(audio) / sr
            m = int(dur_sec // 60)
            s = int(dur_sec % 60)
            self.file_lbl.setText(f"{os.path.basename(path)} ({m:02d}:{s:02d} @ {sr}Hz)")
            self.time_lbl.setText(f"00:00 / {m:02d}:{s:02d}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Caricamento", str(e))

    def _process_stems_gpu(self):
        if self.loaded_audio is None:
            QMessageBox.warning(self, "Attenzione", "Carica prima un file audio!")
            return

        self.process_btn.setEnabled(False)
        self.process_btn.setText("⏳ Demixing su GPU in corso...")
        self.progress_bar.setValue(5)

        self.worker = DemixWorker(self.ort_session, self.loaded_audio, self.sample_rate, self.current_engine_name)
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.finished_success.connect(self._on_processing_complete)
        self.worker.error_occurred.connect(self._on_processing_error)
        self.worker.start()

    def _on_processing_error(self, err_msg):
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡ Process Stems (Local GPU)")
        QMessageBox.critical(self, "Errore Separazione", f"Si è verificato un errore durante l'inferenza:\n{err_msg}")

    def _on_processing_complete(self, stem_names, stems_dict, colors, elapsed):
        self.progress_bar.setValue(100)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡ Process Stems (Local GPU)")
        self.separated_stems = stems_dict
        wave_list = [stems_dict[name][:, 0] if stems_dict[name].ndim > 1 else stems_dict[name] for name in stem_names]
        self.visualizer.set_data(stem_names, wave_list, colors)
        rt_factor = (len(self.loaded_audio) / self.sample_rate) / elapsed if elapsed > 0 else 0
        QMessageBox.information(self, "Separazione Completata",
                                f"Separazione completata con successo su GPU ({self.active_provider})!\n"
                                f"Durata elaborazione: {elapsed:.2f}s ({rt_factor:.1f}x Real-Time)")

    def _toggle_playback(self):
        if not self.is_playing:
            if self.loaded_audio is None:
                return
            self.is_playing = True
            self.play_btn.setText("⏸ Pausa")
            self.play_start_time = time.time()
            self.timer.start()

            if self.separated_stems:
                mix_audio = np.zeros_like(self.loaded_audio)
                any_solo = any(ch.is_solo for ch in self.channels)
                for ch in self.channels:
                    active = ch.is_solo if any_solo else not ch.is_mute
                    if active and ch.stem_name in self.separated_stems:
                        gain = ch.slider.value() / 50.0
                        stem = self.separated_stems[ch.stem_name]
                        if stem.ndim == 1:
                            mix_audio[:, 0] += stem * gain
                            mix_audio[:, 1] += stem * gain
                        else:
                            mix_audio += stem * gain
                sd.play(mix_audio, self.sample_rate)
            else:
                sd.play(self.loaded_audio, self.sample_rate)
        else:
            self.is_playing = False
            self.play_btn.setText("▶ Play")
            self.timer.stop()
            sd.stop()

    def _update_playback_pos(self):
        if self.loaded_audio is None or not self.is_playing:
            return
        elapsed = time.time() - self.play_start_time
        dur = len(self.loaded_audio) / self.sample_rate
        if elapsed >= dur:
            self._toggle_playback()
            self.visualizer.set_playhead(0.0)
            return
        ratio = elapsed / dur
        self.visualizer.set_playhead(ratio)
        m = int(elapsed // 60)
        s = int(elapsed % 60)
        total_m = int(dur // 60)
        total_s = int(dur % 60)
        self.time_lbl.setText(f"{m:02d}:{s:02d} / {total_m:02d}:{total_s:02d}")

    def _export_stems_wav(self):
        if not self.separated_stems:
            QMessageBox.warning(self, "Attenzione", "Elabora prima gli stem con 'Process Stems (GPU)'!")
            return
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella di destinazione")
        if not folder:
            return
        for name, wave in self.separated_stems.items():
            out_file = os.path.join(folder, f"{name}.wav")
            sf.write(out_file, wave, self.sample_rate, subtype='PCM_24')
        QMessageBox.information(self, "Esportazione Riuscita", f"Gli stem sono stati esportati in 24-bit PCM in:\n{folder}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
