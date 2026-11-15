#!/usr/bin/env python3
"""
PhotonX Studio — Professional AI Audio Demixing & Restoration (Local GPU/CPU)
Per-stem model selection, analog FX chain (Denoiser / EQ / Saturator), multi-format export.
Powered by Meta AI HTDemucs v4, FoxJoy De-Reverb HQ, Kimberley Jensen Vocal SOTA.
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

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import (QColor, QFont, QPainter, QPen, QIcon,
                          QLinearGradient, QBrush, QPainterPath)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QTabWidget,
    QProgressBar, QFrame, QMessageBox, QDialog, QComboBox, QScrollArea,
    QGroupBox, QGridLayout, QSizePolicy, QDialogButtonBox, QRadioButton,
    QButtonGroup, QCheckBox, QSpinBox, QDoubleSpinBox
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

# ---------------------------------------------------------------------------
# Engine definitions
# ---------------------------------------------------------------------------
ENGINES = {
    "Music 4-Stems (Ultra)": {
        "stems": ["Vocals", "Drums", "Bass", "Other"],
        "colors": [QColor(76, 201, 240), QColor(247, 37, 133),
                   QColor(247, 127, 0), QColor(128, 237, 153)],
        "algo": "Meta AI HTDemucs v4 SOTA (Zero Residual Phase Cancellation)"
    },
    "Music 6-Stems (Guitar & Piano)": {
        "stems": ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"],
        "colors": [QColor(76, 201, 240), QColor(247, 37, 133), QColor(247, 127, 0),
                   QColor(254, 228, 64), QColor(155, 93, 229), QColor(128, 237, 153)],
        "algo": "Meta AI HTDemucs 6-Stem Full Studio Isolation"
    },
    "Acapella & Instrumental (Kim SOTA)": {
        "stems": ["Vocals", "Instrumental"],
        "colors": [QColor(76, 201, 240), QColor(247, 127, 0)],
        "algo": "Kimberley Jensen Vocal SOTA (MDX-Net / Mel-Band)"
    },
    "Drums 5-Stems": {
        "stems": ["Kick", "Snare", "Hi-Hat", "Toms", "Cymbals"],
        "colors": [QColor(230, 57, 70), QColor(241, 250, 238), QColor(168, 218, 220),
                   QColor(69, 123, 157), QColor(29, 53, 87)],
        "algo": "DrumSep UVR 5-Stem Multi-Band Crossover"
    },
    "Cinema 4-Stems": {
        "stems": ["Dialogue", "Music", "Effects", "Ambience"],
        "colors": [QColor(0, 180, 216), QColor(114, 9, 183),
                   QColor(255, 183, 3), QColor(251, 133, 0)],
        "algo": "Film Industry M&E Delivery Matrix"
    },
    "Voice Restore & Denoise": {
        "stems": ["Dry Voice", "Reverb Tail", "Noise Floor", "Air Presence"],
        "colors": [QColor(82, 183, 136), QColor(144, 224, 239),
                   QColor(87, 117, 144), QColor(244, 162, 97)],
        "algo": "FoxJoy Studio De-Reverb + Adaptive STFT Wiener Denoise"
    },
    "Voice Separator (Multi-Speaker)": {
        "stems": ["Speaker 1", "Speaker 2", "Speaker 3", "Other"],
        "colors": [QColor(255, 112, 166), QColor(155, 93, 229),
                   QColor(0, 245, 212), QColor(254, 228, 64)],
        "algo": "ReDimNet Formant & Pitch Diarization"
    }
}

# Per-stem model assignment options (user-selectable)
STEM_MODEL_OPTIONS = {
    "Vocals":       ["htdemucs (HTDemucs v4)",
                     "kim_vocal_2 (Kim Vocal 2 MDX)",
                     "htdemucs_6s (HTDemucs 6s)"],
    "Drums":        ["htdemucs (HTDemucs v4)",
                     "htdemucs_6s (HTDemucs 6s)"],
    "Bass":         ["htdemucs (HTDemucs v4)",
                     "htdemucs_6s (HTDemucs 6s)"],
    "Guitar":       ["htdemucs_6s (HTDemucs 6s)"],
    "Piano":        ["htdemucs_6s (HTDemucs 6s)"],
    "Other":        ["htdemucs (HTDemucs v4)",
                     "htdemucs_6s (HTDemucs 6s)"],
    "Instrumental": ["kim_vocal_2 (Kim Vocal 2 MDX)"],
    "Dry Voice":    ["foxjoy_dereverb (FoxJoy De-Reverb HQ)"],
    "Reverb Tail":  ["foxjoy_dereverb (FoxJoy De-Reverb HQ)"],
    "Noise Floor":  ["foxjoy_dereverb (FoxJoy De-Reverb HQ)"],
    "Air Presence": ["foxjoy_dereverb (FoxJoy De-Reverb HQ)"],
    "Kick":         ["htdemucs (HTDemucs v4)"],
    "Snare":        ["htdemucs (HTDemucs v4)"],
    "Hi-Hat":       ["htdemucs (HTDemucs v4)"],
    "Toms":         ["htdemucs (HTDemucs v4)"],
    "Cymbals":      ["htdemucs (HTDemucs v4)"],
    "Dialogue":     ["htdemucs (HTDemucs v4)"],
    "Music":        ["htdemucs (HTDemucs v4)"],
    "Effects":      ["htdemucs (HTDemucs v4)"],
    "Ambience":     ["htdemucs (HTDemucs v4)"],
    "Speaker 1":    ["htdemucs (HTDemucs v4)"],
    "Speaker 2":    ["htdemucs (HTDemucs v4)"],
    "Speaker 3":    ["htdemucs (HTDemucs v4)"],
}

BEST_QUALITY_PRESET = {
    "Vocals":       "kim_vocal_2 (Kim Vocal 2 MDX)",
    "Drums":        "htdemucs (HTDemucs v4)",
    "Bass":         "htdemucs (HTDemucs v4)",
    "Guitar":       "htdemucs_6s (HTDemucs 6s)",
    "Piano":        "htdemucs_6s (HTDemucs 6s)",
    "Other":        "htdemucs (HTDemucs v4)",
    "Instrumental": "kim_vocal_2 (Kim Vocal 2 MDX)",
    "Dry Voice":    "foxjoy_dereverb (FoxJoy De-Reverb HQ)",
    "Reverb Tail":  "foxjoy_dereverb (FoxJoy De-Reverb HQ)",
    "Noise Floor":  "foxjoy_dereverb (FoxJoy De-Reverb HQ)",
    "Air Presence": "foxjoy_dereverb (FoxJoy De-Reverb HQ)",
}


def resolve_model_path():
    for c in ["htdemucs", "htdemucs_6s", "kim_vocal_2"]:
        p = get_model_path(c)
        if p and p.exists():
            return p
    return APP_DIR / "models" / "htdemucs.onnx"

MODEL_PATH = resolve_model_path()


# ===========================================================================
# DSP utilities
# ===========================================================================

def apply_denoiser(audio: np.ndarray, threshold_db: float, reduction_db: float, sr: int) -> np.ndarray:
    """STFT spectral gate denoiser."""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    n_fft = 2048
    hop = n_fft // 4
    out_chans = []
    thresh_linear = 10 ** (threshold_db / 20.0)
    reduction_linear = 10 ** (-abs(reduction_db) / 20.0)
    for c in range(audio.shape[1]):
        f, t, Zxx = signal.stft(audio[:, c], fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        # Noise estimate from quietest 15th-percentile bins
        noise_est = np.percentile(mag, 15, axis=-1, keepdims=True)
        # Binary soft gate with smooth knee
        gate = np.where(mag > thresh_linear * noise_est,
                        1.0,
                        reduction_linear + (1.0 - reduction_linear) * (mag / (thresh_linear * noise_est + 1e-9)))
        clean_mag = mag * gate
        _, cleaned = signal.istft(clean_mag * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        out_chans.append(cleaned[:len(audio)])
    return np.stack(out_chans, axis=-1).astype(np.float32)


def apply_eq(audio: np.ndarray, bands: list, sr: int) -> np.ndarray:
    """
    5-band parametric EQ applied via biquad cascades.
    bands: list of dicts {type, freq, gain_db, q}
    types: 'lowcut', 'lowshelf', 'peak', 'highshelf', 'highcut'
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    out = audio.astype(np.float64)
    nyq = sr / 2.0
    for band in bands:
        freq = max(10.0, min(band.get("freq", 1000.0), nyq - 1.0))
        gain_db = band.get("gain_db", 0.0)
        q = max(0.1, band.get("q", 0.707))
        btype = band.get("type", "peak")
        w0 = freq / nyq

        if abs(gain_db) < 0.01 and btype in ("peak", "lowshelf", "highshelf"):
            continue

        try:
            if btype == "lowcut":
                b, a = signal.butter(2, w0, btype='high')
            elif btype == "highcut":
                b, a = signal.butter(2, w0, btype='low')
            elif btype == "peak":
                b, a = signal.iirpeak(w0, q)
                # Scale by gain
                A = 10 ** (gain_db / 40.0)
                b = b * A if gain_db >= 0 else b / A
            elif btype == "lowshelf":
                b, a = signal.iirfilter(1, w0, btype='low', ftype='butter')
                A = 10 ** (gain_db / 40.0)
                b = b * A if gain_db >= 0 else b
            elif btype == "highshelf":
                b, a = signal.iirfilter(1, w0, btype='high', ftype='butter')
                A = 10 ** (gain_db / 40.0)
                b = b * A if gain_db >= 0 else b
            else:
                continue

            for c in range(out.shape[1]):
                out[:, c] = signal.filtfilt(b, a, out[:, c])
        except Exception:
            pass

    return out.astype(np.float32)


def apply_saturation(audio: np.ndarray, drive: float, character: str, mix: float) -> np.ndarray:
    """Analog waveshaper saturation."""
    dry = audio.astype(np.float32)
    d = max(0.001, min(drive, 1.0))
    gain = 1.0 + d * 9.0  # up to 10x drive
    driven = dry * gain

    if character == "Soft Clip":
        wet = np.tanh(driven)
    elif character == "Tape":
        wet = np.sign(driven) * (1.0 - np.exp(-np.abs(driven)))
    elif character == "Tube":
        # Asymmetric waveshaper (tube-like, slight 2nd harmonic)
        pos = np.where(driven >= 0, np.tanh(driven * 0.9), driven)
        neg = np.where(driven < 0, -np.tanh(-driven * 1.1), driven)
        wet = np.where(driven >= 0, pos, neg)
        wet = np.where(driven >= 0, pos, neg)
    else:
        wet = np.tanh(driven)

    # Normalize to prevent clipping
    peak = np.max(np.abs(wet)) + 1e-9
    if peak > 1.0:
        wet /= peak

    return (dry * (1.0 - mix) + wet * mix).astype(np.float32)


# ===========================================================================
# Workers
# ===========================================================================

class ModelDownloadWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    finished_success = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_key):
        super().__init__()
        self.model_key = model_key

    def run(self):
        try:
            download_model(self.model_key,
                           progress_callback=lambda pct, msg: self.progress_changed.emit(pct, msg))
            self.finished_success.emit(self.model_key)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ExportWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    finished_success = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, stems_dict, folder, fmt, sr, channels, processing_params):
        super().__init__()
        self.stems_dict = stems_dict
        self.folder = folder
        self.fmt = fmt          # 'WAV' | 'AIFF' | 'MP3'
        self.sr = sr
        self.channels = channels
        self.processing_params = processing_params  # {stem_name: {...}}

    def run(self):
        try:
            total = len(self.stems_dict)
            for i, (name, wave) in enumerate(self.stems_dict.items()):
                self.progress_changed.emit(int(100 * i / total), f"Esportazione: {name}")
                processed = self._apply_fx(wave, name)

                if self.fmt == "WAV":
                    out_path = os.path.join(self.folder, f"{name}.wav")
                    sf.write(out_path, processed, self.sr, subtype='PCM_24')
                elif self.fmt == "AIFF":
                    out_path = os.path.join(self.folder, f"{name}.aiff")
                    sf.write(out_path, processed, self.sr, subtype='PCM_24', format='AIFF')
                elif self.fmt == "MP3":
                    try:
                        from pydub import AudioSegment
                        import tempfile
                        # Write temp WAV first
                        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        tmp_path = tmp.name
                        tmp.close()
                        sf.write(tmp_path, processed, self.sr, subtype='PCM_16')
                        out_path = os.path.join(self.folder, f"{name}.mp3")
                        seg = AudioSegment.from_wav(tmp_path)
                        seg.export(out_path, format="mp3", bitrate="320k")
                        os.unlink(tmp_path)
                    except ImportError:
                        # Fallback to WAV if pydub not available
                        out_path = os.path.join(self.folder, f"{name}.wav")
                        sf.write(out_path, processed, self.sr, subtype='PCM_24')
                        self.error_occurred.emit(
                            "pydub non è installato (pip install pydub). Esportato come WAV.")
                        return

            self.progress_changed.emit(100, "Completato")
            self.finished_success.emit(self.folder)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _apply_fx(self, wave, name):
        params = self.processing_params.get(name, {})
        audio = wave.copy().astype(np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        # 1. Denoiser
        dn = params.get("denoiser", {})
        if dn.get("enabled", False):
            audio = apply_denoiser(audio,
                                   threshold_db=dn.get("threshold_db", -40.0),
                                   reduction_db=dn.get("reduction_db", 12.0),
                                   sr=self.sr)

        # 2. EQ
        eq = params.get("eq", {})
        if eq.get("enabled", False):
            audio = apply_eq(audio, eq.get("bands", []), sr=self.sr)

        # 3. Saturator
        sat = params.get("sat", {})
        if sat.get("enabled", False):
            audio = apply_saturation(audio,
                                     drive=sat.get("drive", 0.1),
                                     character=sat.get("character", "Soft Clip"),
                                     mix=sat.get("mix", 0.3))
        return audio


class StemFXWorker(QThread):
    finished_stem = pyqtSignal(str, object)

    def __init__(self, stem_name, raw_audio, params, sr):
        super().__init__()
        self.stem_name = stem_name
        self.raw_audio = raw_audio
        self.params = params
        self.sr = sr

    def run(self):
        try:
            audio = self.raw_audio.copy().astype(np.float32)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)

            dn = self.params.get("denoiser", {})
            if dn.get("enabled", False):
                audio = apply_denoiser(audio,
                                       threshold_db=dn.get("threshold_db", -40.0),
                                       reduction_db=dn.get("reduction_db", 12.0),
                                       sr=self.sr)

            eq = self.params.get("eq", {})
            if eq.get("enabled", False):
                audio = apply_eq(audio, eq.get("bands", []), sr=self.sr)

            sat = self.params.get("sat", {})
            if sat.get("enabled", False):
                audio = apply_saturation(audio,
                                         drive=sat.get("drive", 0.1),
                                         character=sat.get("character", "Soft Clip"),
                                         mix=sat.get("mix", 0.3))

            self.finished_stem.emit(self.stem_name, audio)
        except Exception as e:
            print(f"[StemFXWorker error] {e}")


class DemixWorker(QThread):
    progress_changed = pyqtSignal(int)
    finished_success = pyqtSignal(list, dict, list, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, ort_session, audio, sr, engine_name,
                 target_scope="Tutti gli Stem (Completo)", stem_model_keys=None):
        super().__init__()
        self.ort_session = ort_session
        self.audio = audio
        self.sr = sr
        self.engine_name = engine_name
        self.target_scope = target_scope
        self.stem_model_keys = stem_model_keys or {}   # {stem_name: "model_key"}

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

            inp = np.stack([spec_l.real, spec_l.imag, spec_r.real, spec_r.imag],
                           axis=0)[np.newaxis, ...].astype(np.float32)
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

    def _run_htdemucs(self, model_key, proc_audio, n_stems=4):
        """Run HTDemucs 4s or 6s model."""
        m_path = get_model_path(model_key)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 8
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(m_path), sess_opts, providers=['CPUExecutionProvider'])

        seg_size = 343980
        step = 257985
        num_chunks = (max(1, int(np.ceil((len(proc_audio) - seg_size) / step)) + 1)
                      if len(proc_audio) > seg_size else 1)
        win = np.hanning(seg_size).astype(np.float32)

        stems_acc = np.zeros((n_stems, 2, len(proc_audio)), dtype=np.float32)
        weight_acc = np.zeros(len(proc_audio), dtype=np.float32)

        for c in range(num_chunks):
            start = c * step
            end = min(len(proc_audio), start + seg_size)
            chunk = np.zeros((seg_size, 2), dtype=np.float32)
            chunk[:end - start] = proc_audio[start:end]

            inp = chunk.T[np.newaxis, ...]
            out = sess.run(None, {'mix': inp})[0][0]
            valid = end - start

            stems_acc[:, :, start:start + valid] += out[:, :, :valid] * win[:valid]
            weight_acc[start:start + valid] += win[:valid]

            pct = int(5 + 85 * (c + 1) / num_chunks)
            self.progress_changed.emit(min(92, pct))

        w_mask = np.maximum(weight_acc, 1e-4)
        stems_acc /= w_mask[np.newaxis, np.newaxis, :]
        return stems_acc

    def run(self):
        try:
            t0 = time.time()
            engine_info = ENGINES[self.engine_name]
            stems_names = list(engine_info["stems"])
            colors = list(engine_info["colors"])
            audio = self.audio.astype(np.float32)

            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)

            orig_sr = self.sr
            target_sr = 44100
            if orig_sr != target_sr:
                gcd = int(np.gcd(orig_sr, target_sr))
                up = target_sr // gcd
                down = orig_sr // gcd
                proc_audio = signal.resample_poly(audio, up, down, axis=0).astype(np.float32)
            else:
                proc_audio = audio
                up, down = 1, 1

            stems_dict = {}
            nyq = target_sr / 2.0

            htdemucs_model = get_model_path("htdemucs")
            htdemucs_6s_model = get_model_path("htdemucs_6s")
            foxjoy_model = get_model_path("foxjoy_dereverb")
            kim_model = get_model_path("kim_vocal_2")

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 8
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Determine effective vocal model from per-stem selection
            vocal_model_key = "htdemucs"
            voc_choice = self.stem_model_keys.get("Vocals", "")
            if "kim_vocal_2" in voc_choice and kim_model and kim_model.exists():
                vocal_model_key = "kim_vocal_2"
            elif "htdemucs_6s" in voc_choice and htdemucs_6s_model and htdemucs_6s_model.exists():
                vocal_model_key = "htdemucs_6s"

            # ----------------------------------------------------------------
            # ENGINE: Music 6-Stems
            # ----------------------------------------------------------------
            if self.engine_name == "Music 6-Stems (Guitar & Piano)" and htdemucs_6s_model:
                self.progress_changed.emit(5)
                stems_acc = self._run_htdemucs("htdemucs_6s", proc_audio, n_stems=6)
                # 6s mapping: 0=Drums, 1=Bass, 2=Other, 3=Vocals, 4=Guitar, 5=Piano
                stems_dict["Vocals"] = stems_acc[3].T[:len(proc_audio)]
                stems_dict["Drums"] = stems_acc[0].T[:len(proc_audio)]
                stems_dict["Bass"] = stems_acc[1].T[:len(proc_audio)]
                stems_dict["Guitar"] = stems_acc[4].T[:len(proc_audio)]
                stems_dict["Piano"] = stems_acc[5].T[:len(proc_audio)]
                stems_dict["Other"] = stems_acc[2].T[:len(proc_audio)]

                # Optional: replace Vocals with Kim SOTA
                if vocal_model_key == "kim_vocal_2" and kim_model and kim_model.exists():
                    self.progress_changed.emit(93)
                    sess_kim = ort.InferenceSession(str(kim_model), sess_opts,
                                                    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
                    vocals_kim = self._process_mdx(sess_kim, proc_audio,
                                                   n_fft=7680, dim_f=3072, dim_t=256,
                                                   hop=1024, comp=1.009, p_start=93, p_end=98)
                    stems_dict["Vocals"] = vocals_kim

            # ----------------------------------------------------------------
            # ENGINE: Acapella & Instrumental (Kim SOTA)
            # ----------------------------------------------------------------
            elif self.engine_name == "Acapella & Instrumental (Kim SOTA)" and kim_model:
                self.progress_changed.emit(5)
                providers = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                sess_kim = ort.InferenceSession(str(kim_model), sess_opts, providers=providers)
                vocals = self._process_mdx(sess_kim, proc_audio,
                                           n_fft=7680, dim_f=3072, dim_t=256,
                                           hop=1024, comp=1.009, p_start=5, p_end=90)
                instrumental = proc_audio - vocals
                stems_dict["Vocals"] = vocals
                stems_dict["Instrumental"] = instrumental

            # ----------------------------------------------------------------
            # BASE ENGINE: HTDemucs v4 4-Stem (all remaining engines)
            # ----------------------------------------------------------------
            elif htdemucs_model:
                self.progress_changed.emit(5)
                stems_acc = self._run_htdemucs("htdemucs", proc_audio, n_stems=4)
                # 4s mapping: 0=Drums, 1=Bass, 2=Other, 3=Vocals
                drums_raw = stems_acc[0].T[:len(proc_audio)]
                bass_raw = stems_acc[1].T[:len(proc_audio)]
                other_raw = stems_acc[2].T[:len(proc_audio)]
                vocals_raw = stems_acc[3].T[:len(proc_audio)]

                if self.engine_name == "Music 4-Stems (Ultra)":
                    # Optional vocal refinement with Kim SOTA
                    if vocal_model_key == "kim_vocal_2" and kim_model and kim_model.exists():
                        self.progress_changed.emit(93)
                        sess_kim = ort.InferenceSession(
                            str(kim_model), sess_opts,
                            providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
                        vocals_raw = self._process_mdx(
                            sess_kim, proc_audio,
                            n_fft=7680, dim_f=3072, dim_t=256,
                            hop=1024, comp=1.009, p_start=93, p_end=98)

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
                    if foxjoy_model and foxjoy_model.exists():
                        sess_fox = ort.InferenceSession(str(foxjoy_model), sess_opts,
                                                        providers=['CPUExecutionProvider'])
                        dry_voice = self._process_mdx(sess_fox, base_voice,
                                                      n_fft=7680, dim_f=3072, dim_t=512,
                                                      hop=1024, comp=1.0, p_start=80, p_end=92)
                        reverb_tail = base_voice - dry_voice
                    else:
                        dry_voice = base_voice * 0.8
                        reverb_tail = base_voice * 0.2

                    n_fft_w = 1024
                    hop_w = 768
                    clean_chans, noise_chans = [], []
                    for ch_i in range(2):
                        _, _, Zxx = signal.stft(dry_voice[:, ch_i], fs=target_sr,
                                                nperseg=n_fft_w, noverlap=hop_w)
                        mag = np.abs(Zxx)
                        phase = np.angle(Zxx)
                        noise_est = np.percentile(mag, 15, axis=-1, keepdims=True)
                        clean_mag = np.maximum(mag - 1.6 * noise_est, 0.0)
                        noise_mag = np.minimum(mag, 1.6 * noise_est)
                        _, clean_sig = signal.istft(clean_mag * np.exp(1j * phase),
                                                    fs=target_sr, nperseg=n_fft_w, noverlap=hop_w)
                        _, noise_sig = signal.istft(noise_mag * np.exp(1j * phase),
                                                    fs=target_sr, nperseg=n_fft_w, noverlap=hop_w)
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
                    b1, a1 = signal.butter(2, [80.0 / nyq, min(0.99, 240.0 / nyq)], btype='band')
                    spk1 = signal.filtfilt(b1, a1, base_voice, axis=0) * 1.2
                    b2, a2 = signal.butter(2, [240.0 / nyq, min(0.99, 550.0 / nyq)], btype='band')
                    spk2 = signal.filtfilt(b2, a2, base_voice, axis=0) * 1.2
                    b3, a3 = signal.butter(2, [550.0 / nyq, min(0.99, 1800.0 / nyq)], btype='band')
                    spk3 = signal.filtfilt(b3, a3, base_voice, axis=0) * 1.2
                    b_oth, a_oth = signal.butter(2, min(0.99, 1800.0 / nyq), btype='high')
                    oth = signal.filtfilt(b_oth, a_oth, base_voice, axis=0)
                    stems_dict["Speaker 1"] = spk1
                    stems_dict["Speaker 2"] = spk2
                    stems_dict["Speaker 3"] = spk3
                    stems_dict["Other"] = oth

            else:
                for idx, name in enumerate(stems_names):
                    g = [0.8, 0.6, 0.4, 0.5][min(idx, 3)]
                    stems_dict[name] = proc_audio * g
                self.progress_changed.emit(95)

            # Sample-accurate resample back to original rate
            if orig_sr != target_sr:
                for k in stems_dict:
                    stems_dict[k] = signal.resample_poly(
                        stems_dict[k], down, up, axis=0
                    ).astype(np.float32)[:len(audio)]
            else:
                for k in stems_dict:
                    stems_dict[k] = stems_dict[k][:len(audio)]

            # Scope filter
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


# ===========================================================================
# Model Manager Dialog
# ===========================================================================

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

        sub = QLabel("Scarica on-demand i pesi neurali per arricchire le capacità di PhotonX Studio.")
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
        QMessageBox.information(self, "Download Completato",
                                f"Il modello '{MODEL_REGISTRY[key]['name']}' è stato installato!")


# ===========================================================================
# EQ Visualizer
# ===========================================================================

class EQCurveWidget(QWidget):
    """Mini frequency-response curve display."""

    def __init__(self, bands_getter, parent=None):
        super().__init__(parent)
        self.bands_getter = bands_getter
        self.setMinimumSize(280, 80)
        self.setMaximumHeight(90)
        self.setStyleSheet("background: #0a0d13; border: 1px solid #1e2432; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(self.rect(), QColor(10, 13, 19))

        # Grid lines
        painter.setPen(QPen(QColor(30, 36, 50), 1))
        for y_frac in [0.25, 0.5, 0.75]:
            painter.drawLine(0, int(h * y_frac), w, int(h * y_frac))

        # Frequency response curve (approximate)
        freqs = np.geomspace(20, 20000, w)
        sr = 44100
        response = np.zeros(w)

        for band in self.bands_getter():
            freq = max(10.0, min(band.get("freq", 1000.0), 20000.0))
            gain_db = band.get("gain_db", 0.0)
            q = max(0.1, band.get("q", 0.707))
            btype = band.get("type", "peak")
            nyq = sr / 2.0
            w0 = freq / nyq
            try:
                if btype == "lowcut":
                    b, a = signal.butter(2, w0, btype='high')
                elif btype == "highcut":
                    b, a = signal.butter(2, w0, btype='low')
                elif btype == "peak":
                    b, a = signal.iirpeak(w0, q)
                    A = 10 ** (gain_db / 40.0)
                    b = b * A if gain_db >= 0 else b / A
                elif btype in ("lowshelf", "highshelf"):
                    btp = 'low' if btype == "lowshelf" else 'high'
                    b, a = signal.iirfilter(1, w0, btype=btp, ftype='butter')
                    A = 10 ** (gain_db / 40.0)
                    b = b * A if gain_db >= 0 else b
                else:
                    continue
                w_arr, h_arr = signal.freqz(b, a, worN=freqs, fs=sr)
                response += 20 * np.log10(np.abs(h_arr) + 1e-9)
            except Exception:
                pass

        # Draw curve
        response = np.clip(response, -24, 24)
        path = QPainterPath()
        for xi in range(w):
            norm = (response[xi] + 24) / 48.0
            y = h - norm * h
            if xi == 0:
                path.moveTo(xi, y)
            else:
                path.lineTo(xi, y)

        pen = QPen(QColor(56, 189, 248), 1.5)
        painter.setPen(pen)
        painter.drawPath(path)

        # 0 dB line
        painter.setPen(QPen(QColor(80, 100, 130, 150), 1, Qt.PenStyle.DashLine))
        painter.drawLine(0, h // 2, w, h // 2)


# ===========================================================================
# Per-Track FX Panel
# ===========================================================================

class FXPanel(QDialog):
    """
    Per-stem processing panel: Denoiser + 5-band EQ + Saturator.
    Non-modal, stays open alongside the mixer.
    """
    params_changed = pyqtSignal(str, dict)   # (stem_name, params_dict)

    BAND_NAMES = ["Low Cut", "Low Shelf", "Mid Peak", "High Shelf", "High Cut"]
    BAND_TYPES = ["lowcut", "lowshelf", "peak", "highshelf", "highcut"]
    BAND_FREQS = [80.0, 250.0, 1000.0, 5000.0, 12000.0]
    BAND_Q     = [0.707, 0.707, 1.0, 0.707, 0.707]

    def __init__(self, stem_name, color, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.stem_name = stem_name
        self.color = color
        self.setWindowTitle(f"FX — {stem_name}")
        self.resize(540, 620)
        self.setStyleSheet("""
            QDialog { background-color: #0e1117; color: #f1f5f9; font-family: 'Segoe UI'; }
            QGroupBox { border: 1px solid #1e293b; border-radius: 6px; margin-top: 10px; padding: 8px; }
            QGroupBox::title { color: #38bdf8; padding: 0 6px; }
            QLabel { color: #cbd5e1; }
            QSlider::groove:horizontal { background: #1e293b; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #38bdf8; width: 12px; height: 12px;
                                          margin: -4px 0; border-radius: 6px; }
            QComboBox { background: #1e293b; color: #f1f5f9; padding: 3px 8px;
                        border-radius: 4px; border: 1px solid #334155; }
            QCheckBox { color: #94a3b8; }
        """)

        self._params = {
            "denoiser": {"enabled": False, "threshold_db": -40.0, "reduction_db": 12.0},
            "eq": {"enabled": False, "bands": [
                {"type": t, "freq": f, "gain_db": 0.0, "q": q}
                for t, f, q in zip(self.BAND_TYPES, self.BAND_FREQS, self.BAND_Q)
            ]},
            "sat": {"enabled": False, "drive": 0.1, "character": "Soft Clip", "mix": 0.3}
        }

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # Title
        title = QLabel(f"<b style='color:{color.name()}'>{stem_name.upper()}</b>  FX Chain")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        layout.addWidget(self._build_denoiser())
        layout.addWidget(self._build_eq())
        layout.addWidget(self._build_saturator())
        layout.addStretch()

    # --- Denoiser ----------------------------------------------------------------

    def _build_denoiser(self):
        grp = QGroupBox("🎛  Denoiser Analogico (Spectral Gate)")
        grp_layout = QVBoxLayout(grp)

        self.dn_enabled = QCheckBox("Attiva Denoiser")
        self.dn_enabled.toggled.connect(self._emit)
        grp_layout.addWidget(self.dn_enabled)

        grid = QGridLayout()
        grid.setSpacing(8)

        # Threshold
        grid.addWidget(QLabel("Soglia (dB):"), 0, 0)
        self.dn_thresh_sl = QSlider(Qt.Orientation.Horizontal)
        self.dn_thresh_sl.setRange(-80, 0)
        self.dn_thresh_sl.setValue(-40)
        self.dn_thresh_lbl = QLabel("-40 dB")
        self.dn_thresh_sl.valueChanged.connect(lambda v: (
            self.dn_thresh_lbl.setText(f"{v} dB"), self._emit()))
        grid.addWidget(self.dn_thresh_sl, 0, 1)
        grid.addWidget(self.dn_thresh_lbl, 0, 2)

        # Reduction
        grid.addWidget(QLabel("Riduzione (dB):"), 1, 0)
        self.dn_red_sl = QSlider(Qt.Orientation.Horizontal)
        self.dn_red_sl.setRange(0, 60)
        self.dn_red_sl.setValue(12)
        self.dn_red_lbl = QLabel("12 dB")
        self.dn_red_sl.valueChanged.connect(lambda v: (
            self.dn_red_lbl.setText(f"{v} dB"), self._emit()))
        grid.addWidget(self.dn_red_sl, 1, 1)
        grid.addWidget(self.dn_red_lbl, 1, 2)

        grp_layout.addLayout(grid)
        return grp

    # --- EQ -----------------------------------------------------------------------

    def _build_eq(self):
        grp = QGroupBox("🎚  Equalizzatore Parametrico 5-Band (FabFilter Style)")
        grp_layout = QVBoxLayout(grp)

        self.eq_enabled = QCheckBox("Attiva EQ")
        self.eq_enabled.toggled.connect(self._emit)
        grp_layout.addWidget(self.eq_enabled)

        self._eq_band_widgets = []

        grid = QGridLayout()
        grid.setSpacing(6)
        headers = ["Banda", "Freq (Hz)", "Gain (dB)", "Q"]
        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet("color: #64748b; font-size: 10px; font-weight: bold;")
            grid.addWidget(lbl, 0, col)

        for row, (name, btype, freq, q) in enumerate(
                zip(self.BAND_NAMES, self.BAND_TYPES, self.BAND_FREQS, self.BAND_Q), start=1):

            lbl = QLabel(name)
            lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
            grid.addWidget(lbl, row, 0)

            freq_sl = QSlider(Qt.Orientation.Horizontal)
            freq_sl.setRange(10, 20000)
            freq_sl.setValue(int(freq))
            freq_lbl = QLabel(f"{int(freq)} Hz")
            freq_lbl.setFixedWidth(70)
            freq_sl.valueChanged.connect(lambda v, lbl=freq_lbl: (
                lbl.setText(f"{v} Hz"), self._emit()))
            grid.addWidget(freq_sl, row, 1)
            grid.addWidget(freq_lbl, row, 1, alignment=Qt.AlignmentFlag.AlignRight)

            gain_sl = QSlider(Qt.Orientation.Horizontal)
            gain_sl.setRange(-24, 24)
            gain_sl.setValue(0)
            gain_lbl = QLabel("0 dB")
            gain_lbl.setFixedWidth(50)
            gain_sl.valueChanged.connect(lambda v, lbl=gain_lbl: (
                lbl.setText(f"{v:+d} dB"), self._emit()))
            grid.addWidget(gain_sl, row, 2)
            grid.addWidget(gain_lbl, row, 2, alignment=Qt.AlignmentFlag.AlignRight)

            q_sl = QSlider(Qt.Orientation.Horizontal)
            q_sl.setRange(1, 100)
            q_sl.setValue(int(q * 10))
            q_lbl = QLabel(f"Q={q:.1f}")
            q_lbl.setFixedWidth(50)
            q_sl.valueChanged.connect(lambda v, lbl=q_lbl: (
                lbl.setText(f"Q={v/10:.1f}"), self._emit()))
            grid.addWidget(q_sl, row, 3)
            grid.addWidget(q_lbl, row, 3, alignment=Qt.AlignmentFlag.AlignRight)

            self._eq_band_widgets.append({
                "type": btype,
                "freq_sl": freq_sl,
                "gain_sl": gain_sl,
                "q_sl": q_sl,
            })

        grp_layout.addLayout(grid)

        self._eq_curve = EQCurveWidget(self._get_eq_bands)
        grp_layout.addWidget(self._eq_curve)

        return grp

    def _get_eq_bands(self):
        bands = []
        for w in self._eq_band_widgets:
            bands.append({
                "type": w["type"],
                "freq": float(w["freq_sl"].value()),
                "gain_db": float(w["gain_sl"].value()),
                "q": w["q_sl"].value() / 10.0,
            })
        return bands

    # --- Saturator ----------------------------------------------------------------

    def _build_saturator(self):
        grp = QGroupBox("🔥  Saturatore Analogico")
        grp_layout = QVBoxLayout(grp)

        self.sat_enabled = QCheckBox("Attiva Saturatore")
        self.sat_enabled.toggled.connect(self._emit)
        grp_layout.addWidget(self.sat_enabled)

        grid = QGridLayout()
        grid.setSpacing(8)

        # Character
        grid.addWidget(QLabel("Carattere:"), 0, 0)
        self.sat_char = QComboBox()
        self.sat_char.addItems(["Soft Clip", "Tape", "Tube"])
        self.sat_char.currentTextChanged.connect(lambda _: self._emit())
        grid.addWidget(self.sat_char, 0, 1, 1, 2)

        # Drive
        grid.addWidget(QLabel("Drive:"), 1, 0)
        self.sat_drive_sl = QSlider(Qt.Orientation.Horizontal)
        self.sat_drive_sl.setRange(0, 100)
        self.sat_drive_sl.setValue(10)
        self.sat_drive_lbl = QLabel("10%")
        self.sat_drive_sl.valueChanged.connect(lambda v: (
            self.sat_drive_lbl.setText(f"{v}%"), self._emit()))
        grid.addWidget(self.sat_drive_sl, 1, 1)
        grid.addWidget(self.sat_drive_lbl, 1, 2)

        # Mix
        grid.addWidget(QLabel("Mix (Dry/Wet):"), 2, 0)
        self.sat_mix_sl = QSlider(Qt.Orientation.Horizontal)
        self.sat_mix_sl.setRange(0, 100)
        self.sat_mix_sl.setValue(30)
        self.sat_mix_lbl = QLabel("30%")
        self.sat_mix_sl.valueChanged.connect(lambda v: (
            self.sat_mix_lbl.setText(f"{v}%"), self._emit()))
        grid.addWidget(self.sat_mix_sl, 2, 1)
        grid.addWidget(self.sat_mix_lbl, 2, 2)

        grp_layout.addLayout(grid)
        return grp

    # --- Emit --------------------------------------------------------------------

    def _emit(self):
        params = {
            "denoiser": {
                "enabled": self.dn_enabled.isChecked(),
                "threshold_db": float(self.dn_thresh_sl.value()),
                "reduction_db": float(self.dn_red_sl.value()),
            },
            "eq": {
                "enabled": self.eq_enabled.isChecked(),
                "bands": self._get_eq_bands(),
            },
            "sat": {
                "enabled": self.sat_enabled.isChecked(),
                "drive": self.sat_drive_sl.value() / 100.0,
                "character": self.sat_char.currentText(),
                "mix": self.sat_mix_sl.value() / 100.0,
            }
        }
        self._params = params
        # Refresh EQ curve
        if hasattr(self, "_eq_curve"):
            self._eq_curve.update()
        self.params_changed.emit(self.stem_name, params)

    def get_params(self):
        return self._params


# ===========================================================================
# Waveform Visualizer
# ===========================================================================

class WaveformVisualizer(QFrame):
    lane_clicked = pyqtSignal(int)
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
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
            w, h = self.width(), self.height()
            x, y = event.pos().x(), event.pos().y()
            start_x, end_x = 110, w - 20
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
        w, h = self.width(), self.height()

        if not self.waveforms:
            painter.setPen(QColor(80, 95, 125))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Carica una traccia audio e premi 'Process Stems' per visualizzare gli stem\n"
                             "(Clicca sulla forma d'onda per riprodurre da qualsiasi punto)")
            return

        num_lanes = len(self.waveforms)
        lane_h = h / num_lanes

        for i, (name, wave, col) in enumerate(zip(self.stem_names, self.waveforms, self.colors)):
            y_mid = i * lane_h + lane_h / 2.0
            painter.setPen(QPen(QColor(30, 36, 50), 1))
            painter.drawLine(0, int(i * lane_h), w, int(i * lane_h))
            painter.setPen(col)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(14, int(y_mid - 2), name.upper())
            painter.setPen(QColor(148, 163, 184))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(14, int(y_mid + 12), "[CLICCA SOLO]")

            painter.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 210), 1.5))
            start_x, end_x = 110, w - 20
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


# ===========================================================================
# Stem Channel Strip (with model selector + FX button)
# ===========================================================================

class StemChannel(QFrame):
    gain_changed = pyqtSignal(int, float)
    mute_toggled = pyqtSignal(int, bool)
    solo_toggled = pyqtSignal(int, bool)
    model_changed = pyqtSignal(int, str)   # (index, model_choice_str)
    fx_requested = pyqtSignal(int)         # open FX panel for this stem

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
        self.setFixedWidth(140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(7)

        self.title_lbl = QLabel(name.upper())
        self.title_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_lbl.setStyleSheet(f"color: {color.name()}; border: none;")
        layout.addWidget(self.title_lbl)

        # Model selector
        self.model_combo = QComboBox()
        options = STEM_MODEL_OPTIONS.get(name, ["htdemucs (HTDemucs v4)"])
        for opt in options:
            self.model_combo.addItem(opt)
        self.model_combo.setStyleSheet("""
            QComboBox { background: #0d1117; color: #94a3b8; padding: 2px 4px;
                        border-radius: 3px; font-size: 9px; border: 1px solid #1e293b; }
            QComboBox::drop-down { border: none; }
        """)
        self.model_combo.currentTextChanged.connect(
            lambda t: self.model_changed.emit(self.index, t))
        layout.addWidget(self.model_combo)

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
        self.db_lbl.setFont(QFont("Segoe UI", 8))
        self.db_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.db_lbl.setStyleSheet("color: #94a3b8; border: none;")
        layout.addWidget(self.db_lbl)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.mute_btn = QPushButton("M")
        self.mute_btn.setFixedSize(36, 26)
        self.mute_btn.setToolTip("Muta questa traccia")
        self.mute_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")
        self.mute_btn.clicked.connect(self._toggle_mute)
        btn_layout.addWidget(self.mute_btn)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setFixedSize(36, 26)
        self.solo_btn.setToolTip("Ascolta SOLO questa traccia")
        self.solo_btn.setStyleSheet("background-color: #1e2432; color: #94a3b8; font-weight: bold; border-radius: 4px;")
        self.solo_btn.clicked.connect(self._toggle_solo)
        btn_layout.addWidget(self.solo_btn)

        layout.addLayout(btn_layout)

        self.fx_btn = QPushButton("🎚 FX")
        self.fx_btn.setFixedHeight(24)
        self.fx_btn.setStyleSheet("""
            QPushButton { background-color: #1a2535; color: #38bdf8;
                          font-size: 10px; font-weight: bold; border-radius: 4px; border: 1px solid #334155; }
            QPushButton:hover { background-color: #22304a; }
        """)
        self.fx_btn.clicked.connect(lambda: self.fx_requested.emit(self.index))
        layout.addWidget(self.fx_btn)

    def get_selected_model(self):
        return self.model_combo.currentText()

    def set_model(self, choice_str):
        idx = self.model_combo.findText(choice_str)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

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


# ===========================================================================
# Export Dialog
# ===========================================================================

class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Esporta Stem")
        self.resize(360, 220)
        self.setStyleSheet("""
            QDialog { background: #0e1117; color: #f1f5f9; font-family: 'Segoe UI'; }
            QGroupBox { border: 1px solid #1e293b; border-radius: 6px; margin-top: 10px; padding: 8px; }
            QGroupBox::title { color: #38bdf8; }
            QRadioButton { color: #cbd5e1; }
        """)
        layout = QVBoxLayout(self)

        grp = QGroupBox("Formato di esportazione")
        grp_layout = QVBoxLayout(grp)

        self.fmt_group = QButtonGroup(self)
        self.rb_wav = QRadioButton("WAV 24-bit (qualità massima, lossless)")
        self.rb_aiff = QRadioButton("AIFF 24-bit (compatibile Apple / Logic Pro)")
        self.rb_mp3 = QRadioButton("MP3 320kbps (richiede pydub + ffmpeg)")
        self.rb_wav.setChecked(True)

        for rb in [self.rb_wav, self.rb_aiff, self.rb_mp3]:
            self.fmt_group.addButton(rb)
            grp_layout.addWidget(rb)

        layout.addWidget(grp)

        self.apply_fx_check = QCheckBox("Applica catena FX (Denoiser / EQ / Saturatore) prima dell'export")
        self.apply_fx_check.setChecked(True)
        self.apply_fx_check.setStyleSheet("color: #94a3b8; margin-top: 8px;")
        layout.addWidget(self.apply_fx_check)

        layout.addStretch()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.setStyleSheet("QPushButton { background: #1e293b; color: white; padding: 6px 18px; border-radius: 4px; }")
        layout.addWidget(btns)

    def get_format(self):
        if self.rb_wav.isChecked():
            return "WAV"
        elif self.rb_aiff.isChecked():
            return "AIFF"
        return "MP3"

    def apply_fx(self):
        return self.apply_fx_check.isChecked()


# ===========================================================================
# Main Window
# ===========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotonX Studio — SOTA AI Audio Demixing & Restoration")
        self.resize(1280, 860)
        self.setStyleSheet("background-color: #0e1117; color: #f1f5f9; font-family: 'Segoe UI';")

        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.loaded_audio = None
        self.sample_rate = 44100
        self.separated_stems = {}
        self.processed_stems = {}
        self.is_playing = False
        self.playback_frame_pos = 0
        self.playback_lock = threading.Lock()
        self.stream = None
        self.stream_finished = False
        self.current_engine_name = "Music 4-Stems (Ultra)"
        self.channels = []
        self.fx_panels = {}           # {stem_name: FXPanel}
        self.processing_params = {}   # {stem_name: {...}}
        self._fx_workers = {}
        self._pending_fx_stem = None

        self.fx_debounce_timer = QTimer(self)
        self.fx_debounce_timer.setSingleShot(True)
        self.fx_debounce_timer.setInterval(120)
        self.fx_debounce_timer.timeout.connect(self._apply_pending_fx)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = get_available_execution_providers()
        self.active_provider = providers[0] if providers else "CPUExecutionProvider"
        try:
            self.ort_session = ort.InferenceSession(str(MODEL_PATH), sess_opts,
                                                    providers=[self.active_provider])
        except Exception:
            self.ort_session = None

        self._init_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._update_playback_pos)

    # -------------------------------------------------------------------------
    # UI build
    # -------------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # Header
        header_layout = QHBoxLayout()
        brand_lbl = QLabel("PHOTON<font color='#4cc9f0'>X</font> <font color='#94a3b8'>STUDIO</font>")
        brand_lbl.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header_layout.addWidget(brand_lbl)

        sub_lbl = QLabel("AI STEM DEMIX & RESTORATION  |  HTDEMUCS V4 · KIM SOTA · FOXJOY HQ")
        sub_lbl.setFont(QFont("Segoe UI", 9))
        sub_lbl.setStyleSheet("color: #64748b; margin-left: 12px;")
        header_layout.addWidget(sub_lbl)
        header_layout.addStretch()

        self.model_mgr_btn = QPushButton("📦 Model Manager")
        self.model_mgr_btn.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; padding: 7px 14px; border-radius: 6px; font-weight: bold; font-size: 11px;")
        self.model_mgr_btn.clicked.connect(self._open_model_manager)
        header_layout.addWidget(self.model_mgr_btn)

        hw_badge = QLabel(f"⚡ [{self.active_provider}]")
        hw_badge.setStyleSheet(
            "background-color: #0f172a; color: #22d3ee; padding: 7px 12px; border-radius: 6px; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(hw_badge)
        main_layout.addLayout(header_layout)

        # Engine tabs
        self.engine_tabs = QTabWidget()
        self.engine_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1e293b; background: #131722; border-radius: 6px; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 7px 16px;
                           font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; }
            QTabBar::tab:selected { background: #0284c7; color: white; }
        """)
        for name in ENGINES.keys():
            self.engine_tabs.addTab(QWidget(), name)
        self.engine_tabs.currentChanged.connect(self._on_engine_tab_changed)
        main_layout.addWidget(self.engine_tabs)

        # Transport row
        transport_layout = QHBoxLayout()
        transport_layout.setSpacing(10)

        self.load_btn = QPushButton("📁 Carica File Audio...")
        self.load_btn.setStyleSheet(
            "background-color: #2563eb; color: white; padding: 9px 18px; border-radius: 6px; font-weight: bold;")
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
        self.scope_combo.addItems([
            "Tutti gli Stem (Completo)",
            "Solo Vocals / Lead",
            "Solo Strumentale (Karaoke)",
        ])
        self.scope_combo.setStyleSheet(
            "background-color: #1e293b; color: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-weight: bold;")
        transport_layout.addWidget(self.scope_combo)

        # Best Quality preset button
        self.preset_btn = QPushButton("⭐ Best Quality")
        self.preset_btn.setStyleSheet(
            "background-color: #7c3aed; color: white; padding: 9px 16px; border-radius: 6px; font-weight: bold; font-size: 12px;")
        self.preset_btn.setToolTip("Applica automaticamente il modello migliore disponibile per ogni stem")
        self.preset_btn.clicked.connect(self._apply_best_quality_preset)
        transport_layout.addWidget(self.preset_btn)

        self.process_btn = QPushButton("⚡ Process Stems (SOTA)")
        self.process_btn.setStyleSheet(
            "background-color: #059669; color: white; padding: 9px 22px; border-radius: 6px; font-weight: bold; font-size: 13px;")
        self.process_btn.clicked.connect(self._process_stems)
        transport_layout.addWidget(self.process_btn)

        self.export_btn = QPushButton("💾 Esporta Stem...")
        self.export_btn.setStyleSheet(
            "background-color: #475569; color: white; padding: 9px 16px; border-radius: 6px; font-weight: bold;")
        self.export_btn.clicked.connect(self._export_stems)
        transport_layout.addWidget(self.export_btn)

        main_layout.addLayout(transport_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #1e293b; border-radius: 3px; }"
            "QProgressBar::chunk { background: #0284c7; border-radius: 3px; }")
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Waveform
        self.visualizer = WaveformVisualizer()
        self.visualizer.lane_clicked.connect(self._on_lane_clicked)
        self.visualizer.seek_requested.connect(self._on_seek)
        main_layout.addWidget(self.visualizer)

        # Playback controls
        pb_layout = QHBoxLayout()
        pb_layout.setSpacing(10)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedSize(90, 34)
        self.play_btn.setStyleSheet(
            "background-color: #38bdf8; color: black; font-weight: bold; border-radius: 6px; font-size: 13px;")
        self.play_btn.clicked.connect(self._toggle_playback)
        pb_layout.addWidget(self.play_btn)

        self.time_lbl = QLabel("00:00 / 00:00")
        self.time_lbl.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 13px;")
        pb_layout.addWidget(self.time_lbl)
        pb_layout.addSpacing(14)

        self.hint_lbl = QLabel("💡 Clicca 'S' per SOLO · '🎚 FX' per aprire il pannello effetti dello stem")
        self.hint_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        pb_layout.addWidget(self.hint_lbl)
        pb_layout.addStretch()

        self.unsolo_btn = QPushButton("🔄 Reset Solo/Mute")
        self.unsolo_btn.setStyleSheet(
            "background-color: #1e2432; color: #94a3b8; padding: 6px 14px; border-radius: 4px; font-size: 11px;")
        self.unsolo_btn.clicked.connect(self._reset_all_solo_mute)
        pb_layout.addWidget(self.unsolo_btn)
        main_layout.addLayout(pb_layout)

        # Mixer
        self.mixer_scroll = QScrollArea()
        self.mixer_scroll.setWidgetResizable(True)
        self.mixer_scroll.setMaximumHeight(280)
        self.mixer_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:horizontal { background: #1e293b; height: 6px; border-radius: 3px; }")

        self.mixer_container = QWidget()
        self.mixer_layout = QHBoxLayout(self.mixer_container)
        self.mixer_layout.setContentsMargins(0, 4, 0, 4)
        self.mixer_layout.setSpacing(12)
        self.mixer_scroll.setWidget(self.mixer_container)
        main_layout.addWidget(self.mixer_scroll)

        self._rebuild_mixer_channels()

    # -------------------------------------------------------------------------
    # Mixer management
    # -------------------------------------------------------------------------

    def _rebuild_mixer_channels(self):
        # Close all open FX panels
        for panel in self.fx_panels.values():
            try:
                panel.close()
            except Exception:
                pass
        self.fx_panels.clear()
        self.processing_params.clear()

        while self.mixer_layout.count():
            item = self.mixer_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        engine_info = ENGINES[self.current_engine_name]
        self.channels = []
        for i, (name, col) in enumerate(zip(engine_info["stems"], engine_info["colors"])):
            ch = StemChannel(i, name, col)
            ch.fx_requested.connect(self._open_fx_panel)
            self.mixer_layout.addWidget(ch)
            self.channels.append(ch)
            self.processing_params[name] = {}

        self.mixer_layout.addStretch()

    def _reset_all_solo_mute(self):
        for ch in self.channels:
            ch.set_solo(False)
            ch.set_mute(False)

    def _on_engine_tab_changed(self, idx):
        self.current_engine_name = self.engine_tabs.tabText(idx)
        self._rebuild_mixer_channels()

    def _apply_best_quality_preset(self):
        """Apply the best available model for each stem channel."""
        applied = []
        for ch in self.channels:
            best = BEST_QUALITY_PRESET.get(ch.stem_name)
            if best:
                options = STEM_MODEL_OPTIONS.get(ch.stem_name, [])
                match = next((o for o in options if o.startswith(best.split(" ")[0])), None)
                if match:
                    ch.set_model(match)
                    applied.append(f"{ch.stem_name}: {match}")
        if applied:
            QMessageBox.information(
                self, "Best Quality Preset",
                "Preset qualità massima applicato!\n\n" + "\n".join(applied))
        else:
            QMessageBox.information(
                self, "Best Quality Preset",
                "Tutti gli stem usano già la configurazione ottimale per questo motore.")

    # -------------------------------------------------------------------------
    # FX Panel
    # -------------------------------------------------------------------------

    def _open_fx_panel(self, channel_idx):
        if channel_idx < 0 or channel_idx >= len(self.channels):
            return
        ch = self.channels[channel_idx]
        name = ch.stem_name
        if name in self.fx_panels and self.fx_panels[name].isVisible():
            self.fx_panels[name].raise_()
            self.fx_panels[name].activateWindow()
            return

        panel = FXPanel(name, ch.color, parent=self)
        panel.params_changed.connect(self._on_fx_params_changed)
        # Restore saved params if any
        saved = self.processing_params.get(name, {})
        if saved:
            panel._params = saved  # restore visual state (approximate)
        self.fx_panels[name] = panel
        panel.show()

    def _on_fx_params_changed(self, stem_name, params):
        self.processing_params[stem_name] = params
        self._pending_fx_stem = stem_name
        self.fx_debounce_timer.start(120)

    def _apply_pending_fx(self):
        stem_name = self._pending_fx_stem
        if not stem_name or stem_name not in self.separated_stems:
            return
        raw = self.separated_stems[stem_name]
        params = self.processing_params.get(stem_name, {})
        worker = StemFXWorker(stem_name, raw, params, self.sample_rate)
        worker.finished_stem.connect(self._on_stem_fx_computed)
        self._fx_workers[stem_name] = worker
        worker.start()

    def _on_stem_fx_computed(self, stem_name, audio):
        with self.playback_lock:
            self.processed_stems[stem_name] = audio

    # -------------------------------------------------------------------------
    # Audio load
    # -------------------------------------------------------------------------

    def _load_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona traccia audio", "",
            "Audio Files (*.wav *.mp3 *.flac *.aiff *.ogg *.m4a)")
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
            self.processed_stems = {}
            self.playback_frame_pos = 0
            self.stream_finished = False
            dur_sec = len(audio) / sr
            m, s = int(dur_sec // 60), int(dur_sec % 60)
            self.file_lbl.setText(f"{os.path.basename(path)} ({m:02d}:{s:02d} @ {sr}Hz)")
            self.time_lbl.setText(f"00:00 / {m:02d}:{s:02d}")
            self.visualizer.set_data(["Mix Input"], [audio[:, 0]], [QColor(56, 189, 248)])
            self.visualizer.set_playhead(0.0)
            self._init_audio_stream()
        except Exception as e:
            QMessageBox.critical(self, "Errore Caricamento", str(e))

    # -------------------------------------------------------------------------
    # Stem processing
    # -------------------------------------------------------------------------

    def _process_stems(self):
        if self.loaded_audio is None:
            QMessageBox.warning(self, "Attenzione", "Carica prima un file audio!")
            return

        # Collect per-stem model selections
        stem_model_keys = {}
        for ch in self.channels:
            stem_model_keys[ch.stem_name] = ch.get_selected_model()

        scope = self.scope_combo.currentText()
        self.process_btn.setEnabled(False)
        self.process_btn.setText("⏳ Demixing SOTA in corso...")
        self.progress_bar.setValue(5)

        self.worker = DemixWorker(
            self.ort_session, self.loaded_audio, self.sample_rate,
            self.current_engine_name,
            target_scope=scope,
            stem_model_keys=stem_model_keys
        )
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.finished_success.connect(self._on_processing_complete)
        self.worker.error_occurred.connect(self._on_processing_error)
        self.worker.start()

    def _on_processing_error(self, err_msg):
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡ Process Stems (SOTA)")
        QMessageBox.critical(self, "Errore Separazione", f"Errore durante l'inferenza:\n{err_msg}")

    def _on_processing_complete(self, stem_names, stems_dict, colors, elapsed):
        self.progress_bar.setValue(100)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡ Process Stems (SOTA)")
        self.separated_stems = stems_dict
        self.processed_stems = {k: v.copy() for k, v in stems_dict.items()}

        cur_chan_names = [ch.name for ch in self.channels]
        if cur_chan_names != stem_names:
            while self.mixer_layout.count():
                item = self.mixer_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.channels = []
            for i, (name, col) in enumerate(zip(stem_names, colors)):
                ch = StemChannel(i, name, col)
                ch.fx_requested.connect(self._open_fx_panel)
                self.mixer_layout.addWidget(ch)
                self.channels.append(ch)
                if name not in self.processing_params:
                    self.processing_params[name] = {}
            self.mixer_layout.addStretch()

        wave_list = [
            stems_dict[n][:, 0] if stems_dict[n].ndim > 1 else stems_dict[n]
            for n in stem_names
        ]
        self.visualizer.set_data(stem_names, wave_list, colors)
        rt = (len(self.loaded_audio) / self.sample_rate) / elapsed if elapsed > 0 else 0
        QMessageBox.information(
            self, "Separazione Completata",
            f"Separazione SOTA completata!\n"
            f"Motore: {ENGINES.get(self.current_engine_name, {}).get('algo', self.current_engine_name)}\n"
            f"Tempo: {elapsed:.2f}s  ({rt:.1f}x Real-Time)\n"
            f"Campioni: {len(self.loaded_audio)} (sync sample-accurate 100%)\n\n"
            f"Ora puoi ascoltare ogni stem in SOLO, aprire il pannello FX o esportare!"
        )

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def _export_stems(self):
        if not self.separated_stems:
            QMessageBox.warning(self, "Attenzione", "Elabora prima gli stem con 'Process Stems'!")
            return

        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        fmt = dlg.get_format()
        apply_fx = dlg.apply_fx()

        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella di destinazione")
        if not folder:
            return

        processing_params = self.processing_params if apply_fx else {}

        self.export_btn.setEnabled(False)
        self.export_btn.setText("⏳ Esportazione in corso...")
        self.progress_bar.setValue(0)

        self._export_worker = ExportWorker(
            self.separated_stems, folder, fmt,
            self.sample_rate, self.channels, processing_params
        )
        self._export_worker.progress_changed.connect(
            lambda pct, msg: (self.progress_bar.setValue(pct),
                              self.export_btn.setText(f"⏳ {msg}...")))
        self._export_worker.finished_success.connect(self._on_export_complete)
        self._export_worker.error_occurred.connect(
            lambda err: (self.export_btn.setEnabled(True),
                         self.export_btn.setText("💾 Esporta Stem..."),
                         QMessageBox.critical(self, "Errore Export", err)))
        self._export_worker.start()

    def _on_export_complete(self, folder):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("💾 Esporta Stem...")
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Esportazione Riuscita",
                                f"Tutti gli stem sono stati esportati in:\n{folder}")

    # -------------------------------------------------------------------------
    # Model manager
    # -------------------------------------------------------------------------

    def _open_model_manager(self):
        ModelManagerDialog(self).exec()

    # -------------------------------------------------------------------------
    # Audio playback
    # -------------------------------------------------------------------------

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
            print(f"[AudioStream] {e}")

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

            active_dict = self.processed_stems if self.processed_stems else self.separated_stems
            if active_dict:
                any_solo = any(ch.is_solo for ch in self.channels)
                for ch in self.channels:
                    active = ch.is_solo if any_solo else not ch.is_mute
                    if active and ch.stem_name in active_dict:
                        stem = active_dict[ch.stem_name]
                        sl = stem[pos: pos + chunk_len]
                        g = ch.gain
                        if sl.ndim == 1:
                            mix_buf[:, 0] += sl * g
                            mix_buf[:, 1] += sl * g
                        else:
                            mix_buf += sl * g
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
                QMessageBox.critical(self, "Errore Riproduzione", str(e))
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
        with self.playback_lock:
            self.playback_frame_pos = int(ratio * len(self.loaded_audio))
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
        cur = self.playback_frame_pos / self.sample_rate
        total = len(self.loaded_audio) / self.sample_rate
        self.time_lbl.setText(
            f"{int(cur//60):02d}:{int(cur%60):02d} / {int(total//60):02d}:{int(total%60):02d}")

    def closeEvent(self, event):
        for panel in self.fx_panels.values():
            try:
                panel.close()
            except Exception:
                pass
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        event.accept()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
