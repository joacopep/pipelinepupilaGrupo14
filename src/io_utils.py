"""Lectura de video con timestamps reales por frame.

Los celulares suelen grabar con frame rate variable (VFR), por lo que el tiempo de cada
frame se toma del timestamp de presentación del contenedor (CAP_PROP_POS_MSEC) y no de n / fps.
"""
import struct
from pathlib import Path

import cv2
import numpy as np


def _open(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV no pudo abrir el video: {path}")
    return cap


def video_info(path):
    """Metadatos nominales del contenedor (no requieren decodificar el video)."""
    cap = _open(path)
    info = {
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_nominal": cap.get(cv2.CAP_PROP_FPS),
        "frames_nominal": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "rotacion": int(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
    }
    cap.release()
    return info


def iter_frames(path, gray=False, step=1):
    """Generador de (t, frame) para cada frame decodificable del video (o uno de cada `step`).

    t: tiempo en segundos relativo al primer frame (timestamp real del contenedor).
    frame: BGR uint8 (o escala de grises si gray=True), ya rotado según el metadato del video.
    Recorre el video sin cargarlo entero en memoria. Con step > 1 los frames intermedios se avanzan con
    grab() (se decodifican, porque el códec lo necesita, pero no se convierten a BGR) y no se devuelven.
    """
    cap = _open(path)
    t0 = None
    k = 0
    try:
        while True:
            if k % step:
                k += 1
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            k += 1
            if not ok:
                break
            # Tras read(), POS_MSEC es el timestamp de presentación del frame recién leído
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t0 is None:
                t0 = t_ms
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield (t_ms - t0) / 1000.0, frame
    finally:
        cap.release()


def read_video(path, gray=False):
    """Lee el video completo. Devuelve (frames, t).

    frames: array (N, H, W, 3) BGR o (N, H, W) en grises, uint8.
    t: array (N,) de tiempos reales en segundos, relativos al primer frame.
    Ojo: a resolución completa ocupa N*H*W*3 bytes (~6 MB por frame a 1080p);
    para videos largos conviene iterar con iter_frames.
    """
    frames, t = [], []
    for ti, f in iter_frames(path, gray=gray):
        t.append(ti)
        frames.append(f)
    if not frames:
        raise IOError(f"No se pudo decodificar ningún frame de {path}")
    t = np.asarray(t)
    check_timestamps(t, path)
    return np.stack(frames), t


def read_frame_at(path, t_s):
    """Devuelve (t, frame) del primer frame con tiempo >= t_s (segundos desde el primer frame).

    Se recorre el video en lugar de hacer seek para usar la misma referencia temporal que iter_frames.
    """
    for t, frame in iter_frames(path):
        if t >= t_s:
            return t, frame
    raise ValueError(f"t={t_s} s está fuera del video {path}")


_TRANSFER = {1: "BT.709 (SDR)", 6: "BT.601 (SDR)", 13: "sRGB", 14: "BT.2020 (SDR)", 15: "BT.2020 (SDR)",
             16: "PQ (HDR)", 18: "HLG (HDR)"}


def _moov(path):
    """Bytes de la caja 'moov' (metadatos) de un MP4/MOV, recorriendo solo las cabeceras de nivel superior."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            size, typ = struct.unpack(">I4s", hdr)
            head = 8
            if size == 1:
                size, head = struct.unpack(">Q", f.read(8))[0], 16
            elif size == 0:  # la caja llega hasta el final del archivo
                return f.read() if typ == b"moov" else None
            if typ == b"moov":
                return f.read(size - head)
            f.seek(size - head, 1)


def _avc_bit_depth(rec):
    """Profundidad de luma de un AVCDecoderConfigurationRecord (H.264), o None.

    Los perfiles Baseline/Main/Extended/High (66, 77, 88, 100) son siempre de 8 bits. Los demás perfiles
    "High" llevan la profundidad en la extensión que sigue a las listas de SPS y PPS.
    """
    profile = rec[1]
    if profile in (66, 77, 88, 100):
        return 8
    k = 5
    n_sps = rec[k] & 0x1F
    k += 1
    for _ in range(n_sps):
        k += 2 + struct.unpack(">H", rec[k:k + 2])[0]
    n_pps = rec[k]
    k += 1
    for _ in range(n_pps):
        k += 2 + struct.unpack(">H", rec[k:k + 2])[0]
    # extensión: chroma_format (1 byte), bit_depth_luma_minus8 (3 bits bajos del byte siguiente)
    return 8 + (rec[k + 1] & 0x07) if k + 1 < len(rec) else None


def source_format(path):
    """Códec, profundidad de bits y función de transferencia del stream original (MP4/MOV).

    OpenCV siempre entrega frames de 8 bits, pero el códec puede haber grabado más (p. ej. 10 bits).
    Se lee del registro del códec ('hvcC' para HEVC, 'avcC' para H.264) y de la caja de color 'colr'
    dentro de 'moov' (no en todo el archivo: los datos comprimidos pueden contener esas cadenas por azar).
    Valores None si no se encuentran.
    """
    moov = _moov(path) or b""
    out = {"codec": None, "bits": None, "transferencia": None}
    i = moov.find(b"hvcC")
    if i >= 0 and moov[i + 4] == 1:  # configurationVersion = 1
        # HEVCDecoderConfigurationRecord: bitDepthLumaMinus8 en el byte 17 (3 bits bajos)
        out["codec"], out["bits"] = "HEVC", 8 + (moov[i + 4 + 17] & 0x07)
    i = moov.find(b"avcC")
    if out["codec"] is None and i >= 0 and moov[i + 4] == 1:
        size = struct.unpack(">I", moov[i - 4:i])[0]
        out["codec"], out["bits"] = "H.264", _avc_bit_depth(moov[i + 4:i - 4 + size])
    j = moov.find(b"colr")
    if j >= 0 and moov[j + 4:j + 8] in (b"nclx", b"nclc"):
        tc = struct.unpack(">H", moov[j + 10:j + 12])[0]
        out["transferencia"] = _TRANSFER.get(tc, f"código {tc}")
    return out


def check_timestamps(t, path=""):
    """Verifica que los timestamps sean estrictamente crecientes (si no, el backend no los reporta bien)."""
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError(f"Timestamps no estrictamente crecientes en {path}: "
                         "el backend de OpenCV no está reportando tiempos reales")


def timestamp_stats(t, vfr_tolerance):
    """Estadísticas de los intervalos entre frames consecutivos.

    vfr_tolerance: desvío relativo máximo de un intervalo respecto de la mediana para
    considerarlo regular. El video se marca como VFR si algún intervalo lo excede.
    """
    dt = np.diff(t) * 1000.0  # ms
    med = np.median(dt)
    irregular = np.abs(dt - med) > vfr_tolerance * med
    return {
        "dt_media_ms": dt.mean(),
        "dt_mediana_ms": med,
        "dt_std_ms": dt.std(),
        "dt_min_ms": dt.min(),
        "dt_max_ms": dt.max(),
        "n_irregulares": int(irregular.sum()),
        "fps_real": (len(t) - 1) / (t[-1] - t[0]),
        "vfr": bool(irregular.any()),
    }
