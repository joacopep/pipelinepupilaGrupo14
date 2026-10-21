"""Detección automática del instante del estímulo luminoso a partir del brillo medio de cada frame."""
import json
from pathlib import Path

import numpy as np


def stimulus_time(t, brillo):
    """Instante del estímulo: el frame con el mayor salto de brillo respecto del anterior.

    El estímulo (encender o apagar la luz) es un escalón de brillo de uno o dos frames, mucho más brusco
    que cualquier otra variación (movimientos, ajustes de la cámara). Devuelve (t_estimulo, salto), con
    t_estimulo el tiempo del primer frame ya afectado y salto > 0 si la luz se enciende, < 0 si se apaga.
    """
    d = np.diff(np.asarray(brillo, dtype=float))
    k = int(np.argmax(np.abs(d)))
    return float(t[k + 1]), float(d[k])


def video_timing(path, cache_dir=None):
    """(t_estimulo, salto, t_ultimo_frame) de un video, recorriendo todos sus frames.

    Recorrer un video 4K completo lleva tiempo, así que si se da `cache_dir` el resultado se guarda en
    cache_dir/estimulo.json, indexado por ruta, tamaño y fecha de modificación del archivo (si el video
    cambia, se recalcula).
    """
    from .io_utils import iter_frames
    path = Path(path)
    st = path.stat()
    clave = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    archivo = Path(cache_dir) / "estimulo.json" if cache_dir else None
    cache = {}
    if archivo and archivo.is_file():
        cache = json.loads(archivo.read_text(encoding="utf-8"))
        if len(cache.get(clave, [])) == 3:
            return tuple(cache[clave])
    t, b = zip(*[(ti, f.mean()) for ti, f in iter_frames(path, gray=True)])
    res = (*stimulus_time(np.asarray(t), np.asarray(b)), float(t[-1]))
    if archivo:
        archivo.parent.mkdir(parents=True, exist_ok=True)
        cache[clave] = list(res)
        archivo.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return res


def video_stimulus(path, cache_dir=None):
    """(t_estimulo, salto) de un video (ver video_timing, que además da la duración)."""
    return video_timing(path, cache_dir)[:2]


def frame_relative(path, offset_s, cache_dir=None):
    """(t, frame, t_estimulo, salto) del frame ubicado `offset_s` segundos después del estímulo
    (negativo = antes). El tiempo se recorta al rango del video."""
    from .io_utils import read_frame_at
    t_est, salto = video_stimulus(path, cache_dir)
    t, frame = read_frame_at(path, max(t_est + offset_s, 0.0))
    return t, frame, t_est, salto


def step_metrics(t, brillo, t_est, p):
    """Forma del escalón de brillo alrededor del estímulo (control de calidad de la grabación).

    Niveles: antes = mediana del brillo en [t_est + pre_s[0], t_est + pre_s[1]]; después = mediana en la meseta
    [t_est + meseta_s[0], t_est + meseta_s[1]]. Devuelve:
      transicion_frames: intervalos entre frames desde el último frame por debajo del 10 % del escalón hasta el
                         primero por encima del 90 % (1 = el escalón ocurre entre dos frames consecutivos);
      sobrepico_pct: exceso máximo más allá del nivel final en los primeros meseta_s[0] s, en % del escalón (el
                     control automático de exposición de la cámara sobrecorrige y después se recupera);
      asentamiento_s: desde el estímulo hasta que el brillo queda dentro de +-tol_pct % del escalón alrededor del
                      nivel final;
      meseta_pct / base_pct: rango p5-p95 del brillo en la meseta y en la línea de base [base_s[0], t_est + pre_s[1]],
                             en % del escalón (planitud).
    p: config inspeccion.escalon.
    """
    t, b = np.asarray(t, float), np.asarray(brillo, float)

    def tramo(t0, t1):
        return b[(t >= t0) & (t <= t1)]
    antes = np.median(tramo(t_est + p["pre_s"][0], t_est + p["pre_s"][1]))
    despues = np.median(tramo(t_est + p["meseta_s"][0], t_est + p["meseta_s"][1]))
    esc = despues - antes
    f = (b - antes) / esc                                    # 0 antes, 1 después, para subidas y bajadas
    k0 = int(np.searchsorted(t, t_est + p["pre_s"][1]))
    k10 = k0 + int(np.argmax(f[k0:] >= 0.1))                 # primer frame por encima del 10 %
    k90 = k10 + int(np.argmax(f[k10:] >= 0.9))
    ventana = (t >= t_est) & (t <= t_est + p["meseta_s"][0])
    fuera = ventana & (np.abs(f - 1) > p["tol_pct"] / 100)
    meseta = tramo(t_est + p["meseta_s"][0], t_est + p["meseta_s"][1])
    base = tramo(p["base_s"], t_est + p["pre_s"][1])
    return {
        "brillo antes": antes, "brillo después": despues, "escalón": esc,
        "transicion_frames": k90 - k10 + 1,
        "sobrepico_pct": 100 * max(0.0, float(f[ventana].max() - 1)),
        "asentamiento_s": float(t[fuera][-1] - t_est) if fuera.any() else 0.0,
        "meseta_pct": 100 * float(np.ptp(np.percentile(meseta, [5, 95]))) / abs(esc),
        "base_pct": 100 * float(np.ptp(np.percentile(base, [5, 95]))) / abs(esc),
    }
