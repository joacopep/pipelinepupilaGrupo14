"""Módulo 3.2: curva temporal del diámetro pupilar y parámetros fisiológicos del reflejo fotomotor (pipeline).

1. procesar_video: un recorrido del video. En cada frame, brillo medio (para detectar el estímulo), ROI con
   seguimiento del iris (src/roi.py), preprocesamiento + watershed + indicador de calidad (src/segment.py). La
   segmentación se reparte entre procesos; la ROI es secuencial (cada frame parte de la ROI del anterior). Segunda
   pasada con continuidad temporal (la última pupila aceptada restringe la segmentación del frame).
2. limpiar_curva: rechazo de valores atípicos temporales (Hampel), parpadeos (tramos sin medición o caída brusca del
   área entre las mediciones aceptadas) e interpolación de los huecos cortos; los largos quedan NaN.
3. suavizar: Butterworth pasa-bajos de fase cero (filtfilt) por tramos continuos, sobre una grilla uniforme en el
   tiempo real; derivada (velocidad) de la curva suavizada.
4. parametros: latencia, amplitud y velocidades máximas.
Todos los tiempos son reales (timestamps del contenedor) y los parámetros temporales están en segundos; los tamaños,
en mm (escala del iris) o relativos al basal, así que nada depende de la cámara ni del frame rate.
Parámetros: secciones `curva` y `parametros` de config.yaml.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from . import segment as sm
from .estimulo import stimulus_time
from .io_utils import iter_frames
from .roi import crop_roi, detect_eye_roi, init_video_roi

# Campos de medir_pupila que se guardan por frame
_CAMPOS = ("diametro_roi", "area_roi", "inliers", "residuo", "circularidad", "diametro_iris", "confiable")


def _medir(args):
    """Medición de una ROI en un proceso de trabajo: solo los números (las máscaras no se devuelven)."""
    bgr, iris, p, previa = args
    o = sm.medir_pupila(bgr, iris, p, previa)
    e = o["elipse"]
    return {**{k: o[k] for k in _CAMPOS}, "cx_roi": o["centroide_roi"][0], "cy_roi": o["centroide_roi"][1],
            "excentricidad": float(np.sqrt(1 - (e["b"] / e["a"]) ** 2)) if e else np.nan}


def previas(df, vigencia_s):
    """Última pupila aceptada por el indicador (estricto) antes de cada frame, si es de hace menos de vigencia_s:
    lista de {cx, cy, r, edad_s} (px de la ROI; la ROI está centrada en el iris en cada frame, así que el centro de la
    pupila es comparable entre frames) o None."""
    out, ultima = [], None
    for f in df.itertuples():
        out.append({"cx": ultima.cx_roi, "cy": ultima.cy_roi, "r": ultima.diametro_roi / 2, "edad_s": f.t - ultima.t}
                   if ultima is not None and f.t - ultima.t <= vigencia_s else None)
        if f.confiable and np.isfinite(f.diametro_roi):
            ultima = f
    return out


def procesar_video(path, cfg, procesos=None, progreso=None):
    """Recorre el video y mide la pupila en cada frame.

    Primera pasada: brillo, ROI con seguimiento (secuencial) y segmentación de cada frame (repartida entre procesos).
    Segunda pasada (continuidad temporal, segmentacion.continuidad): en cada frame con una pupila aceptada por el
    indicador en los vigencia_s anteriores (de la primera pasada), se vuelve a segmentar con esa pupila como referencia
    (segment.medir_pupila con `previa`) sobre la misma ROI; queda la medición de la segunda pasada si el indicador la
    acepta o si la primera tampoco era aceptada, y si no, la de la primera.

    Devuelve (df, info). df: una fila por frame con t [s] (timestamp real), brillo (gris medio del frame), iris
    (si se encontró el iris en el frame), metodo de la ROI y su geometría (roi_cx, roi_cy, roi_side, px del frame),
    medidas en px de la ROI (diametro_roi, area_roi, cx_roi, cy_roi), medidas del indicador, `confiable`, la pupila de
    referencia usada (previa_cx, previa_cy, previa_r, previa_edad_s; NaN sin referencia) y `segunda_pasada` (la medición
    es la de la segunda pasada). info: ROI inicial (escala), mm por px del frame original y de la ROI, iris de la ROI.
    Si no se encuentra el iris en el primer segundo, la escala no existe: ValueError.
    """
    c = cfg["roi"]
    roi0 = init_video_roi(path, c)
    if roi0 is None:
        raise ValueError(f"No se encontró el iris en el primer {c['scale_window_s']} s de {path}")
    iris = sm.iris_en_roi(c["size"], c["side_iris_diameters"])
    p = cfg["segmentacion"]
    procesos = procesos or cfg["curva"]["procesos"] or os.cpu_count()
    filas, futuros, rois = [], [], []
    prev = roi0
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        for k, (t, frame) in enumerate(iter_frames(path)):
            b, g, r = cv2.mean(frame)[:3]
            roi = detect_eye_roi(frame, c, prev, t)
            hay_iris = roi.iris_r_frame is not None
            filas.append({"frame": k, "t": t, "brillo": 0.114 * b + 0.587 * g + 0.299 * r, "iris": hay_iris,
                          "metodo_roi": roi.metodo, "roi_cx": roi.cx, "roi_cy": roi.cy, "roi_side": roi.side})
            # Sin iris en el frame (parpadeo, ojo fuera de cuadro) no se mide: queda NaN
            futuros.append(pool.submit(_medir, (crop_roi(frame, roi), iris, p, None)) if hay_iris else None)
            rois.append(roi if hay_iris else None)
            if hay_iris:
                prev = roi
            if progreso:
                progreso(k, t)
        medidas = [f.result() if f is not None else {} for f in futuros]
        df = pd.DataFrame(filas).join(pd.DataFrame(medidas))
        df["confiable"] = df["confiable"].eq(True)             # sin medición = no confiable
        # Segunda pasada: continuidad temporal con la última pupila aceptada
        pc = p["continuidad"]
        ref = previas(df, pc["vigencia_s"]) if pc["usar"] else [None] * len(df)
        futuros = {}
        if any(x is not None and rois[k] is not None for k, x in enumerate(ref)):
            for k, (t, frame) in enumerate(iter_frames(path)):
                if ref[k] is not None and rois[k] is not None:
                    futuros[k] = pool.submit(_medir, (crop_roi(frame, rois[k]), iris, p, ref[k]))
        segunda = {k: f.result() for k, f in futuros.items()}
    df["segunda_pasada"] = False
    for k, m2 in segunda.items():
        if m2["confiable"] or not df.at[k, "confiable"]:
            for campo, v in m2.items():
                df.at[k, campo] = v
            df.at[k, "segunda_pasada"] = True
    df["confiable"] = df["confiable"].eq(True)
    for campo in ("cx", "cy", "r", "edad_s"):
        df[f"previa_{campo}"] = [x[campo] if x is not None else np.nan for x in ref]
    mm_roi = cfg["iris_diameter_mm"] / (2 * iris[2])
    info = {"roi": roi0, "iris_roi": iris, "mm_por_px_roi": mm_roi, "px_frame_por_px_roi": 1 / roi0.scale,
            "mm_por_px": roi0.mm_per_px(cfg["iris_diameter_mm"]), "iris_diametro_px": 2 * roi0.iris_r}
    return df, info


# ---------------------------------------------------------------------------------------------
# Estímulo
# ---------------------------------------------------------------------------------------------

def estimulo(t, brillo):
    """Instante del estímulo con precisión menor a un frame y su tipo.

    El frame k es el primero afectado (mayor salto de brillo, estimulo.stimulus_time). Con obturador rodante las
    filas se leen a lo largo de un período de frame: si la luz cambió durante la lectura del frame k, solo una
    fracción f de él está iluminada (f = (b[k] - b[k-1]) / (b[k+1] - b[k-1]), entre 0 y 1). Se estima el cambio en
    t[k] - f * (t[k] - t[k-1]): f = 1 (frame k completo ya afectado) lo ubica en t[k-1], f = 0.5 a mitad de camino.
    Devuelve (t_estimulo, salto, tipo) con tipo 'luz ON' (salto > 0) o 'luz OFF'.
    """
    t, b = np.asarray(t, float), np.asarray(brillo, float)
    t_k, salto = stimulus_time(t, b)
    k = int(np.searchsorted(t, t_k))
    t_est = t_k
    if 1 <= k < len(b) - 1 and b[k + 1] != b[k - 1]:
        f = float(np.clip((b[k] - b[k - 1]) / (b[k + 1] - b[k - 1]), 0, 1))
        t_est = t[k] - f * (t[k] - t[k - 1])
    return t_est, salto, "luz ON" if salto > 0 else "luz OFF"


# ---------------------------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------------------------

def _ventana(t, s):
    """Semiancho [muestras] de una ventana de s segundos con el intervalo mediano entre frames."""
    return max(1, int(round(s / np.median(np.diff(t)))))


def hampel(x, k, n_sigmas):
    """Filtro de Hampel: True donde |x - mediana móvil| > n_sigmas * 1.4826 * MAD móvil (ventana de 2k+1 muestras,
    ignorando NaN). Las muestras NaN no se marcan."""
    s = pd.Series(x)
    med = s.rolling(2 * k + 1, center=True, min_periods=k + 1).median()
    mad = (s - med).abs().rolling(2 * k + 1, center=True, min_periods=k + 1).median() * 1.4826
    return ((s - med).abs() > n_sigmas * mad.clip(lower=1e-9)).to_numpy() & np.isfinite(x)


def _tramos(mask):
    """Tramos [i0, i1) donde mask es True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def limpiar_curva(t, d, area, p, aceptada=None):
    """Rechazo de artefactos e interpolación de huecos cortos.

    t [s], d (diámetro, NaN si el frame no es confiable) y area (de la máscara, NaN si no hay medición). p: sección
    `curva`. Pasos:
      1. atípicos: filtro de Hampel sobre d (ventana hampel_ventana_s, umbral hampel_n_sigmas) -> NaN;
      2. parpadeos: un parpadeo tapa la pupila desde arriba y el área cae antes de que se pierda la medición. Se
         marca parpadeo todo tramo sin medición de más de parpadeo_min_s y los frames cuya área cae más de
         parpadeo_caida_area por debajo de la mediana de los parpadeo_ventana_s anteriores y también de la de los
         posteriores (caída transitoria: una constricción, que no se recupera, no la cumple); cada parpadeo se extiende
         parpadeo_margen_s a cada lado (párpado entrando y saliendo) -> NaN. La caída se busca solo en las mediciones
         que acepta el indicador estricto (`aceptada`; por defecto, d finito) y las medianas de referencia se calculan
         solo con ellas, aunque la fase se haya aceptado con un nivel menor: un área de una medición rechazada
         (p. ej. un derrame, mucho más grande) no puede hacer que una pupila bien medida parezca una caída (A8 y B7:
         con todas las mediciones, el 66-69 % de los frames quedaba marcado como parpadeo);
      3. interpolación lineal (en el tiempo real) de los huecos de hasta hueco_max_s; los más largos quedan NaN.
    Devuelve (d_limpio, estado): estado por frame 'ok', 'rechazado' (indicador o sin medición), 'atipico',
    'parpadeo', 'interpolado' o 'hueco' (NaN final).
    """
    t, d, area = (np.asarray(v, float) for v in (t, d, area))
    estado = np.where(np.isfinite(d), "ok", "rechazado").astype(object)
    at = hampel(d, _ventana(t, p["hampel_ventana_s"]), p["hampel_n_sigmas"])
    estado[at] = "atipico"
    d = np.where(at, np.nan, d)

    parp = np.zeros(len(t), bool)
    for i0, i1 in _tramos(~np.isfinite(area)):
        if t[min(i1, len(t) - 1)] - t[i0] >= p["parpadeo_min_s"] or i1 - i0 > 1 and \
                t[i1 - 1] - t[i0] >= p["parpadeo_min_s"]:
            parp[i0:i1] = True
    # Caída transitoria: más baja que el nivel de antes Y que el de después (una constricción sostenida no lo es)
    # Solo con las mediciones aceptadas: las rechazadas no entran en la referencia ni se marcan
    w = _ventana(t, p["parpadeo_ventana_s"])
    aceptada = (np.isfinite(d) if aceptada is None else np.asarray(aceptada, bool)) & np.isfinite(d) & np.isfinite(area)
    s_a = pd.Series(np.where(aceptada, area, np.nan))
    antes = s_a.rolling(w, min_periods=1).median().shift(1).to_numpy()
    despues = s_a[::-1].rolling(w, min_periods=1).median().shift(1)[::-1].to_numpy()
    ref = np.fmin(antes, despues)
    parp |= aceptada & np.isfinite(ref) & (area < (1 - p["parpadeo_caida_area"]) * ref)
    margen = p["parpadeo_margen_s"]
    for i0, i1 in _tramos(parp):
        parp |= (t >= t[i0] - margen) & (t <= t[i1 - 1] + margen)
    estado[parp] = "parpadeo"
    d = np.where(parp, np.nan, d)

    out = d.copy()
    for i0, i1 in _tramos(~np.isfinite(d)):
        if i0 == 0 or i1 == len(t):
            estado[i0:i1] = np.where(estado[i0:i1] == "ok", "hueco", estado[i0:i1])
            continue
        if t[i1] - t[i0 - 1] <= p["hueco_max_s"]:
            out[i0:i1] = np.interp(t[i0:i1], [t[i0 - 1], t[i1]], [d[i0 - 1], d[i1]])
            estado[i0:i1] = "interpolado"
    return out, estado


# ---------------------------------------------------------------------------------------------
# Suavizado y derivada
# ---------------------------------------------------------------------------------------------

def suavizar(t, d, fc_hz, orden):
    """Butterworth pasa-bajos de fase cero (filtfilt) con corte fc_hz, y su derivada [unidades de d / s].

    filtfilt necesita muestreo uniforme: cada tramo continuo (sin NaN) se interpola a una grilla uniforme con el
    intervalo mediano entre frames, se filtra y se vuelve a los tiempos reales (si el video es CFR la grilla
    coincide con los frames). Los tramos demasiado cortos para el filtro quedan NaN. La velocidad se descarta (NaN) a
    menos de medio período del corte, 1 / (2 fc), de los bordes de cada tramo: ahí filtfilt tiene transitorio y un
    tramo que empieza en plena respuesta (sin nivel previo) da una derivada espuria. Devuelve (d_suave, velocidad).
    """
    from scipy.signal import butter, filtfilt
    t, d = np.asarray(t, float), np.asarray(d, float)
    dt = float(np.median(np.diff(t)))
    b, a = butter(orden, fc_hz / (0.5 / dt))
    pad = 3 * max(len(a), len(b))
    borde = 0.5 / fc_hz
    suave, vel = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i0, i1 in _tramos(np.isfinite(d)):
        tt = t[i0:i1]
        n = int(round((tt[-1] - tt[0]) / dt)) + 1
        if n <= pad:
            continue
        tu = tt[0] + dt * np.arange(n)
        y = filtfilt(b, a, np.interp(tu, tt, d[i0:i1]))
        suave[i0:i1] = np.interp(tt, tu, y)
        vel[i0:i1] = np.where((tt >= tt[0] + borde) & (tt <= tt[-1] - borde), np.interp(tt, tu, np.gradient(y, dt)),
                              np.nan)
    return suave, vel


def espectro(t, d):
    """Espectro de amplitud de d(t) (tramo continuo más largo, sin la media, ventana de Hann), en una grilla
    uniforme con el intervalo mediano entre frames. Devuelve (f [Hz], |D(f)|)."""
    t, d = np.asarray(t, float), np.asarray(d, float)
    i0, i1 = max(_tramos(np.isfinite(d)), key=lambda r: r[1] - r[0])
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[i0], t[i1 - 1], dt)
    y = np.interp(tu, t[i0:i1], d[i0:i1])
    y = (y - y.mean()) * np.hanning(len(y))
    return np.fft.rfftfreq(len(y), dt), np.abs(np.fft.rfft(y)) * 2 / np.hanning(len(y)).sum()


# ---------------------------------------------------------------------------------------------
# Parámetros fisiológicos
# ---------------------------------------------------------------------------------------------

def _cruce(t, y, nivel, i0, bajando):
    """Primer cruce de y por `nivel` desde la muestra i0 (bajando o subiendo), con interpolación lineal entre las
    dos muestras que lo rodean (precisión menor a un frame). NaN si no cruza."""
    s = (y[i0:] < nivel) if bajando else (y[i0:] > nivel)
    s &= np.isfinite(y[i0:])
    if not s.any():
        return np.nan
    j = i0 + int(np.argmax(s))
    if j == 0 or not np.isfinite(y[j - 1]) or y[j] == y[j - 1]:
        return float(t[j])
    return float(t[j - 1] + (nivel - y[j - 1]) * (t[j] - t[j - 1]) / (y[j] - y[j - 1]))


NIVELES = {3: "alta", 2: "media", 1: "baja", 0: ""}


def sigmoide(t, d0, a, t50, s):
    """Sigmoide logística d0 + a / (1 + exp(-(t - t50) / s)): pasa de d0 a d0 + a, con la mitad del cambio en t50 y
    pendiente máxima a / (4 s) en t50."""
    return d0 + a / (1 + np.exp(-np.clip((t - t50) / s, -50, 50)))


def ajustar_sigmoide(t, d, t0, t1, signo, p):
    """Ajuste robusto de una sigmoide a la curva d(t) en [t0, t1] (velocidad máxima sin derivar el ruido).

    t, d: tiempo [s] y diámetro limpio sin suavizar [mm] (NaN = sin dato). signo: sentido del cambio (-1 constricción,
    +1 dilatación); la amplitud se restringe a ese sentido. p: sección `parametros` (modelo_*). Mínimos cuadrados con
    pérdida soft L1 (escala modelo_f_scale_mm): las muestras que se apartan mucho pesan menos. Devuelve dict con los
    parámetros (d0, a, t50, s), vel_max = a / (4 s) [mm/s], r2 (sobre las muestras de la ventana), n, ok (convergió),
    en_borde (t50 o s en el límite de su rango: la ventana no contiene la transición entera) y sel (muestras usadas), o
    None si hay menos de modelo_min_puntos muestras.
    """
    from scipy.optimize import least_squares
    sel = (t >= t0) & (t <= t1) & np.isfinite(d)
    if sel.sum() < p["modelo_min_puntos"]:
        return None
    tt, dd = t[sel], d[sel]
    k = max(1, len(dd) // 10)
    d0, d1 = float(np.median(dd[:k])), float(np.median(dd[-k:]))
    a0 = d1 - d0 if signo * (d1 - d0) > 0 else signo * max(float(np.ptp(dd)), 1e-3)
    t50_0 = float(tt[np.argmin(np.abs(dd - (d0 + a0 / 2)))])
    s_lo, s_hi = p["modelo_s_s"]
    x0 = [d0, a0, float(np.clip(t50_0, t0, t1)), float(np.clip((t1 - t0) / 10, s_lo, s_hi))]
    lo = [-np.inf, 0 if signo > 0 else -np.inf, t0, s_lo]
    hi = [np.inf, np.inf if signo > 0 else 0, t1, s_hi]
    x0 = [float(np.clip(v, l + 1e-9 if np.isfinite(l) else v, h - 1e-9 if np.isfinite(h) else v))
          for v, l, h in zip(x0, lo, hi)]
    try:
        r = least_squares(lambda x: sigmoide(tt, *x) - dd, x0, bounds=(lo, hi), loss="soft_l1",
                          f_scale=p["modelo_f_scale_mm"])
    except (ValueError, np.linalg.LinAlgError):
        return {"ok": False, "vel_max": np.nan, "r2": np.nan, "n": int(sel.sum()), "en_borde": True, "sel": sel}
    d0, a, t50, s = r.x
    res = sigmoide(tt, *r.x) - dd
    ss = float(np.sum((dd - dd.mean()) ** 2))
    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.0
    borde = t50 <= t0 + dt or t50 >= t1 - dt or np.isclose(s, s_lo) or np.isclose(s, s_hi)
    return {"ok": bool(r.success), "d0": d0, "a": a, "t50": t50, "s": s, "vel_max": a / (4 * s),
            "r2": 1 - float(np.sum(res ** 2)) / ss if ss > 0 else np.nan, "n": int(sel.sum()), "en_borde": bool(borde),
            "sel": sel}


def niveles_frame(df, p_cal, p_media):
    """Nivel de aceptación de cada frame: 3 = alta (indicador estricto), 2 = media (umbrales relajados de
    `segmentacion.calidad_media`), 1 = baja (hay medición pero no cumple ninguno), 0 = sin medición."""
    from .morfologia import es_confiable
    media = np.array([es_confiable({"elipse": True, "inliers": f.inliers, "residuo": f.residuo,
                                    "circularidad": f.circularidad, "diametro_iris": f.diametro_iris},
                                   {**p_cal, **p_media}) if np.isfinite(f.inliers) else False
                      for f in df.itertuples()])
    medido = np.isfinite(df.diametro_roi.to_numpy(float))
    return np.select([df.confiable.to_numpy(bool), media, medido], [3, 2, 1], 0)


def nivel_por_fase(t, nivel, t_est, min_frac):
    """Nivel mínimo aceptado en cada frame según su fase (antes / después del estímulo).

    En cada fase se usa el nivel más alto (alta, media, baja) que alcanzan al menos min_frac de sus frames: si el
    indicador estricto rechaza una fase entera (p. ej. oscuridad sin luz de relleno), se acepta con umbrales relajados
    y, si tampoco alcanza, cualquier medición. Devuelve (nivel mínimo por frame, {fase: nivel}).
    """
    minimo = np.full(len(t), 3)
    fases = {}
    for nombre, sel in (("pre", t < t_est), ("post", t >= t_est)):
        if not sel.any():
            continue
        L = next((L for L in (3, 2, 1) if (nivel[sel] >= L).mean() >= min_frac), 1)
        minimo[sel] = L
        fases[nombre] = L
    return minimo, fases


def _respaldo(t, d, crudo, t_est, tipo, p, out, nc):
    """Parámetros principales sin curva en la ventana de respuesta (completa `out` y `nc`; confianza baja en todos).

    Mediciones: las de la curva limpia `d` (cualquier nivel aceptado, sin suavizar) en [t_est + partida_s[0], t_est +
    respuesta_s]; si son menos de respaldo_min_puntos, el diámetro crudo (todas las mediciones) en esa ventana; si tampoco
    alcanza, el crudo en todo lo posterior al estímulo. Con ellas se ajusta la sigmoide de la respuesta (ajustar_sigmoide):
    extremo = meseta del ajuste, velocidad máxima = a / (4 s), latencia = la tangente en t50 hasta el nivel de partida.
    Si el ajuste falla: extremo = percentil respaldo_percentil (luz ON) o 100 - respaldo_percentil (luz OFF) de las
    mediciones, velocidad = cambio hasta ese extremo / tiempo desde el fin de la partida, latencia = primera medición más
    allá de la mitad del cambio. Redilatación con luz ON: pendiente de una recta ajustada a las mediciones de los
    redilatacion_s posteriores al extremo. Solo queda NaN si después del estímulo no hay ninguna medición.
    """
    signo = -1 if tipo == "luz ON" else 1
    t0, t1 = t_est + p["partida_s"][0], t_est + p["respuesta_s"]
    n_min = p["respaldo_min_puntos"]
    fuente = None
    for nombre, x, a, b in (("curva", d, t0, t1), ("crudo", crudo, t0, t1), ("crudo", crudo, t0, t[-1])):
        sel = (t >= a) & (t <= b) & np.isfinite(x)
        if sel.sum() >= n_min:
            fuente, y, t1 = nombre, x, b
            break
    if fuente is None:
        sel = (t >= t0) & np.isfinite(crudo)
        if sel.sum() < 2:
            return
        fuente, y, t1 = "crudo", crudo, t[-1]
    tt, yy = t[sel], y[sel]
    dp = out["partida_mm"] if np.isfinite(out["partida_mm"]) else out["basal_mm"]
    f = ajustar_sigmoide(t, y, t0, t1, signo, {**p, "modelo_min_puntos": min(n_min, len(tt))})
    if f is not None and f["ok"] and np.isfinite(f["vel_max"]) and f["r2"] > 0:
        extremo, vmax = f["d0"] + f["a"], f["vel_max"]
        t_ext = min(f["t50"] + 2 * f["s"], t1)
        y50 = f["d0"] + f["a"] / 2
        lat = f["t50"] + ((dp if np.isfinite(dp) else f["d0"]) - y50) / vmax - t_est
        nombre = "constriccion" if tipo == "luz ON" else "redilatacion"
        out[f"ajuste_{nombre}_r2"] = f["r2"]
        out["_modelos"][nombre] = {**{c: f[c] for c in ("d0", "a", "t50", "s")}, "t0": t0, "t1": t1}
    else:
        q = p["respaldo_percentil"] if signo < 0 else 100 - p["respaldo_percentil"]
        extremo = float(np.percentile(yy, q))
        t_ext = float(tt[np.argmin(np.abs(yy - extremo))])
        ref = dp if np.isfinite(dp) else float(np.median(yy[:max(1, len(yy) // 5)]))
        vmax = (extremo - ref) / max(t_ext - (t_est + p["partida_s"][1]), float(np.median(np.diff(t))))
        mitad = signo * (yy - ref) >= signo * (extremo - ref) / 2
        lat = float(tt[np.argmax(mitad)] - t_est) if mitad.any() else t_ext - t_est
    out["extremo_mm"], out["t_extremo_s"], out["latencia_s"] = float(extremo), float(t_ext - t_est), float(lat)
    out["t_vel_max_s"] = np.nan
    d0, df_ = out["basal_mm"], out["final_mm"]
    if tipo == "luz ON":
        amp, amp_b = dp - extremo, d0 - extremo
        out["vel_max_constriccion_mm_s"] = out["vel_max_constriccion_derivada_mm_s"] = float(vmax)
        despues = (t > t_ext) & (t <= t_ext + p["redilatacion_s"]) & np.isfinite(y)
        if despues.sum() < 2:
            despues = (t > t_ext) & np.isfinite(y)
        if despues.sum() >= 2:
            out["vel_max_redilatacion_mm_s"] = float(np.polyfit(t[despues], y[despues], 1)[0])
            out["vel_max_redilatacion_derivada_mm_s"] = out["vel_max_redilatacion_mm_s"]
    else:
        fin = df_ if np.isfinite(df_) else extremo
        amp, amp_b = fin - dp, fin - d0
        out["vel_max_redilatacion_mm_s"] = out["vel_max_redilatacion_derivada_mm_s"] = float(vmax)
    out["amplitud_mm"], out["amplitud_desde_basal_mm"] = float(amp), float(amp_b)
    out["amplitud_pct"] = 100 * float(amp) / dp if np.isfinite(dp) and dp else np.nan
    out["amplitud_referencia"] = "partida" if np.isfinite(out["partida_mm"]) else "basal"
    for k in ("vel_max_constriccion", "vel_max_redilatacion"):
        if np.isfinite(out[f"{k}_mm_s"]):
            out[f"{k}_metodo"] = "respaldo"
    for k in ("extremo_mm", "t_extremo_s", "latencia_s", "amplitud_mm", "amplitud_pct", "amplitud_desde_basal_mm",
              "vel_max_constriccion_mm_s", "vel_max_redilatacion_mm_s", "vel_max_constriccion_derivada_mm_s",
              "vel_max_redilatacion_derivada_mm_s", "ajuste_constriccion_r2", "ajuste_redilatacion_r2"):
        if np.isfinite(out[k]):
            nc[k] = 1
    out["respaldo_fuente"] = fuente


def parametros(t, d_suave, vel, t_est, tipo, p, conf=None, crudo=None, d=None):
    """Parámetros del reflejo a partir de la curva suavizada (mm) y su velocidad (mm/s), con su confianza.

    p: sección `parametros`. Niveles (medianas de la curva suavizada), relativos al estímulo:
      basal (inicial): [t_est - basal_s, t_est);  final: los últimos final_s del registro;
      partida: [t_est + partida_s[0], t_est + partida_s[1]]. En ese intervalo la iluminación ya cambió pero la pupila
      todavía no reaccionó (la latencia del reflejo fotomotor es mayor que partida_s[1]). Cuando cambia la luz el
      diámetro medido salta en el mismo frame; `escalon_estimulo_mm` = partida - basal lo cuantifica.
    Respuesta: (t_est + partida_s[1], t_est + respuesta_s]: ahí se buscan el extremo y la velocidad máxima.
    Amplitud (principal): luz ON = partida - mínimo; luz OFF = final - partida; en mm y en % del nivel de partida. Se mide
    desde el nivel de partida y no desde el basal porque el salto del diámetro medido en el frame del estímulo es un sesgo
    de la medición que depende de la iluminación y persiste después (notebook 05: contra las anotaciones, el cambio medido
    desde la partida tiene error medio absoluto 0.19 mm y sesgo 0.00 mm; desde el basal, 0.39 mm, con sesgos de -0.38 y
    +0.52 mm en A7 y B5). amplitud_desde_basal_mm: la definición de la guía (basal - mínimo; final - inicial), como
    referencia. Si el nivel de partida tiene menor confianza que el basal (p. ej. un estímulo gradual o la transición a
    oscuridad sin luz de relleno), la amplitud principal se mide desde el basal; `amplitud_referencia` dice cuál se usó.
    Velocidad máxima de la fase de respuesta (principal): pendiente máxima a / (4 s) de una sigmoide logística ajustada a
    la curva limpia sin suavizar `d` (ajustar_sigmoide). Luz ON: constricción en [t_est + partida_s[0], instante del
    mínimo]; luz OFF: redilatación en [t_est + partida_s[0], t_est + respuesta_s]; la de constricción no corresponde (NaN).
    El modelo usa toda la transición en vez de la pendiente en un instante (no depende del corte del suavizado ni de un
    frame ruidoso), pero no corrige una transición medida con otra forma (un salto de la segmentación a mitad de la
    respuesta cambia la pendiente del modelo tanto como la de la derivada; notebook 05). La redilatación con luz ON (no es
    fase de respuesta) se mide solo por derivada.
    Por derivada (`vel_max_*_derivada_mm_s`, referencia): luz ON, mínimo de la velocidad de la curva suavizada antes del
    mínimo y máximo en los redilatacion_s posteriores; luz OFF, máximo en la respuesta. `vel_max_*_metodo` dice cuál quedó
    como principal: si el ajuste no converge, no hay muestras suficientes o R² <= 0, la derivada, con confianza baja. La
    confianza del modelo es la mínima de los frames de su ventana, y baja si R² < modelo_r2_min o si la transición no cabe
    en la ventana (t50 o s en el límite de su rango). `_modelos`: los ajustes (para dibujarlos).
    Latencia (criterio de inicio): la tangente a la curva en el instante de velocidad máxima de la respuesta se extiende
    hacia atrás hasta el nivel de partida; el inicio es esa intersección y la latencia es inicio - t_est. Usa el tramo más
    empinado, donde la relación señal/ruido es máxima, en lugar de la primera muestra que se aparta del nivel previo (que
    depende del ruido de cada video), y da un tiempo continuo (precisión menor a un frame). Referencia: latencia_umbral =
    primer cruce (interpolado entre frames) de partida +/- max(umbral_n_sigmas * desvío robusto del basal,
    umbral_frac_amplitud * amplitud).

    Confianza (degradación con elegancia: un parámetro solo es NaN si no hay ninguna forma de estimarlo).
    conf: nivel (3 alta, 2 media, 1 baja) de cada frame de la curva. La confianza de un nivel o de un tramo es el
    nivel mínimo de los frames que usa. Si una ventana (basal, partida, final) no tiene curva válida, se usa la
    mediana de `crudo` (diámetro medido en todos los frames, sin indicador) en esa ventana, con confianza baja. La
    latencia baja a confianza baja si la tangente y el umbral difieren en más de latencia_discrepancia_max_s o si cae
    fuera de latencia_rango_s; una velocidad, si supera el máximo fisiológico (vel_*_max_mm_s); la amplitud, si va en
    sentido contrario al esperado. Si el basal tiene menor confianza que el nivel de partida, el basal es el nivel de
    partida (confianza media como máximo). Si la ventana de respuesta no tiene curva (menos de 3 frames con curva suavizada
    y velocidad), los parámetros principales se estiman igual con las mediciones disponibles (_respaldo), con confianza
    baja y `*_metodo` = "respaldo". Devuelve (parámetros, {parámetro: "alta" | "media" | "baja"}).
    """
    t, y, v = (np.asarray(x, float) for x in (t, d_suave, vel))
    conf = np.full(len(t), 3) if conf is None else np.asarray(conf)
    crudo = np.full(len(t), np.nan) if crudo is None else np.asarray(crudo, float)
    signo = -1 if tipo == "luz ON" else 1          # sentido esperado del cambio
    nc = {}

    def nivel(sel, nombre):
        ok = sel & np.isfinite(y)
        if ok.any():
            nc[nombre] = int(conf[ok].min())
            return float(np.median(y[ok])), int(ok.sum())
        c = sel & np.isfinite(crudo)
        nc[nombre] = 1
        return (float(np.median(crudo[c])), 0) if c.any() else (np.nan, 0)
    en_basal = (t >= t_est - p["basal_s"]) & (t < t_est)
    d0, n0 = nivel(en_basal, "basal_mm")
    dp, _ = nivel((t >= t_est + p["partida_s"][0]) & (t <= t_est + p["partida_s"][1]), "partida_mm")
    df_, _ = nivel(t >= t[-1] - p["final_s"], "final_mm")
    # Basal sin curva confiable (p. ej. oscuridad sin luz de relleno): en la ventana de partida la pupila todavía tiene
    # su tamaño basal (no reaccionó) y ya hay luz, así que es mejor estimador que las mediciones crudas en oscuridad.
    # Confianza media como máximo: el salto de la medición en el estímulo queda incluido.
    if nc["basal_mm"] < min(nc["partida_mm"], 2) and np.isfinite(dp):
        d0, nc["basal_mm"] = dp, min(nc["partida_mm"], 2)
    nc["escalon_estimulo_mm"] = min(nc["basal_mm"], nc["partida_mm"])
    out = {"basal_mm": d0, "final_mm": df_, "partida_mm": dp, "escalon_estimulo_mm": dp - d0,
           "extremo_mm": np.nan, "t_extremo_s": np.nan, "amplitud_mm": np.nan, "amplitud_pct": np.nan,
           "amplitud_desde_basal_mm": np.nan, "amplitud_referencia": "", "latencia_s": np.nan, "latencia_umbral_s": np.nan,
           "vel_max_constriccion_mm_s": np.nan, "vel_max_redilatacion_mm_s": np.nan,
           "vel_max_constriccion_derivada_mm_s": np.nan, "vel_max_redilatacion_derivada_mm_s": np.nan,
           "vel_max_constriccion_metodo": "", "vel_max_redilatacion_metodo": "",
           "ajuste_constriccion_r2": np.nan, "ajuste_redilatacion_r2": np.nan, "_modelos": {}, "t_vel_max_s": np.nan,
           "respaldo_fuente": "",
           "fraccion_basal_valida": n0 / max(1, int(en_basal.sum()))}
    ir = np.flatnonzero((t > t_est + p["partida_s"][1]) & (t <= t_est + p["respuesta_s"]) & np.isfinite(y)
                        & np.isfinite(v))
    if len(ir) < 3:
        # Sin curva en la respuesta después de todos los niveles de confianza: respaldo con las mediciones que haya
        _respaldo(t, np.asarray(d if d is not None else y, float), crudo, t_est, tipo, p, out, nc)
        return out, {k: NIVELES[c] for k, c in nc.items()}
    c_resp = int(conf[ir].min())
    # Extremo de la respuesta y velocidad máxima en el sentido esperado (antes del extremo)
    ie = ir[np.argmax(signo * y[ir])]
    out["extremo_mm"], out["t_extremo_s"] = float(y[ie]), float(t[ie] - t_est)
    nc["extremo_mm"] = nc["t_extremo_s"] = c_resp
    antes = ir[ir <= ie]
    iv = antes[np.argmax(signo * v[antes])]
    vmax = float(v[iv])
    out["t_vel_max_s"] = float(t[iv] - t_est)
    nc["t_vel_max_s"] = c_resp
    if tipo == "luz ON":
        amp, amp_b = dp - out["extremo_mm"], d0 - out["extremo_mm"]
        out["vel_max_constriccion_derivada_mm_s"] = vmax
        nc["vel_max_constriccion_derivada_mm_s"] = c_resp
        despues = np.flatnonzero((t > t[ie]) & (t <= t[ie] + p["redilatacion_s"]) & np.isfinite(v))
        if len(despues):
            out["vel_max_redilatacion_derivada_mm_s"] = float(np.max(v[despues]))
            nc["vel_max_redilatacion_derivada_mm_s"] = int(conf[despues].min())
        nc["amplitud_mm"] = min(nc["partida_mm"], c_resp)
        nc["amplitud_desde_basal_mm"] = min(nc["basal_mm"], c_resp)
    else:
        amp, amp_b = df_ - dp, df_ - d0
        out["vel_max_redilatacion_derivada_mm_s"] = vmax
        nc["vel_max_redilatacion_derivada_mm_s"] = c_resp
        nc["amplitud_mm"] = min(nc["partida_mm"], nc["final_mm"])
        nc["amplitud_desde_basal_mm"] = min(nc["basal_mm"], nc["final_mm"])
    if not amp > 0:                    # la respuesta tiene que ir en el sentido del estímulo
        nc["amplitud_mm"] = 1
    if not amp_b > 0:
        nc["amplitud_desde_basal_mm"] = 1
    ref, d_ref = "partida", dp
    if nc["partida_mm"] < nc["basal_mm"] or not np.isfinite(dp):    # partida poco confiable: desde el basal
        ref, d_ref, amp, nc["amplitud_mm"] = "basal", d0, amp_b, nc["amplitud_desde_basal_mm"]
    out["amplitud_referencia"] = ref
    nc["amplitud_pct"] = nc["amplitud_mm"]
    # Velocidades por modelo: sigmoide ajustada a cada transición
    d = y if d is None else np.asarray(d, float)
    t_ext = float(t[ie])
    if tipo == "luz ON":
        ventanas = {"constriccion": (t_est + p["partida_s"][0], t_ext, -1)}
        # La redilatación con la luz encendida no es la fase de respuesta: es un escape lento y chico (~0.2-0.3 mm en 3 s,
        # del orden del ruido) que la sigmoide no describe (R² 0.3-0.8 en A6 y A7); queda la derivada
        out["vel_max_redilatacion_mm_s"] = out["vel_max_redilatacion_derivada_mm_s"]
        out["vel_max_redilatacion_metodo"] = "derivada"
        if "vel_max_redilatacion_derivada_mm_s" in nc:
            nc["vel_max_redilatacion_mm_s"] = nc["vel_max_redilatacion_derivada_mm_s"]
    else:
        ventanas = {"redilatacion": (t_est + p["partida_s"][0], t_est + p["respuesta_s"], 1)}
    for nombre, (t0, t1, sg) in ventanas.items():
        k, kd = f"vel_max_{nombre}_mm_s", f"vel_max_{nombre}_derivada_mm_s"
        f = ajustar_sigmoide(t, d, t0, t1, sg, p)
        # Falla: no converge o no explica más que una constante (R² <= 0: amplitud nula, p. ej. B4 en oscuridad)
        if f is not None and f["ok"] and np.isfinite(f["vel_max"]) and f["r2"] > 0:
            out[k], out[f"vel_max_{nombre}_metodo"], out[f"ajuste_{nombre}_r2"] = f["vel_max"], "modelo", f["r2"]
            nc[k] = 1 if f["en_borde"] or not f["r2"] >= p["modelo_r2_min"] else int(conf[f["sel"]].min())
            nc[f"ajuste_{nombre}_r2"] = nc[k]
            out["_modelos"][nombre] = {**{c: f[c] for c in ("d0", "a", "t50", "s")}, "t0": t0, "t1": t1}
        elif np.isfinite(out[kd]):                 # sin modelo: la derivada, con confianza baja
            out[k], out[f"vel_max_{nombre}_metodo"], nc[k] = out[kd], "derivada", 1
    # Velocidades fuera del rango fisiológico: confianza baja
    for nombre, lim in (("constriccion", p["vel_constriccion_max_mm_s"]), ("redilatacion", p["vel_redilatacion_max_mm_s"])):
        for k in (f"vel_max_{nombre}_mm_s", f"vel_max_{nombre}_derivada_mm_s"):
            if np.isfinite(out[k]) and abs(out[k]) > lim:
                nc[k] = 1
    out["amplitud_mm"], out["amplitud_pct"], out["amplitud_desde_basal_mm"] = amp, 100 * amp / d_ref, amp_b
    if not np.isfinite(dp):
        _completar(t, d if d is not None else y, crudo, t_est, tipo, p, out, nc)
        return out, {k: NIVELES[c] for k, c in nc.items()}
    # Latencia: tangente en la velocidad máxima hasta el nivel de partida
    if vmax * signo > 0:
        out["latencia_s"] = float(t[iv] + (dp - y[iv]) / vmax - t_est)
    # Latencia por umbral sobre el nivel de partida
    vb = y[en_basal & np.isfinite(y)]
    # el desvío del basal solo si el basal es de confianza alta (en oscuridad las mediciones son ruido y el umbral no se
    # cruzaría nunca); si no, solo la fracción de la amplitud
    sd = 1.4826 * float(np.median(np.abs(vb - d0))) if len(vb) and nc["basal_mm"] == 3 else 0.0
    umbral = max(p["umbral_n_sigmas"] * sd, p["umbral_frac_amplitud"] * abs(amp))
    t_u = _cruce(t, y, dp + signo * umbral, int(ir[0]), bajando=signo < 0)
    out["latencia_umbral_s"] = t_u - t_est if np.isfinite(t_u) else np.nan
    c_lat = min(nc["partida_mm"], c_resp)
    lat, lat_u = out["latencia_s"], out["latencia_umbral_s"]
    lo, hi = p["latencia_rango_s"]
    if not (np.isfinite(lat) and lo <= lat <= hi) or not np.isfinite(lat_u) \
            or abs(lat - lat_u) > p["latencia_discrepancia_max_s"]:
        c_lat = 1
    nc["latencia_s"] = nc["latencia_umbral_s"] = c_lat
    _completar(t, d if d is not None else y, crudo, t_est, tipo, p, out, nc)
    return out, {k: NIVELES[c] for k, c in nc.items()}


def _completar(t, d, crudo, t_est, tipo, p, out, nc):
    """Parámetros principales que quedaron NaN aunque hubo curva en la respuesta (confianza baja): latencia sin tangente
    válida (la velocidad máxima va al revés, p. ej. B4) = la de umbral o, si tampoco hay, el instante del extremo;
    redilatación con luz ON sin velocidad después del mínimo = pendiente de una recta ajustada a las mediciones (curva
    limpia o, si no hay, crudo) de los redilatacion_s posteriores al extremo."""
    if not np.isfinite(out["latencia_s"]):
        lat = out["latencia_umbral_s"] if np.isfinite(out["latencia_umbral_s"]) else out["t_extremo_s"]
        if np.isfinite(lat):
            out["latencia_s"], nc["latencia_s"] = float(lat), 1
    if tipo == "luz ON" and not np.isfinite(out["vel_max_redilatacion_mm_s"]) and np.isfinite(out["t_extremo_s"]):
        t_ext = t_est + out["t_extremo_s"]
        for x in (np.asarray(d, float), np.asarray(crudo, float)):
            sel = (t > t_ext) & (t <= t_ext + p["redilatacion_s"]) & np.isfinite(x)
            if sel.sum() >= 2:
                out["vel_max_redilatacion_mm_s"] = float(np.polyfit(t[sel], x[sel], 1)[0])
                out["vel_max_redilatacion_metodo"], nc["vel_max_redilatacion_mm_s"] = "respaldo", 1
                break


# ---------------------------------------------------------------------------------------------
# Curva completa
# ---------------------------------------------------------------------------------------------

def curva(df, info, cfg):
    """Curva frame a frame a partir de procesar_video: estímulo, unidades, limpieza, suavizado y parámetros.

    Degradación con elegancia: en cada fase (antes / después del estímulo) se aceptan los frames del nivel más alto
    que cubre al menos curva.fase_min_frac de la fase (nivel_por_fase). La columna `confianza` dice con qué nivel se
    aceptó cada frame (los interpolados heredan el menor de sus dos extremos).
    Devuelve (curva, params): curva es un DataFrame con tiempo, diámetro y área (px del frame original, mm, normalizados
    por el basal), indicador de calidad, confianza, estado del frame e interpolación, curva suavizada y velocidad;
    params, un dict con los parámetros fisiológicos, el instante del estímulo y la escala, más "_confianza" (confianza
    de cada parámetro).
    """
    pc, pp = cfg["curva"], cfg["parametros"]
    t = df.t.to_numpy()
    t_est, salto, tipo = estimulo(t, df.brillo.to_numpy())
    k_px, mm_roi = info["px_frame_por_px_roi"], info["mm_por_px_roi"]
    nv = niveles_frame(df, cfg["segmentacion"]["calidad"], cfg["segmentacion"]["calidad_media"])
    minimo, fases = nivel_por_fase(t, nv, t_est, pc["fase_min_frac"])
    medido = np.where(nv >= minimo, df.diametro_roi.to_numpy(float), np.nan)   # NaN si no alcanza el nivel de la fase
    d_mm, estado = limpiar_curva(t, medido * mm_roi, df.area_roi.to_numpy(float), pc, aceptada=nv == 3)
    conf = np.where(estado == "ok", nv, 0)
    for i0, i1 in _tramos(estado == "interpolado"):
        conf[i0:i1] = min(conf[i0 - 1], conf[i1]) if i0 > 0 and i1 < len(t) else 1
    suave, vel = suavizar(t, d_mm, pc["fc_hz"], pc["butter_orden"])
    crudo = df.diametro_roi.to_numpy(float) * mm_roi
    prm, conf_p = parametros(t, suave, vel, t_est, tipo, pp, conf, crudo, d_mm)
    d0 = prm["basal_mm"]
    # Área: los mismos frames válidos que el diámetro; en los interpolados, interpolación lineal en el tiempo
    a_mm = np.where(estado == "ok", df.area_roi.to_numpy(float) * mm_roi ** 2, np.nan)
    interp = estado == "interpolado"
    if interp.any() and np.isfinite(a_mm).sum() > 1:
        v = np.isfinite(a_mm)
        a_mm[interp] = np.interp(t[interp], t[v], a_mm[v])
    en_basal = (t >= t_est - pp["basal_s"]) & (t < t_est)
    a0 = np.nanmedian(a_mm[en_basal]) if np.isfinite(a_mm[en_basal]).any() else d0 ** 2 * np.pi / 4
    # Sigmoides ajustadas (velocidades por modelo), evaluadas en sus ventanas
    d_mod, v_mod = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for m in prm["_modelos"].values():
        w = (t >= m["t0"]) & (t <= m["t1"])
        d_mod[w] = sigmoide(t[w], m["d0"], m["a"], m["t50"], m["s"])
        e = np.exp(-np.clip((t[w] - m["t50"]) / m["s"], -50, 50))
        v_mod[w] = m["a"] / m["s"] * e / (1 + e) ** 2
    out = pd.DataFrame({
        "frame": df.frame, "t_s": t, "t_rel_estimulo_s": t - t_est,
        "diametro_px": d_mm / mm_roi * k_px, "diametro_mm": d_mm, "diametro_norm": d_mm / d0,
        "area_px": a_mm / mm_roi ** 2 * k_px ** 2, "area_mm2": a_mm, "area_norm": a_mm / a0,
        "diametro_crudo_mm": crudo,
        "calidad": df.confiable.astype(int), "confianza": [NIVELES[c] for c in conf],
        "inliers": df.inliers, "residuo": df.residuo,
        "circularidad": df.circularidad, "iris_detectado": df.iris.astype(int),
        "estado": estado, "interpolado": (estado == "interpolado").astype(int),
        "diametro_suave_mm": suave, "velocidad_mm_s": vel,
        "diametro_modelo_mm": d_mod, "velocidad_modelo_mm_s": v_mod,
    })
    principales = [k for k in ("amplitud_mm", "latencia_s", "vel_max_constriccion_mm_s", "vel_max_redilatacion_mm_s")
                   if np.isfinite(prm[k])]
    conf_p["confianza_global"] = NIVELES[min((3 if conf_p[k] == "alta" else 2 if conf_p[k] == "media" else 1)
                                             for k in principales)] if principales else "baja"
    params = {"video": None, "t_estimulo_s": t_est, "tipo_estimulo": tipo, "salto_brillo": salto,
              "mm_por_px": info["mm_por_px"], "iris_diametro_px": info["iris_diametro_px"],
              "fps_medio": (len(t) - 1) / (t[-1] - t[0]), "duracion_s": float(t[-1]),
              "frames_validos_pct": 100 * float(np.isfinite(d_mm).mean()),
              "frames_interpolados_pct": 100 * float((estado == "interpolado").mean()),
              "nivel_fase_pre": NIVELES[fases.get("pre", 0)], "nivel_fase_post": NIVELES[fases.get("post", 0)],
              "fc_hz": pc["fc_hz"], **prm, "confianza_global": conf_p["confianza_global"], "_confianza": conf_p}
    return out, params


UNIDADES = {"t_estimulo_s": "s", "tipo_estimulo": "", "salto_brillo": "niveles de gris", "mm_por_px": "mm/px",
            "iris_diametro_px": "px", "fps_medio": "Hz", "duracion_s": "s", "frames_validos_pct": "%",
            "frames_interpolados_pct": "%", "fc_hz": "Hz", "basal_mm": "mm", "final_mm": "mm", "extremo_mm": "mm",
            "partida_mm": "mm", "escalon_estimulo_mm": "mm", "amplitud_desde_basal_mm": "mm", "amplitud_referencia": "",
            "t_extremo_s": "s", "amplitud_mm": "mm", "amplitud_pct": "%", "latencia_s": "s",
            "latencia_umbral_s": "s", "vel_max_constriccion_mm_s": "mm/s", "vel_max_redilatacion_mm_s": "mm/s",
            "vel_max_constriccion_derivada_mm_s": "mm/s", "vel_max_redilatacion_derivada_mm_s": "mm/s",
            "vel_max_constriccion_metodo": "", "vel_max_redilatacion_metodo": "", "ajuste_constriccion_r2": "",
            "ajuste_redilatacion_r2": "", "respaldo_fuente": "",
            "t_vel_max_s": "s", "fraccion_basal_valida": "", "video": "", "nivel_fase_pre": "",
            "nivel_fase_post": "", "confianza_global": ""}


def tabla_parametros(params):
    """parametros.csv: una fila por parámetro (parametro, valor, unidad, confianza). La confianza solo se da para los
    parámetros medidos sobre la curva (alta, media o baja; ver parametros)."""
    conf = params.get("_confianza", {})
    return pd.DataFrame([{"parametro": k, "valor": v, "unidad": UNIDADES.get(k, ""),
                          "confianza": conf.get(k, "") if k != "confianza_global" else ""}
                         for k, v in params.items() if not k.startswith("_")])
