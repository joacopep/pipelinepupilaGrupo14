"""Detección automática del ojo y normalización de la ROI a tamaño fijo.

Estrategia en cascada, para servir tanto en tomas de cara completa como de ojo en primer plano:
  1. Cara (Haar) -> ojos (Haar) en la parte superior de la cara. Si no hay cara derecha, se prueba con el
     frame rotado (celular de costado) y las cajas se llevan a las coordenadas originales.
  2. Sin cara, o sin iris en el ojo de la cara: ojos (Haar) en todo el frame, también rotado.
  3. Sin ojos: se asume que el ojo ocupa el frame (protocolo de primer plano) y se usa el cuadrado central, junto con
     ventanas más chicas repartidas por el frame (iris chico y descentrado, p. ej. media cara con la mirada desviada).
En cada caja candidata se proponen círculos con la transformada de Hough y se elige el de mayor contraste
radial en el borde (operador integro-diferencial de Daugman, sobre los arcos laterales).
La ROI es un cuadrado centrado en el iris cuyo lado es un múltiplo fijo del diámetro del iris, de modo que tras
normalizar a `size` px el iris ocupa siempre la misma fracción de la ROI (invariancia de escala) y sirve de
referencia en mm.

En video la escala es fija: `init_video_roi` toma la mediana del diámetro del iris en los primeros
`scale_window_s` segundos, y luego `detect_eye_roi(frame, c, prev)` solo actualiza el centro (seguimiento dentro de
la ROI previa, o detección completa si se pierde). Así el lado de la ROI no hereda el ruido de Hough frame a frame.

Rendimiento: toda la detección (cara, ojos, iris y seguimiento) se hace sobre una copia del frame reducida a
`deteccion_lado_max` px de lado mayor; la ROI se expresa en coordenadas del frame original y `crop_roi` la recorta
de él a resolución completa.

Todos los parámetros vienen de la sección `roi` de config.yaml y se expresan como fracciones
(del frame, de la cara o de la caja del ojo), nunca en píxeles absolutos del frame original.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import cv2
import numpy as np

_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_EYE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

Box = Tuple[int, int, int, int]  # (x, y, ancho, alto) en píxeles del frame original


@dataclass
class EyeROI:
    """ROI cuadrada del ojo en coordenadas del frame original."""
    cx: float                 # centro de la ROI (= centro del iris si se encontró)
    cy: float
    side: float               # lado de la ROI en px del frame original
    size: int                 # lado de la ROI normalizada en px
    metodo: str               # dónde se encontró el iris: "seguimiento" (ROI previa), "cara+ojo", "ojo" o "frame"
    eye_box: Box
    face_box: Optional[Box] = None
    iris_r: Optional[float] = None   # radio del iris de referencia (fija la escala) en px del frame original
    iris_r_frame: Optional[float] = None  # radio del iris medido en este frame (None si no se encontró)
    rot: int = 0              # rotación [grados antihorarios] en la que el ojo aparece derecho (párpados arriba/abajo)
    t: Optional[float] = None  # tiempo [s] del frame en que se midió (para limitar el movimiento entre frames)

    @property
    def scale(self):
        """Factor px ROI normalizada / px frame original."""
        return self.size / self.side

    @property
    def box(self):
        """Caja (x, y, lado, lado) de la ROI en el frame original (puede salirse del frame)."""
        return (int(round(self.cx - self.side / 2)), int(round(self.cy - self.side / 2)),
                int(round(self.side)), int(round(self.side)))

    def mm_per_px(self, iris_diameter_mm):
        """mm por píxel del frame original usando el iris como referencia; None si no hay iris."""
        return iris_diameter_mm / (2 * self.iris_r) if self.iris_r else None


def _gray_eq(img):
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def _detect_face(frame, c):
    """Cara más grande del frame (o None). Se busca sobre una copia reducida a detect_width."""
    s = c["detect_width"] / frame.shape[1]
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    min_side = int(c["face_min_frac"] * small.shape[1])
    faces = _FACE.detectMultiScale(_gray_eq(small), c["cascade_scale_factor"], c["cascade_min_neighbors"],
                                   minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return tuple(int(round(v / s)) for v in (x, y, w, h))


def _detect_eyes(img, ref, min_frac, max_frac, c):
    """Ojos en img con tamaño entre min_frac y max_frac de `ref` px. Devuelve cajas en coords de img."""
    # Se reduce para que el ojo más grande posible mida ~detect_width/2 px: acota el costo sin perder detalle
    s = min(1.0, c["detect_width"] / (2 * max_frac * ref))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    lo, hi = int(min_frac * ref * s), int(max_frac * ref * s)
    eyes = _EYE.detectMultiScale(_gray_eq(small), c["cascade_scale_factor"], c["cascade_min_neighbors"],
                                 minSize=(lo, lo), maxSize=(hi, hi))
    return [tuple(int(round(v / s)) for v in e) for e in eyes]


def _choose_eye(eyes, c, prev=None):
    """Elige un ojo. Con una ROI previa, el más cercano a ella (mantiene el mismo ojo a lo largo del video).

    Sin ROI previa, según config: 'izquierdo'/'derecho' = lado en la imagen; 'auto' = la caja más grande.
    """
    if prev is not None:
        return min(eyes, key=lambda e: np.hypot(e[0] + e[2] / 2 - prev.cx, e[1] + e[3] / 2 - prev.cy))
    if c["eye"] == "auto":
        return max(eyes, key=lambda e: e[2] * e[3])
    eyes = sorted(eyes, key=lambda e: e[0] + e[2] / 2)
    return eyes[0] if c["eye"] == "izquierdo" else eyes[-1]


def _iris_channel(img, c):
    return img[:, :, 2] if c["iris_channel"] == "R" else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def iris_candidates(frame, box, radius_frac, c):
    """Círculos candidatos a iris [(cx, cy, r), ...] en px del frame original dentro de `box`.

    La caja se normaliza a iris_work_size px antes de aplicar Hough, así que el suavizado, los umbrales
    del acumulador y el rango de radios son independientes de la resolución del video.
    """
    x, y, w, h = box
    x0, y0 = max(x, 0), max(y, 0)
    crop = frame[y0:y + h, x0:x + w]
    if crop.size == 0:
        return []
    S = c["iris_work_size"]
    work = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
    canal = cv2.GaussianBlur(_iris_channel(work, c), (0, 0), c["iris_blur_sigma_frac"] * S)
    circles = cv2.HoughCircles(canal, cv2.HOUGH_GRADIENT, dp=1, minDist=c["hough_min_dist_frac"] * S,
                               param1=c["hough_canny_high"], param2=c["hough_acc_threshold"],
                               minRadius=int(radius_frac[0] * S), maxRadius=int(radius_frac[1] * S))
    if circles is None:
        return []
    sx, sy = crop.shape[1] / S, crop.shape[0] / S
    return [(x0 + cx * sx, y0 + cy * sy, r * (sx + sy) / 2)
            for cx, cy, r in circles[0, :c["hough_max_candidates"]]]


def iris_score(canal, cx, cy, r, c, rot=0, sin_reflejos=False):
    """Contraste radial a través del borde del círculo (operador integro-diferencial de Daugman).

    Diferencia entre la intensidad media justo afuera (r*(1+delta)) y justo adentro (r*(1-delta)),
    muestreada solo en los arcos laterales (+-arc alrededor de la horizontal), porque arriba y abajo
    el iris suele estar tapado por los párpados. El borde iris/esclera da un valor alto y positivo
    (iris oscuro, esclera clara). Está en niveles de gris, así que no depende de la escala.
    `rot`: rotación del ojo en el frame (celular de costado); los arcos se giran con él para seguir
    cayendo sobre la esclera y no sobre los párpados.
    sin_reflejos (seguimiento): en cada anillo se descartan las muestras de reflejo, saturadas (>= iris_reflejo_nivel) o
    más claras que la mediana del anillo + iris_reflejo_k_mad MAD. Un reflejo de la lámpara junto a la pupila sube la
    media del anillo interno de un círculo bien puesto y puede hacer ganar a un círculo corrido (B7). En la detección
    inicial no se usa: cambia qué candidato gana en algunos frames y con eso el consenso de la escala (A8 pasaba de un
    iris de 279 px a un círculo de 900 px).
    """
    a = np.deg2rad(np.linspace(-c["iris_score_arc_deg"], c["iris_score_arc_deg"], 31))
    ang = np.concatenate([a, np.pi + a]) - np.deg2rad(rot)  # y crece hacia abajo: antihorario = ángulo negativo
    d = c["iris_score_delta"]
    H, W = canal.shape

    def media_en(radio):
        xs = np.clip(cx + radio * np.cos(ang), 0, W - 1).astype(np.float32)
        ys = np.clip(cy + radio * np.sin(ang), 0, H - 1).astype(np.float32)
        v = cv2.remap(canal, xs[None], ys[None], cv2.INTER_LINEAR).ravel()
        if not sin_reflejos:
            return float(v.mean())
        ok = v < c["iris_reflejo_nivel"]
        if ok.sum() >= 3:
            med = np.median(v[ok])
            ok &= v <= med + c["iris_reflejo_k_mad"] * max(1.4826 * np.median(np.abs(v[ok] - med)), 1.0)
        return float(v[ok].mean()) if ok.any() else float(v.mean())

    return float(media_en(r * (1 + d)) - media_en(r * (1 - d)))


def find_iris(frame, boxes, c, accept=None, sin_reflejos=False):
    """Mejor círculo de iris entre las cajas candidatas [(box, radius_frac[, rot]), ...].

    Devuelve ((cx, cy, r), indice_de_caja) o (None, None). Cada caja aporta hasta hough_max_candidates
    círculos de Hough y se elige el de mayor contraste radial (iris_score). `accept(cx, cy, r)` filtra
    opcionalmente los círculos admisibles (p. ej., compatibles con la ROI previa).
    """
    canal = _iris_channel(frame, c).astype(np.float32)
    best, best_score, best_i = None, -np.inf, None
    for i, (box, radius_frac, *resto) in enumerate(boxes):
        rot = resto[0] if resto else 0
        for circ in iris_candidates(frame, box, radius_frac, c):
            if accept is not None and not accept(*circ):
                continue
            sc = iris_score(canal, *circ, c, rot, sin_reflejos)
            if sc > best_score:
                best, best_score, best_i = circ, sc, i
    return best, best_i


def _salto_max(prev, t, c):
    """Desplazamiento máximo del centro [radios de iris] desde `prev` hasta el tiempo t: velocidad típica del ojo
    (seguimiento_vel_max_radios_s) por el tiempo transcurrido, con un mínimo (seguimiento_salto_min_radios). Como el
    tiempo se cuenta desde la última ROI con iris, el límite crece mientras no se encuentra el iris y el seguimiento se
    recupera. None (sin límite) si no se conocen los tiempos."""
    if t is None or prev is None or prev.t is None or not np.isfinite(c["seguimiento_vel_max_radios_s"]):
        return None
    # valor absoluto: en el primer segundo la ROI de referencia (init_video_roi) puede ser de un frame posterior
    return max(c["seguimiento_salto_min_radios"], c["seguimiento_vel_max_radios_s"] * abs(t - prev.t))


def _track_iris(frame, prev, c, salto=None):
    """Busca el iris en la ROI previa, con el radio acotado a +-track_radius_tol del radio de referencia y el centro a
    menos de `salto` radios de iris del anterior (si se da).

    Solo se actualiza el centro: el lado de la ROI y el radio de referencia se heredan de `prev`.
    """
    rel = prev.iris_r / prev.side
    tol = c["track_radius_tol"]

    def cerca(x, y, r):
        return np.hypot(x - prev.cx, y - prev.cy) <= salto * prev.iris_r
    iris, _ = find_iris(frame, [(prev.box, (rel * (1 - tol), rel * (1 + tol)), prev.rot)], c,
                        None if salto is None else cerca, sin_reflejos=True)
    if iris is None:
        return None
    cx, cy, r = iris
    return replace(prev, cx=cx, cy=cy, metodo="seguimiento", eye_box=prev.box, face_box=None, iris_r_frame=r)


def _compatible(prev, c, salto=None):
    """Filtro de círculos compatibles con la ROI previa: centro a menos de redetect_max_jump radios de
    iris (y de `salto`, si se da) y radio dentro de +-track_radius_tol del radio de referencia."""
    lim = c["redetect_max_jump"] if salto is None else min(c["redetect_max_jump"], salto)

    def accept(cx, cy, r):
        return (np.hypot(cx - prev.cx, cy - prev.cy) <= lim * prev.iris_r
                and abs(r / prev.iris_r - 1) <= c["track_radius_tol"])
    return accept


def _reduce(frame, c):
    """Copia del frame reducida a deteccion_lado_max px de lado mayor (sin ampliar) y el factor usado."""
    s = min(1.0, c["deteccion_lado_max"] / max(frame.shape[:2]))
    if s == 1.0:
        return frame, 1.0
    return cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def _scale_roi(roi, s):
    """Lleva una EyeROI a otra escala de coordenadas (s = factor nuevo / actual). `size` no cambia."""
    if roi is None or s == 1.0:
        return roi

    def caja(b):
        return tuple(int(round(v * s)) for v in b) if b is not None else None
    return replace(roi, cx=roi.cx * s, cy=roi.cy * s, side=roi.side * s,
                   iris_r=roi.iris_r * s if roi.iris_r else None,
                   iris_r_frame=roi.iris_r_frame * s if roi.iris_r_frame else None,
                   eye_box=caja(roi.eye_box), face_box=caja(roi.face_box))


def detect_eye_roi(frame, c, prev=None, t=None):
    """Detecta el ojo en un frame BGR y devuelve un EyeROI en coordenadas del frame original.

    La detección se hace sobre una copia reducida (ver _reduce); `prev` y el resultado están en
    coordenadas del frame original. t: tiempo del frame [s]; con él (y prev.t) se limita cuánto puede moverse el centro
    (_salto_max). Detalle del algoritmo en _detect_eye_roi.
    """
    small, s = _reduce(frame, c)
    return _scale_roi(_detect_eye_roi(small, c, _scale_roi(prev, s), t), 1 / s)


def _detect_eye_roi(frame, c, prev=None, t=None):
    """Detecta el ojo en un frame BGR y devuelve un EyeROI (iris_r_frame=None si no se encontró el iris).

    `c` es la sección `roi` del config. Sin `prev` (frame aislado) hace la detección completa. Con `prev`
    (ROI de un frame anterior del mismo video):
      1. busca el iris dentro de la ROI previa (seguimiento);
      2. si no aparece, hace la detección completa aceptando solo círculos compatibles con `prev`
         (cerca y del mismo tamaño), para no saltar al otro ojo ni a un círculo espurio;
      3. si tampoco, devuelve la ROI previa con iris_r_frame=None (frame sin iris: se marca y se sigue).
    Con `prev` la escala no cambia: el lado de la ROI y el radio de referencia son los de `prev`.
    """
    if prev is None or not prev.iris_r:
        return replace(_detect_full(frame, c), t=t)
    salto = _salto_max(prev, t, c)
    tracked = _track_iris(frame, prev, c, salto)
    if tracked is not None:
        return replace(tracked, t=t)
    roi = _detect_full(frame, c, prev, accept=_compatible(prev, c, salto), sin_reflejos=True)
    if roi is None or roi.iris_r_frame is None:
        return replace(prev, metodo="sin iris", face_box=None, iris_r_frame=None)   # conserva el t de la última con iris
    return replace(roi, side=prev.side, iris_r=prev.iris_r, t=t)


def _unrotate_box(box, k, W, H):
    """Caja (x, y, w, h) en el frame rotado k*90° antihorario (np.rot90) -> caja en el frame original WxH."""
    x, y, w, h = box

    def punto(xr, yr):
        if k == 1:
            return W - 1 - yr, xr
        if k == 2:
            return W - 1 - xr, H - 1 - yr
        if k == 3:
            return yr, H - 1 - xr
        return xr, yr
    xs, ys = zip(punto(x, y), punto(x + w, y + h))
    return (int(min(xs)), int(min(ys)), int(abs(xs[1] - xs[0])), int(abs(ys[1] - ys[0])))


def _face_and_eyes(frame, c):
    """Cara y ojos (cajas en coords del frame original), probando las rotaciones de c["rotaciones"].

    Los clasificadores Haar solo detectan caras derechas; si el celular se sostuvo de costado la cara queda
    rotada en el video. Se usa la primera rotación (en grados antihorarios, múltiplos de 90) en la que
    aparece una cara. Devuelve (cara, ojos, grados) o (None, [], 0) si en ninguna rotación hay cara.
    """
    H, W = frame.shape[:2]
    for grados in c["rotaciones"]:
        k = (grados // 90) % 4
        rot = np.ascontiguousarray(np.rot90(frame, k)) if k else frame
        face = _detect_face(rot, c)
        if face is None:
            continue
        fx, fy, fw, fh = face
        sub = rot[fy:fy + int(c["eye_search_height_frac"] * fh), fx:fx + fw]
        eyes = [(fx + ex, fy + ey, ew, eh)
                for ex, ey, ew, eh in _detect_eyes(sub, fw, c["eye_min_frac"], c["eye_max_frac"], c)]
        return _unrotate_box(face, k, W, H), [_unrotate_box(e, k, W, H) for e in eyes], grados
    return None, [], 0


def _eyes_without_face(frame, c):
    """Cajas de ojo buscadas en todo el frame (sin cara), en todas las rotaciones de c["rotaciones"].

    En primer plano o con el celular de costado no hay cara detectable y el ojo puede estar rotado; el
    clasificador de ojos tampoco es invariante a la rotación. Se juntan las cajas de todas las rotaciones
    (en coords del frame original) con la rotación en que se encontraron, y el contraste del iris decide.
    Devuelve [(caja, grados), ...].
    """
    H, W = frame.shape[:2]
    L = min(H, W)
    cajas = []
    for grados in c["rotaciones"]:
        k = (grados // 90) % 4
        rot = np.ascontiguousarray(np.rot90(frame, k)) if k else frame
        cajas += [(_unrotate_box(e, k, W, H), grados)
                  for e in _detect_eyes(rot, L, c["eye_alone_min_frac"], 1.0, c)]
    return cajas


def _ventanas(W, H, lados_frac):
    """Último recurso sin caja de ojo: ventanas cuadradas de lado lados_frac * lado corto del frame, repartidas por todo
    el frame con paso de medio lado (la última fila y columna pegadas al borde). Devuelve [(x, y, lado, lado), ...]."""
    L = min(H, W)
    cajas = []
    for f in lados_frac:
        w = int(round(f * L))
        paso = max(1, w // 2)
        xs = sorted(set(list(range(0, W - w + 1, paso)) + [W - w]))
        ys = sorted(set(list(range(0, H - w + 1, paso)) + [H - w]))
        cajas += [(x, y, w, w) for y in ys for x in xs]
    return cajas


def _detect_full(frame, c, prev=None, accept=None, sin_reflejos=False):
    """Detección completa (cara -> ojo -> iris) con la escala medida en este frame.

    Primero se busca el iris en el ojo elegido de la cara; si no hay cara, ni ojos, ni iris en ese ojo
    (p. ej. un falso positivo de cara), se pasa a los candidatos sin cara: ojos buscados en todo el frame
    y el último recurso, y decide el contraste del iris. Último recurso: el cuadrado central del frame (primer plano,
    iris grande) y, si Haar no encontró ninguna caja de ojo, ventanas más chicas repartidas por todo el frame
    (`ventanas_lado_frac`), donde el iris se busca con el rango de radios de una caja de ojo: encuentran un iris chico y
    descentrado cuando Haar no ve ni la cara ni el ojo (media cara con la mirada desviada, B6). En el cuadrado central un
    iris así no es admisible y gana un círculo grande alrededor del ojo; en una ventana el iris verdadero tiene un
    contraste radial 2-3 veces mayor.
    """
    H, W = frame.shape[:2]
    L = min(H, W)
    face, eyes, rot_cara = _face_and_eyes(frame, c)
    rot_frame = prev.rot if prev is not None else 0
    intentos = []
    if eyes:
        intentos.append([(_choose_eye(eyes, c, prev), c["iris_radius_frac"], "cara+ojo", rot_cara)])
    sin_cara = [(e, c["iris_radius_frac"], "ojo", g) for e, g in _eyes_without_face(frame, c)]
    hay_ojo = bool(eyes or sin_cara)
    sin_cara.append((((W - L) // 2, (H - L) // 2, L, L), c["iris_radius_frac_frame"], "frame", rot_frame))
    if not hay_ojo:      # las ventanas solo cuando Haar no ve ningún ojo: donde lo ve, la detección no cambia
        sin_cara += [(b, c["iris_radius_frac"], "frame", rot_frame) for b in _ventanas(W, H, c["ventanas_lado_frac"])]
    intentos.append(sin_cara)

    for cajas in intentos:
        iris, i = find_iris(frame, [(b, rf, g) for b, rf, _, g in cajas], c, accept, sin_reflejos)
        if iris is not None:
            eye_box, _, metodo, rot = cajas[i]
            cx, cy, r = iris
            return EyeROI(cx=cx, cy=cy, side=c["side_iris_diameters"] * 2 * r, size=c["size"], metodo=metodo,
                          eye_box=eye_box, face_box=face, iris_r=r, iris_r_frame=r, rot=rot)
    # Sin iris: ROI centrada en la mejor caja disponible, sin escala de iris
    eye_box, _, metodo, rot = intentos[0][0]
    ex, ey, ew, eh = eye_box
    return EyeROI(cx=ex + ew / 2, cy=ey + eh / 2, side=max(ew, eh), size=c["size"], metodo=metodo,
                  eye_box=eye_box, face_box=face, iris_r=None, iris_r_frame=None, rot=rot)


def init_video_roi(path, c):
    """ROI inicial de un video con escala fija, por consenso temporal en los primeros scale_window_s.

    En cada frame de la ventana se hace la detección completa, independiente de los demás frames. El iris
    verdadero aparece en el mismo lugar y con el mismo tamaño en muchos frames; los círculos espurios
    cambian de un frame a otro. Para cada detección se cuentan las que la confirman (centro a menos de
    init_consensus_dist radios y radio dentro de +-track_radius_tol) y se elige la más confirmada, dando
    prioridad a las que vienen de una caja de ojo (ver init_min_support).
    La escala es la mediana del radio entre sus confirmaciones. Devuelve la ROI de la última confirmación,
    con iris_r = esa mediana, o None si en la ventana no se encontró el iris.
    """
    from .io_utils import iter_frames
    rois = []
    for t, frame in iter_frames(path):
        if t > c["scale_window_s"]:
            break
        small, s = _reduce(frame, c)
        r = _scale_roi(_detect_full(small, c), 1 / s)
        if r is not None and r.iris_r_frame is not None:
            rois.append(replace(r, t=t))
    if not rois:
        return None

    def confirman(a):
        return [b for b in rois
                if np.hypot(b.cx - a.cx, b.cy - a.cy) <= c["init_consensus_dist"] * a.iris_r_frame
                and abs(b.iris_r_frame / a.iris_r_frame - 1) <= c["track_radius_tol"]]
    # Las detecciones de una caja de ojo tienen prioridad sobre las del cuadrado central del frame, que es un
    # último recurso: en una escena quieta un círculo espurio del cuadrado central también se repite en
    # todos los frames y ganaría el consenso. Se usan si alguna reúne al menos init_min_support de la ventana.
    grupos = [confirman(a) for a in rois]
    de_ojo = [g for g, a in zip(grupos, rois)
              if a.metodo != "frame" and len(g) >= c["init_min_support"] * len(rois)]
    grupo = max(de_ojo or grupos, key=len)
    r_med = float(np.median([b.iris_r_frame for b in grupo]))
    return replace(grupo[-1], iris_r=r_med, side=c["side_iris_diameters"] * 2 * r_med)


def crop_roi(frame, roi, normalize=True):
    """Recorta la ROI del frame. Si sale del frame se rellena con ceros (no se desplaza el centro).

    normalize=True: redimensiona a roi.size x roi.size (la imagen sobre la que trabaja el pipeline).
    normalize=False: devuelve el recorte a la resolución nativa del frame.
    """
    x, y, L, _ = roi.box
    H, W = frame.shape[:2]
    pad = max(0, -x, -y, x + L - W, y + L - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x, y = x + pad, y + pad
    crop = frame[y:y + L, x:x + L]
    if not normalize:
        return crop
    interp = cv2.INTER_AREA if L > roi.size else cv2.INTER_LINEAR
    return cv2.resize(crop, (roi.size, roi.size), interpolation=interp)


def draw_detection(frame, roi, thickness_frac=0.003):
    """Copia del frame con cara (verde), caja del ojo (azul), iris (amarillo) y ROI (rojo)."""
    vis = frame.copy()
    t = max(1, int(thickness_frac * max(frame.shape[:2])))
    if roi.face_box is not None:
        x, y, w, h = roi.face_box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), t)
    x, y, w, h = roi.eye_box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 120, 0), t)
    if roi.iris_r:
        cv2.circle(vis, (int(roi.cx), int(roi.cy)), int(roi.iris_r), (0, 220, 255), t)
    x, y, L, _ = roi.box
    cv2.rectangle(vis, (x, y), (x + L, y + L), (0, 0, 255), t)
    return vis
