"""Módulo 2.2: morfología sobre la máscara de pupila, selección de la componente y ajuste de la elipse.

Cadena (postprocesar):
  1. cierre: une los fragmentos de la pupila cortados por reflejos o pestañas finas y cierra las entradas
     que deja un reflejo apoyado sobre el borde;
  2. relleno de huecos: los reflejos dentro de la pupila quedan como agujeros cerrados y se rellenan;
  3. apertura: borra pestañas, sombras finas y ruido, y desprende de la pupila las regiones oscuras unidas por
     un istmo (sombra del párpado, borde oscuro del iris);
  4. componente: de las componentes conexas que quedan, la más compatible con una pupila (circular y cercana al
     centro del iris, con un tamaño fisiológico); si `rellenar`, se rellenan también sus huecos; opcionalmente, se
     reemplaza por su
     envolvente convexa (la pupila es convexa: un reflejo apoyado sobre el borde deja una muesca que no es un
     hueco cerrado y el relleno no la recupera);
  5. esqueleto del contorno (curva de 1 px) y elipse ajustada a sus puntos -> diámetro.
Los elementos estructurantes son discos de diámetro expresado en diámetros de iris (invariancia de escala).
Erosión y dilatación se usan por separado en el notebook para ilustrar su efecto; apertura y cierre son sus
composiciones.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import convex_hull_image, skeletonize


def disco_ee(frac, iris_diam_px):
    """Elemento estructurante: disco de `frac` diámetros de iris (lado impar >= 3), o None si frac = 0."""
    if not frac:
        return None
    k = max(3, int(round(frac * iris_diam_px)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _op(mask, op, ee):
    if ee is None:
        return mask.copy()
    return cv2.morphologyEx(mask.astype(np.uint8), op, ee) > 0


def erosion(mask, ee):
    return _op(mask, cv2.MORPH_ERODE, ee)


def dilatacion(mask, ee):
    return _op(mask, cv2.MORPH_DILATE, ee)


def apertura(mask, ee):
    """Erosión seguida de dilatación: elimina lo más angosto que el elemento estructurante."""
    return _op(mask, cv2.MORPH_OPEN, ee)


def cierre(mask, ee):
    """Dilatación seguida de erosión: cierra huecos y entradas más angostos que el elemento estructurante."""
    return _op(mask, cv2.MORPH_CLOSE, ee)


def rellenar_huecos(mask):
    """Rellena los agujeros (regiones de fondo no conectadas con el borde de la imagen)."""
    return binary_fill_holes(mask)


def contorno(mask):
    """Borde interior de la máscara (píxeles de la máscara con algún vecino 4-conexo afuera)."""
    return mask & ~erosion(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))


def esqueleto(mask):
    """Esqueleto morfológico (eje medio adelgazado a 1 px)."""
    return skeletonize(mask)


def esqueleto_contorno(mask):
    """Esqueleto del contorno pupilar: el borde adelgazado a una curva de 1 px (para ajustar la elipse)."""
    return skeletonize(contorno(mask))


def componentes(mask, iris):
    """Propiedades de cada componente conexa: área, perímetro, circularidad 4*pi*A/P^2, distancia del centroide
    al centro del iris (en radios de iris) y diámetro equivalente (en diámetros de iris)."""
    cx, cy, r = iris
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        comp = lab == k
        x, y, w, h = stats[k, :4]   # el contorno se busca en la caja de la componente (más rápido)
        cont, _ = cv2.findContours((lab[y:y + h, x:x + w] == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        area = float(stats[k, cv2.CC_STAT_AREA])
        perim = max(cv2.arcLength(max(cont, key=len), True), 1e-6)
        out.append({"etiqueta": k, "mascara": comp, "area": area, "circularidad": min(1.0, 4 * np.pi * area / perim ** 2),
                    "distancia": float(np.hypot(cent[k, 0] - cx, cent[k, 1] - cy) / r),
                    "diametro": 2 * np.sqrt(area / np.pi) / (2 * r)})
    return out


def elegir_componente(mask, iris, p):
    """Componente más compatible con una pupila, o None.

    Se descartan las de diámetro equivalente fuera de [diam_min_iris, diam_max_iris] (fracción del diámetro del
    iris: la pupila mide entre ~2 y ~8 mm con un iris de 11.7 mm). Entre las restantes gana la de mayor
    puntaje = circularidad - peso_distancia * distancia al centro del iris (en radios de iris).
    """
    cand = [c for c in componentes(mask, iris) if p["diam_min_iris"] <= c["diametro"] <= p["diam_max_iris"]]
    if not cand:
        return None, []
    for c in cand:
        c["puntaje"] = c["circularidad"] - p["peso_distancia"] * c["distancia"]
    return max(cand, key=lambda c: c["puntaje"]), cand


def ajustar_elipse(mask):
    """Elipse por mínimos cuadrados (cv2.fitEllipse) a los puntos del esqueleto del contorno.

    Devuelve {cx, cy, a, b, angulo_deg} con a >= b, o None si hay menos de 5 puntos.
    """
    ys, xs = np.nonzero(esqueleto_contorno(mask))
    if len(xs) < 5:
        return None
    cv2.setRNGSeed(0)                                   # ver _elipse_cv
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(np.column_stack([xs, ys]).astype(np.float32))
    a, b = d1 / 2, d2 / 2
    if b > a:
        a, b, ang = b, a, ang + 90
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "angulo_deg": float(ang % 180)}


def postprocesar(mask, iris, p):
    """Cadena morfológica completa (ver docstring del módulo). p: sección modulo2.morfologia (una combinación).

    Devuelve dict con las máscaras intermedias (pasos), la componente elegida (pupila), las candidatas, la elipse
    (None si no hay pupila) y los esqueletos. Por velocidad se trabaja en el rectángulo que contiene la máscara,
    con un margen mayor que los elementos estructurantes; los resultados se devuelven en coordenadas de la imagen.
    """
    d_iris = 2 * iris[2]
    if not mask.any():
        return {"pasos": {"original": mask}, "pupila": None, "candidatas": [], "elipse": None}
    ys, xs = np.nonzero(mask)
    m_px = int(max(p["cierre_iris_frac"], p["apertura_iris_frac"]) * d_iris) + 2
    y0, x0 = max(ys.min() - m_px, 0), max(xs.min() - m_px, 0)
    y1, x1 = ys.max() + 1 + m_px, xs.max() + 1 + m_px
    sub = mask[y0:y1, x0:x1]
    iris_sub = (iris[0] - x0, iris[1] - y0, iris[2])

    def pegar(m):
        out = np.zeros(mask.shape, bool)
        out[y0:y1, x0:x1] = m
        return out

    pasos = {"original": sub}
    m = cierre(sub, disco_ee(p["cierre_iris_frac"], d_iris))
    pasos["cierre"] = m
    if p["rellenar"]:
        m = rellenar_huecos(m)
    pasos["relleno"] = m
    m = apertura(m, disco_ee(p["apertura_iris_frac"], d_iris))
    pasos["apertura"] = m
    comp, cand = elegir_componente(m, iris_sub, p)
    for c in cand:
        c["mascara"] = pegar(c["mascara"])
    out = {"pupila": None, "candidatas": cand, "elipse": None}
    if comp is not None:
        pup = comp["mascara"][y0:y1, x0:x1]
        if p["rellenar"]:
            pup = rellenar_huecos(pup)
        pasos["componente"] = pup
        if p["envolvente_convexa"]:
            pup = convex_hull_image(pup)
            pasos["envolvente"] = pup
        e = ajustar_elipse(pup)
        if e is not None:
            e["cx"], e["cy"] = e["cx"] + x0, e["cy"] + y0
        out.update(pupila=pegar(pup), elipse=e, esqueleto_contorno=pegar(esqueleto_contorno(pup)),
                   esqueleto=pegar(esqueleto(pup)))
    out["pasos"] = {k: pegar(v) for k, v in pasos.items()}
    return out


# ---------------------------------------------------------------------------------------------
# Ajuste robusto de la elipse (RANSAC) e indicador de calidad
# ---------------------------------------------------------------------------------------------

def distancia_elipse(pts, e):
    """Distancia radial [px] de cada punto (N x 2, x y) a la elipse e: |rho - rho_elipse| en la dirección del punto
    desde el centro. Para elipses casi circulares (pupilas) aproxima bien la distancia geométrica y es vectorizable."""
    th = np.radians(e["angulo_deg"])
    dx, dy = pts[:, 0] - e["cx"], pts[:, 1] - e["cy"]
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    rn = np.sqrt((u / e["a"]) ** 2 + (v / e["b"]) ** 2)
    rho = np.hypot(u, v)
    return rho * np.abs(1 - 1 / np.maximum(rn, 1e-9))


def _elipse_cv(pts):
    """Elipse de cv2.fitEllipse como dict {cx, cy, a, b, angulo_deg} con a >= b, o None si es degenerada.

    Si el ajuste queda mal condicionado (pocos puntos, como las muestras de 5 del RANSAC), fitEllipse perturba los
    puntos con el generador aleatorio global de OpenCV, cuyo estado avanza con cada llamada del proceso: sin fijar la
    semilla, el resultado de un frame depende de los frames que el mismo proceso de trabajo midió antes.
    """
    cv2.setRNGSeed(0)
    try:
        (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts.astype(np.float32))
    except cv2.error:
        return None
    a, b = d1 / 2, d2 / 2
    if b > a:
        a, b, ang = b, a, ang + 90
    if not (np.isfinite([cx, cy, a, b]).all() and b > 0):
        return None
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "angulo_deg": float(ang % 180)}


def ransac_elipse(pts, umbral_px, valida, iteraciones, rng):
    """Elipse robusta a puntos espurios (RANSAC): se ajustan elipses a muestras de 5 puntos al azar y se queda la que
    tiene más puntos a menos de `umbral_px` (inliers); la elipse final se reajusta con todos sus inliers.

    `valida(e)`: restricciones a una elipse candidata (tamaño y excentricidad de una pupila). Devuelve (elipse,
    máscara de inliers) o (None, None). Los derrames de la cuenca y los bordes ajenos a la pupila quedan como
    outliers mientras el arco de pupila visible sea el conjunto consistente más grande.
    """
    mejor, mejor_n = None, 0
    n = len(pts)
    for _ in range(iteraciones):
        e = _elipse_cv(pts[rng.choice(n, 5, replace=False)])
        if e is None or not valida(e):
            continue
        k = int((distancia_elipse(pts, e) < umbral_px).sum())
        if k > mejor_n:
            mejor, mejor_n = e, k
    if mejor is None:
        return None, None
    inl = distancia_elipse(pts, mejor) < umbral_px
    final = _elipse_cv(pts[inl]) if inl.sum() >= 5 else None
    if final is not None and valida(final):
        mejor = final
        inl = distancia_elipse(pts, mejor) < umbral_px
    return mejor, inl


def circularidad(mask):
    """4*pi*A/P^2 de la componente más grande (1 = círculo)."""
    cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return 0.0
    c = max(cont, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    return float(min(1.0, 4 * np.pi * cv2.contourArea(c) / per ** 2)) if per > 0 else 0.0


def ajuste_robusto(mask, excluir, iris, r_region, p_mm, p_ra, rng):
    """Elipse por RANSAC sobre el contorno de una máscara de pupila y medidas de calidad.

    1. relleno de huecos (p_mm['rellenar']) y componente más grande;
    2. puntos del esqueleto del contorno, sin los que tocan `excluir` (reflejos dilatados: ahí el borde es el del
       reflejo, no el de la pupila) ni los que están sobre el borde de la región de búsqueda (borde artificial);
    3. RANSAC (umbral en radios de iris) con elipses de tamaño admisible (multimarcador.diam_iris), b/a >= eje_min y
       cuyo interior esté cubierto por la máscara en al menos cobertura_min (una elipse que abarca el hueco de un
       derrame en forma de gancho no es una pupila; RANSAC, que maximiza los inliers, tiende a preferirla porque
       el contorno del derrame aporta muchos puntos).
    Calidad: fracción de inliers entre los puntos usados; residuo = RMS de la distancia de esos puntos a la elipse,
    truncada en residuo_trunc_iris, dividida por el radio equivalente de la elipse (un derrame lo sube);
    circularidad de la máscara; diámetro equivalente en diámetros de iris.
    """
    cx, cy, r = iris
    m = rellenar_huecos(mask) if p_mm["rellenar"] else mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    vacio = {"elipse": None, "inliers": np.nan, "residuo": np.nan, "circularidad": 0.0, "diametro_iris": np.nan,
             "n_puntos": 0, "mascara": m}
    if n < 2:
        return vacio
    m = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    vacio.update(mascara=m, circularidad=circularidad(m))
    ys, xs = np.nonzero(esqueleto_contorno(m))
    fuera = excluir[ys, xs] | (np.hypot(xs - cx, ys - cy) > (r_region - p_ra["margen_region_iris"]) * r)
    pts = np.column_stack([xs, ys])[~fuera].astype(np.float64)
    vacio["n_puntos"] = len(pts)
    if len(pts) < p_ra["min_puntos"]:
        return vacio
    d_min, d_max = p_mm["diam_iris"]

    def valida(e):
        if not (e["b"] / e["a"] >= p_ra["eje_min"] and d_min <= np.sqrt(e["a"] * e["b"]) / r <= d_max):
            return False
        if p_ra["cobertura_min"] <= 0:
            return True
        lienzo = np.zeros(m.shape, np.uint8)
        cv2.ellipse(lienzo, ((e["cx"], e["cy"]), (2 * e["a"], 2 * e["b"]), e["angulo_deg"]), 1, -1)
        adentro = lienzo > 0
        return adentro.any() and m[adentro].mean() >= p_ra["cobertura_min"]

    e, inl = ransac_elipse(pts, p_ra["umbral_iris"] * r, valida, p_ra["iteraciones"], rng)
    if e is None:
        return vacio
    r_eq = np.sqrt(e["a"] * e["b"])
    d = np.minimum(distancia_elipse(pts, e), p_ra["residuo_trunc_iris"] * r)
    vacio.update(elipse=e, inliers=float(inl.mean()), residuo=float(np.sqrt(np.mean(d ** 2)) / r_eq),
                 diametro_iris=float(r_eq / r))
    return vacio


def es_confiable(q, p_cal):
    """Indicador de calidad: True si la medición es confiable (si no, el diámetro del frame es NaN).

    q: medidas de ajuste_robusto; p_cal: umbrales (inliers_min, residuo_max, circularidad_min) y rango de tamaño
    (diam_iris_min, diam_iris_max, en diámetros de iris)."""
    if q["elipse"] is None:
        return False
    return (q["inliers"] >= p_cal["inliers_min"] and q["residuo"] <= p_cal["residuo_max"]
            and q["circularidad"] >= p_cal["circularidad_min"]
            and p_cal["diam_iris_min"] <= q["diametro_iris"] <= p_cal["diam_iris_max"])
