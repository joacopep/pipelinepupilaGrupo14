"""Segmentación de la pupila elegida en el Módulo 2 (pipeline) e indicador de calidad.

Sobre la ROI normalizada (iris centrado, `roi.size` px):
  1. preprocesamiento del Módulo 1 (src/preprocess.py): canal R, estiramiento con percentiles del iris, mediana;
  2. watershed por marcadores sobre la magnitud del gradiente (Sobel), dentro de un disco concéntrico con el iris.
     Marcadores: pupila = disco alrededor del punto más oscuro cerca del centro del iris; reflejos especulares
     (mediana + k MAD sobre la imagen sin mediana) como marcador propio; fondo = fuera del disco + píxeles claros;
  3. relleno de huecos, componente más compatible con una pupila (circular y centrada), relleno, envolvente convexa;
  4. elipse ajustada al esqueleto del contorno -> diámetro equivalente 2 sqrt(a b).
Indicador de calidad: elipse por RANSAC sobre los puntos del contorno de la componente (antes de la envolvente, para
que un derrame se note), sin los puntos pegados a reflejos ni al borde de la región. Fracción de inliers, residuo,
circularidad y tamaño respecto del iris; si alguno no cumple su umbral, la medición no es confiable (NaN).
Continuidad temporal (opcional, `previa`): con la última pupila aceptada, el marcador se busca cerca de ella y el watershed
se restringe a un rango de radios razonable (restriccion_previa), para que no salte a un derrame.

Todos los tamaños se expresan en radios (*_r_iris, *_iris) o diámetros (*_iris_frac) de iris de la ROI, así que
no dependen de la cámara. Parámetros: sección `segmentacion` de config.yaml.
"""
import cv2
import numpy as np
from skimage.morphology import convex_hull_image

from .morfologia import ajustar_elipse, ajuste_robusto, elegir_componente, es_confiable, rellenar_huecos
from .preprocess import disco, preprocesar


def iris_en_roi(size, side_iris_diameters):
    """Círculo del iris (cx, cy, r) en la ROI normalizada: centrado y con radio fijo por construcción."""
    return size / 2, size / 2, size / side_iris_diameters / 2


def region_busqueda(shape, iris, r_iris):
    """Disco concéntrico con el iris, de radio r_iris radios de iris, donde se busca la pupila."""
    cx, cy, r = iris
    return disco(shape, cx, cy, r_iris * r)


def caja(mask):
    """Límites (y0, y1, x0, x1) del rectángulo que contiene la máscara."""
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


# ---------------------------------------------------------------------------------------------
# Watershed
# ---------------------------------------------------------------------------------------------

def reflejos(img_est, region, k_mad, dil_px):
    """Reflejos especulares: píxeles de la región más brillantes que mediana + k_mad * MAD (escalado), sobre la
    imagen estirada sin mediana (la mediana los atenúa). Se dilatan dil_px para cubrir su halo."""
    v = img_est[region].astype(np.float32)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    m = region & (img_est > med + k_mad * max(mad, 1.0))
    if dil_px > 0 and m.any():
        m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_px + 1,) * 2)) > 0
    return m


def centro_oscuro(img, region, sigma_px):
    """Punto más oscuro de la región después de un suavizado gaussiano fuerte (centro aproximado de la pupila,
    que puede estar descentrada respecto del iris)."""
    g = cv2.GaussianBlur(img, (0, 0), sigma_px).astype(np.float32)
    g[~region] = np.inf
    y, x = np.unravel_index(np.argmin(g), g.shape)
    return float(x), float(y)


def marcadores(img, img_est, iris, region, p, centro=None, refl=None, zona_centro=None, marcador_r=None):
    """Marcadores del watershed: 1 = fondo (iris claro y exterior de la región), 2 = pupila, 3 = reflejos.

    Pupila: disco de marcador_pupila_iris radios de iris alrededor del punto más oscuro (centro_oscuro) a menos de
    centro_r_iris radios del centro del iris (así no lo atrae la sombra del párpado), sin los reflejos. Reflejos:
    marcador propio, separado de la pupila, para que el watershed no los asigne al fondo por ser claros ni los mezcle
    con la pupila; los huecos que dejan dentro de la pupila los rellena la morfología. Fondo: fuera de la región de
    búsqueda, más los píxeles de la región más claros que el percentil fondo_percentil (calculado en cada frame) que
    no son reflejos y están lejos del marcador de pupila. `centro` (x, y) y `refl` permiten fijar el centro del
    marcador de pupila y reusar la máscara de reflejos; por defecto se calculan acá. `zona_centro` (máscara) reemplaza el
    disco de centro_r_iris donde se busca el centro y `marcador_r` [px], el radio del marcador (continuidad temporal).
    """
    cx, cy, r = iris
    d_iris = 2 * r
    if refl is None:
        refl = reflejos(img_est, region, p["reflejo_k_mad"], int(round(p["reflejo_dil_iris_frac"] * d_iris)))
    if centro is None:
        # la pupila nunca está muy descentrada del iris (o, con continuidad, cerca de la pupila anterior)
        cerca = disco(img.shape, cx, cy, p["centro_r_iris"] * r) if zona_centro is None else zona_centro
        zona = region & cerca & ~refl
        if not zona.any():
            zona = region & cerca if (region & cerca).any() else region
        px, py = centro_oscuro(img, zona, p["centro_sigma_iris_frac"] * d_iris)
    else:
        px, py = centro
    pup = disco(img.shape, px, py, p["marcador_pupila_iris"] * r if marcador_r is None else marcador_r) & ~refl
    lejos = ~disco(img.shape, px, py, p["fondo_dist_iris"] * r)
    claro = region & (img >= np.percentile(img[region], p["fondo_percentil"])) & ~refl & lejos
    m = np.zeros(img.shape, np.int32)
    m[~region | claro] = 1
    m[refl] = 3
    m[pup] = 2
    return m, (px, py)


def cuenca_pupila(img, img_est, iris, region, p, zona_centro=None, marcador_r=None):
    """Watershed por marcadores sobre la magnitud del gradiente (Sobel) de la imagen preprocesada.

    Las cuencas crecen desde los marcadores y se encuentran sobre las crestas del gradiente, es decir sobre los
    bordes; la pupila es la cuenca del marcador 2. Se calcula en el cuadrado que contiene la región (más 1 px):
    afuera todo es marcador de fondo, así que dentro de la región el resultado es el mismo que sobre la imagen
    entera. Devuelve (cuenca, marcadores, reflejos, centro del marcador).
    """
    from skimage.segmentation import watershed
    cx, cy, r = iris
    refl = reflejos(img_est, region, p["reflejo_k_mad"], int(round(p["reflejo_dil_iris_frac"] * 2 * r)))
    m, centro = marcadores(img, img_est, iris, region, p, refl=refl, zona_centro=zona_centro, marcador_r=marcador_r)
    y0, y1, x0, x1 = caja(region)
    y0, x0, y1, x1 = max(y0 - 1, 0), max(x0 - 1, 0), y1 + 1, x1 + 1
    sub = img[y0:y1, x0:x1]
    grad = np.hypot(cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3))
    cuenca = np.zeros(img.shape, bool)
    cuenca[y0:y1, x0:x1] = watershed(grad, m[y0:y1, x0:x1]) == 2
    return region & cuenca, m, refl, centro


# ---------------------------------------------------------------------------------------------
# Morfología, elipse e indicador de calidad
# ---------------------------------------------------------------------------------------------

def morfologia_final(cuenca, iris, p):
    """Relleno de huecos -> componente más compatible con una pupila -> relleno -> envolvente convexa (opcional).

    p: sección `componente` (diam_min_iris, diam_max_iris, peso_distancia, envolvente_convexa).
    Devuelve (componente sin envolvente, pupila final) o (None, None) si ninguna componente es admisible.
    """
    m = rellenar_huecos(cuenca)
    comp, _ = elegir_componente(m, iris, p)
    if comp is None:
        return None, None
    comp = rellenar_huecos(comp["mascara"])
    return comp, (convex_hull_image(comp) if p["envolvente_convexa"] else comp)


def medidas_calidad(componente, refl, iris, r_region, p_cal, excluir_extra=None):
    """Medidas del indicador sobre la componente (antes de la envolvente): inliers, residuo, circularidad y
    diámetro de la elipse de RANSAC en diámetros de iris (morfologia.ajuste_robusto). p_cal: sección `calidad`.
    excluir_extra: puntos del contorno que tampoco cuentan (borde artificial de la restricción por continuidad)."""
    r = iris[2]
    k_ex = max(1, int(round(p_cal["excluir_reflejo_iris_frac"] * 2 * r)))
    excluir = cv2.dilate(refl.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_ex + 1,) * 2)) > 0
    if excluir_extra is not None:
        excluir |= excluir_extra
    p_mm = {"rellenar": True, "diam_iris": p_cal["diam_iris_ransac"]}
    rng = np.random.default_rng(p_cal["semilla"])     # semilla fija por frame: el resultado no depende del orden
    q = ajuste_robusto(componente, excluir, iris, r_region, p_mm, {**p_cal, "cobertura_min": 0}, rng)
    return {k: q[k] for k in ("inliers", "residuo", "circularidad", "diametro_iris")} | {"elipse_ransac": q["elipse"]}


def restriccion_previa(shape, iris, previa, p_cont, p_ws, p_comp):
    """Restricciones de la segmentación a partir de la última pupila aceptada (continuidad temporal).

    previa: {cx, cy, r, edad_s} en px de la ROI (centro y radio de la pupila aceptada hace edad_s segundos). La pupila
    puede haber cambiado de tamaño y de lugar: el radio admisible es r (1 +- radio_tol) +- vel_radio_iris_s * edad y el
    centro se busca a menos de centro_tol_iris + vel_centro_iris_s * edad (radios de iris) del anterior. Devuelve
    (disco donde se restringe el watershed, zona del centro del marcador, radio del marcador [px], sección `componente`
    con el rango de diámetros admisible, anillo del borde de la restricción que no cuenta para el indicador).
    """
    r = iris[2]
    edad = previa.get("edad_s", 0.0)
    holgura = p_cont["vel_radio_iris_s"] * edad * r
    r_max = previa["r"] * (1 + p_cont["radio_tol"]) + holgura
    r_min = max(previa["r"] * (1 - p_cont["radio_tol"]) - holgura, 1.0)
    d_c = (p_cont["centro_tol_iris"] + p_cont["vel_centro_iris_s"] * edad) * r
    restr = disco(shape, previa["cx"], previa["cy"], r_max + d_c)
    zona = disco(shape, previa["cx"], previa["cy"], d_c)
    marcador_r = max(1.0, min(p_ws["marcador_pupila_iris"] * r, p_cont["marcador_rel"] * r_min))
    comp = {**p_comp, "diam_min_iris": max(p_comp["diam_min_iris"], r_min / r),
            "diam_max_iris": min(p_comp["diam_max_iris"], r_max / r)}
    borde = restr & ~disco(shape, previa["cx"], previa["cy"], r_max + d_c - p_cont["margen_borde_iris"] * r)
    return restr, zona, marcador_r, comp, borde


def medir_pupila(bgr, iris, p, previa=None):
    """Pupila de una ROI BGR con el método final y su indicador de calidad.

    previa: última pupila aceptada {cx, cy, r, edad_s} (px de la ROI) o None. Con ella (sección `continuidad`) el
    marcador se busca cerca de la pupila anterior y el watershed se restringe a un disco de radio admisible
    (restriccion_previa): así no salta a un derrame, que es mucho más grande. Los puntos del contorno sobre el borde de
    esa restricción no cuentan para el indicador (un derrame cortado por la restricción tendría un borde circular falso).
    p: sección `segmentacion` del config. Devuelve dict con
      elipse: {cx, cy, a, b, angulo_deg} en px de la ROI (None si no hay pupila);
      diametro_roi / area_roi: diámetro equivalente 2 sqrt(ab) y área de la máscara final, en px de la ROI;
      centroide_roi (x, y) y circularidad_mascara de la máscara final;
      inliers, residuo, circularidad, diametro_iris: medidas del indicador; confiable: True / False;
      mascara (pupila final), componente, cuenca, reflejos, marcadores, img (preprocesada), para figuras.
    """
    p_pre, p_ws, p_cal = p["preprocesamiento"], p["watershed"], p["calidad"]
    img = preprocesar(bgr, iris, p_pre)
    img_est = preprocesar(bgr, iris, p_pre, mediana=False)
    region = region_busqueda(img.shape, iris, p["busqueda_r_iris"])
    zona = marcador_r = borde = None
    p_comp = p["componente"]
    if previa is not None:
        restr, zona, marcador_r, p_comp, borde = restriccion_previa(img.shape, iris, previa, p["continuidad"], p_ws,
                                                                    p_comp)
        region = region & restr
    cuenca, m, refl, centro = cuenca_pupila(img, img_est, iris, region, p_ws, zona, marcador_r)
    comp, pup = morfologia_final(cuenca, iris, p_comp)
    out = {"elipse": None, "diametro_roi": np.nan, "area_roi": np.nan, "centroide_roi": (np.nan, np.nan),
           "inliers": np.nan, "residuo": np.nan, "circularidad": np.nan, "diametro_iris": np.nan,
           "confiable": False, "mascara": pup, "componente": comp, "cuenca": cuenca, "reflejos": refl,
           "marcadores": m, "centro_marcador": centro, "img": img, "excluir_extra": borde}
    if pup is None:
        return out
    e = ajustar_elipse(pup)
    q = medidas_calidad(comp, refl, iris, p["busqueda_r_iris"], p_cal, borde)
    ys, xs = np.nonzero(pup)
    out.update(q, elipse=e, area_roi=float(pup.sum()), centroide_roi=(float(xs.mean()), float(ys.mean())))
    if e is not None:
        out["diametro_roi"] = 2 * np.sqrt(e["a"] * e["b"])
        out["confiable"] = es_confiable({**q, "elipse": q["elipse_ransac"]}, p_cal)
    return out
