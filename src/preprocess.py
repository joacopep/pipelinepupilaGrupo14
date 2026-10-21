"""Preprocesamiento elegido en el Módulo 1 (pipeline).

Sobre la ROI normalizada (BGR, `roi.size` px, iris centrado):
  1. canal R: el de mejor separabilidad pupila-iris (Módulos 1.1 y 1.3);
  2. estiramiento lineal [f1, f2] -> [0, 255], con f1, f2 = percentiles de intensidad dentro del disco del iris,
     calculados en cada frame (la exposición automática cambia el brillo de un frame a otro);
  3. mediana de lado `mediana_iris_frac` diámetros de iris: borra la mayor parte de los reflejos especulares y
     conserva el borde pupilar (Módulo 1.2).
Los tamaños se expresan en radios o diámetros del iris, así que no dependen de la cámara.
"""
import cv2
import numpy as np


def disco(shape, cx, cy, r):
    """Máscara booleana del disco de centro (cx, cy) y radio r [px]."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def kernel_impar(frac, iris_diam_px):
    """Lado (impar, >= 3) de un kernel de `frac` diámetros de iris."""
    k = int(round(frac * iris_diam_px))
    return max(3, k + (k % 2 == 0))


def canal(bgr, nombre):
    """Canal de trabajo de una imagen BGR: 'B', 'G', 'R' o 'gris'."""
    if nombre in ("B", "G", "R"):
        return bgr[:, :, "BGR".index(nombre)].copy()
    if nombre == "gris":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Canal desconocido: {nombre}")


def estirar(img, mascara, percentiles):
    """Estiramiento lineal con f1, f2 = percentiles de `img` dentro de `mascara`; lo de afuera se satura."""
    f1, f2 = np.percentile(img[mascara], percentiles)
    g = (img.astype(np.float32) - f1) * 255.0 / max(f2 - f1, 1e-6)
    return np.clip(g, 0, 255).astype(np.uint8)


def preprocesar(bgr, iris, p, mediana=True):
    """Imagen preprocesada (uint8) de la ROI `bgr`.

    iris: (cx, cy, r) del iris en px de la ROI. p: parámetros (config modulo2.preprocesamiento): canal,
    percentiles, disco_r_iris (radio del disco donde se calculan los percentiles, en radios de iris) y
    mediana_iris_frac. mediana=False devuelve la imagen estirada sin filtrar (para ubicar los reflejos).
    """
    cx, cy, r = iris
    img = estirar(canal(bgr, p["canal"]), disco(bgr.shape[:2], cx, cy, p["disco_r_iris"] * r), p["percentiles"])
    if mediana:
        img = cv2.medianBlur(img, kernel_impar(p["mediana_iris_frac"], 2 * r))
    return img
