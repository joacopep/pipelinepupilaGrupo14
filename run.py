"""Punto de entrada del pipeline de reflejo fotomotor pupilar.

Uso:
  python run.py                        procesa el video de `video_path` en config.yaml
  python run.py ruta/al/video.mov      procesa ese video (la ruta tiene prioridad sobre config.yaml)
  python run.py video.mov --config otro_config.yaml

Genera en output_dir/<nombre del video>/:
  curva.csv       curva frame a frame: tiempo real, diámetro y área (px del frame original, mm y normalizados por el
                  basal), indicador de calidad, confianza (alta / media / baja), estado del frame, si el valor fue
                  interpolado, curva suavizada y velocidad;
  parametros.csv  parámetros fisiológicos (latencia, amplitud, velocidades), instante del estímulo y escala (mm/px), cada
                  uno con su confianza. Un parámetro solo es NaN si no hay ninguna forma de estimarlo.
Todo se detecta en el video: instante del estímulo, ojo, iris y escala.
"""
import argparse
import sys
import time
from pathlib import Path

from src.config import load_config
from src.features import curva, procesar_video, tabla_parametros


def main(video_path=None, config_path=None):
    cfg = load_config(config_path)
    if video_path is not None:
        cfg["video_path"] = Path(video_path).resolve()
    video = cfg["video_path"]
    if not video.is_file():
        origen = "argumento de run.py" if video_path is not None else "video_path en config.yaml"
        sys.exit(f"Error: no existe el video {video} (revisar {origen})")
    salida = cfg["output_dir"] / video.stem
    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def progreso(k, t):
        if k % 100 == 0:
            print(f"  frame {k} (t = {t:.1f} s, {time.perf_counter() - t0:.0f} s transcurridos)", flush=True)

    print(f"Procesando {video}")
    df, info = procesar_video(video, cfg, progreso=progreso)
    tabla, params = curva(df, info, cfg)
    params["video"] = video.name
    tabla.to_csv(salida / "curva.csv", index=False, float_format="%.5g")
    tabla_parametros(params).to_csv(salida / "parametros.csv", index=False, float_format="%.5g")
    print(f"Listo en {time.perf_counter() - t0:.0f} s: {salida / 'curva.csv'}, {salida / 'parametros.csv'}")
    conf = params["_confianza"]
    for k in ("t_estimulo_s", "tipo_estimulo", "mm_por_px", "basal_mm", "amplitud_mm", "amplitud_pct", "latencia_s",
              "vel_max_constriccion_mm_s", "vel_max_redilatacion_mm_s", "frames_validos_pct", "confianza_global"):
        v = params[k]
        c = f" (confianza {conf[k]})" if k in conf and k != "confianza_global" else ""
        print(f"  {k}: {v:.4g}{c}" if isinstance(v, float) else f"  {k}: {v}{c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Curva pupilar y parámetros del reflejo fotomotor a partir de un video.")
    ap.add_argument("video", nargs="?", help="ruta al video (si no se da, se usa video_path de config.yaml)")
    ap.add_argument("--config", help="archivo de configuración (por defecto, config.yaml del proyecto)")
    args = ap.parse_args()
    main(args.video, args.config)
