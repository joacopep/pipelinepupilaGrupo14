"""Carga de config.yaml con rutas resueltas respecto de la raíz del proyecto."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """Lee config.yaml y convierte video_path y output_dir en rutas absolutas (relativas a la raíz del proyecto)."""
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.is_file():
        raise SystemExit(f"Error: no existe el archivo de configuración {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("video_path", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else ROOT / p
    return cfg
