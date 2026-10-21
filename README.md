# Pipeline de reflejo fotomotor pupilar

A partir de un video de un ojo frente a un estímulo luminoso (luz que se prende o se apaga), el pipeline devuelve la curva
temporal del diámetro y el área de la pupila frame a frame, y los parámetros del reflejo fotomotor: latencia, amplitud y
velocidades de constricción y de redilatación. Cada valor viene con su confianza.

Todo se detecta en el propio video: el instante del estímulo (el mayor salto de brillo), el ojo y el iris, y la escala en mm
(se toma un iris de 11.7 mm). No hay que seleccionar nada a mano. Los frames que no se pueden medir (parpadeos, pupila no
detectada) quedan como NaN y el procesamiento sigue.

## Instalación

Requiere Python 3.13. Probado con **Python 3.13.15** (64 bits) en Windows 11, con las versiones de `requirements.txt`.

```
python -m venv .venv
.venv\Scripts\activate          # en Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo procesar un video

```
python run.py ruta/al/video.mov
```

Si no se pasa la ruta, se usa `video_path` de `config.yaml` (relativa a la carpeta del proyecto, o absoluta):

```
python run.py
```

La salida queda en `outputs/<nombre del video sin extensión>/`: `curva.csv` y `parametros.csv`. Al terminar se imprime un
resumen de los parámetros principales. Un video 4K de 30 s tarda del orden de 1 a 3 minutos. Si el video no existe, `run.py`
termina con un mensaje de error y no crea nada. El resto de `config.yaml` son los parámetros del método y no hace falta
tocarlos.

Lo que se espera del video: un ojo visible (primer plano o media cara, también con el celular de costado), unos segundos de
línea de base antes del estímulo y un cambio de luz brusco.

## `curva.csv`: una fila por frame

| Columna | Unidad | Significado |
|---|---|---|
| `frame` | | índice del frame en el video (desde 0) |
| `t_s` | s | tiempo real del frame (timestamp del contenedor, no n / fps) |
| `t_rel_estimulo_s` | s | tiempo relativo al estímulo detectado (negativo antes) |
| `diametro_px` | px | diámetro de la pupila en píxeles del frame original |
| `diametro_mm` | mm | diámetro de la pupila (NaN si el frame no es válido; interpolado en huecos de hasta 0.25 s) |
| `diametro_norm` | | diámetro / diámetro basal |
| `area_px`, `area_mm2`, `area_norm` | px², mm², | área de la pupila en las mismas tres escalas |
| `diametro_crudo_mm` | mm | diámetro medido en el frame, sin control de calidad ni limpieza |
| `calidad` | 0 / 1 | 1 si el indicador de calidad estricto acepta la segmentación |
| `confianza` | | `alta`, `media` o `baja` (ver abajo); vacía si el frame no es válido |
| `inliers`, `residuo`, `circularidad` | | medidas del indicador de calidad |
| `iris_detectado` | 0 / 1 | 1 si en el frame se encontró el iris |
| `estado` | | `ok`, `rechazado` (indicador de calidad), `atipico` (filtro de Hampel), `parpadeo` o `interpolado` |
| `interpolado` | 0 / 1 | 1 si el valor es interpolado |
| `diametro_suave_mm` | mm | curva suavizada (Butterworth de fase cero, 3 Hz) |
| `velocidad_mm_s` | mm/s | derivada de la curva suavizada |
| `diametro_modelo_mm`, `velocidad_modelo_mm_s` | mm, mm/s | sigmoide ajustada a la fase de respuesta y su derivada (solo en la ventana del ajuste) |

## `parametros.csv`: una fila por parámetro

Columnas: `parametro`, `valor`, `unidad`, `confianza` (vacía en los datos que no son una medición de la pupila). Los
tiempos con sufijo `_s` se miden desde el estímulo, salvo `t_estimulo_s` y `duracion_s` (desde el comienzo del video).

| Parámetro | Significado |
|---|---|
| `video`, `duracion_s`, `fps_medio` | video procesado |
| `t_estimulo_s`, `tipo_estimulo`, `salto_brillo` | instante del estímulo, `luz ON` o `luz OFF`, tamaño del salto de brillo |
| `mm_por_px`, `iris_diametro_px` | escala: mm por píxel del frame original y diámetro del iris en px |
| `frames_validos_pct`, `frames_interpolados_pct` | porcentaje de frames con diámetro válido e interpolado |
| `nivel_fase_pre`, `nivel_fase_post` | nivel de confianza mínimo aceptado antes y después del estímulo |
| `fc_hz` | frecuencia de corte del suavizado |
| `basal_mm` | diámetro basal: mediana en los 2 s previos al estímulo |
| `partida_mm` | nivel de partida: mediana entre +0.10 y +0.18 s (la luz ya cambió y la pupila todavía no reaccionó) |
| `escalon_estimulo_mm` | partida − basal: salto de la medición al cambiar la iluminación (no es fisiológico) |
| `extremo_mm`, `t_extremo_s` | mínimo (luz ON) o máximo (luz OFF) de la respuesta y su instante |
| `final_mm` | mediana de los últimos 3 s |
| **`latencia_s`** | inicio de la respuesta: tangente en el instante de velocidad máxima extendida hasta el nivel de partida |
| `latencia_umbral_s` | referencia: primer cruce de un umbral (3 desvíos del basal o 5 % de la amplitud) |
| **`amplitud_mm`**, `amplitud_pct` | luz ON: partida − mínimo; luz OFF: final − partida (en mm y en % de la partida) |
| `amplitud_desde_basal_mm` | basal − mínimo (luz ON) o final − basal (luz OFF) |
| `amplitud_referencia` | `partida` o `basal`: desde dónde se midió `amplitud_mm` |
| **`vel_max_constriccion_mm_s`** | velocidad máxima de constricción (negativa), por modelo sigmoide; solo luz ON |
| **`vel_max_redilatacion_mm_s`** | velocidad máxima de redilatación: luz OFF por modelo; luz ON por derivada en los 3 s posteriores al mínimo |
| `vel_max_*_derivada_mm_s` | las mismas velocidades por derivada de la curva suavizada |
| `vel_max_*_metodo` | `modelo`, `derivada` o `respaldo` |
| `ajuste_*_r2` | R² de la sigmoide ajustada |
| `t_vel_max_s` | instante de la velocidad máxima |
| `fraccion_basal_valida` | fracción de frames válidos en la ventana basal |
| `respaldo_fuente` | si se usó la estimación de respaldo, de qué datos salió |
| **`confianza_global`** | la menor confianza entre los cuatro parámetros principales (en negrita) |

Un parámetro solo es NaN si no corresponde (velocidad de constricción con luz OFF) o si no hay ningún dato para estimarlo.

## Confianza

**Por frame** (columna `confianza` de `curva.csv`):
- **alta**: la segmentación pasa el indicador de calidad estricto (contorno elíptico, circular y de tamaño plausible).
- **media**: pasa umbrales relajados del indicador.
- **baja**: cualquier medición, sin control de calidad.

En cada fase (antes y después del estímulo) se usa el nivel más alto que cubre al menos el 25 % de la fase: si el indicador
rechaza casi toda una fase, se baja de nivel en vez de dejarla vacía. Los frames interpolados heredan el menor nivel de sus
dos extremos.

**Por parámetro** (columna `confianza` de `parametros.csv`): la menor de los frames que usa. Además es **baja** si el valor es
poco plausible: latencia fuera de 0.18–0.8 s o con la tangente y el umbral a más de 0.15 s, velocidad mayor que 6 mm/s
(constricción) o 3 mm/s (redilatación), amplitud en el sentido contrario al esperado, sigmoide con R² < 0.8, o valor estimado
por respaldo (la ventana de respuesta se quedó sin curva). Un parámetro de confianza baja es una estimación: conviene
revisarlo junto con la curva.

## Unidades

- Tiempos en segundos reales (timestamps de cada frame).
- Diámetros en px del frame original (`*_px`), en mm (`*_mm`, con iris de 11.7 mm) y normalizados por el basal (`*_norm`,
  adimensional). Áreas en px², mm² y normalizadas.
- Velocidades en mm/s (negativas cuando la pupila se achica). `amplitud_pct` en % del nivel de partida.

## Estructura

```
run.py          punto de entrada
config.yaml     ruta del video, carpeta de salida y parámetros del método
requirements.txt
src/
  config.py     carga de config.yaml
  io_utils.py   lectura del video con timestamps reales
  estimulo.py   instante del estímulo
  roi.py        detección del ojo y del iris, seguimiento y ROI normalizada
  preprocess.py preprocesamiento (canal R, estiramiento, mediana)
  segment.py    segmentación de la pupila (watershed) e indicador de calidad
  morfologia.py operaciones morfológicas y ajuste de elipse
  features.py   recorrido del video, curva y parámetros fisiológicos
```
