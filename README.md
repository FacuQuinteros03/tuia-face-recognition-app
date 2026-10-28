# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── services/
│       │   ├── face_service.py  # ⚠️ Funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME=det_10g.onnx` (SCRFD de InsightFace, en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE`, `ALIGN_MARGIN` y `DET_PAD` (borde que se agrega antes de detectar; sin él SCRFD no detecta primeros planos, ver notebook 1).
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado por el grupo

La detección, alineación y normalización están en un único módulo, [`src/lib/face_pipeline.py`](src/lib/face_pipeline.py), que importan **las dos notebooks y el backend**. Así el rostro que ve el modelo en la app se procesa exactamente igual que en el entrenamiento.

1. `detect_faces`: SCRFD (`det_10g.onnx`) con un borde de `DET_PAD` alrededor de la imagen. Devuelve cajas `(x1, y1, x2, y2)` enteras, recortadas a la imagen y filtradas por `MIN_FACE_SIZE`; lista vacía si no hay rostros.
2. `align_face`: busca los 5 landmarks en un recorte con margen (`ALIGN_MARGIN`) alrededor de la caja y aplica una transformación de similitud a la plantilla de ArcFace (`norm_crop`) → rostro BGR de `FACE_SIZE × FACE_SIZE`. Los keypoints se devuelven relativos a la caja.
3. `extract_embedding_from_face`: BGR→RGB, `(x − 127.5) / 127.5`, NCHW, inferencia (`.onnx` o `.pth`) y normalización L2. Devuelve `EMBEDDING_DIM` floats.

Mejoras opcionales incluidas:

- `identify` usa `PgVectorEmbeddingStore.search` (operador `<=>` de pgvector) cuando la métrica es coseno, en lugar de recorrer todos los registros en Python. Se corrigió además un bug en `search` (armaba los registros con `model_validate` sobre una tupla).
- Tests unitarios en [`tests/test_face_service.py`](tests/test_face_service.py) (no requieren modelos): `python -m pytest tests`.

## Modelo y fine-tuning

Detalle completo en [`2-train.ipynb`](2-train.ipynb).

- **Modelo elegido**: ArcFace R50 de InsightFace (`w600k_r50.onnx`, ResNet-50 entrenada con pérdida ArcFace sobre WebFace600K, embedding de 512) junto con el detector SCRFD (`det_10g.onnx`). Licencia de InsightFace: solo uso no comercial / investigación.
- **Link público de descarga** (release oficial, solo lectura): https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip — descomprimir y copiar `w600k_r50.onnx` y `det_10g.onnx` en `models/`.
- **Modelo comparado (descartado)**: CNN custom ([`src/lib/custom_cnn.py`](src/lib/custom_cnn.py)): ResNet chica (4 etapas residuales de 32 a 256 canales, pooling GDC, embedding de 512, 2,9 M de parámetros) entrenada desde cero sobre 75 identidades de Pins. Se hicieron 5 experimentos (softmax vs. CosFace/ArcFace, GAP vs. GDC, con y sin augmentation); el mejor fue ArcFace + GDC + augmentation. Hiperparámetros: AdamW (lr 2e-3, weight decay 5e-4), warmup de 2 épocas + coseno, batch 128, hasta 15 épocas con early stopping por AUC en identidades no vistas, fp32 (la GTX 1650 da NaN en fp16).
- **Métricas** (20 personas nunca vistas por la CNN):

  | Métrica | CNN custom | ArcFace R50 |
  |---|---|---|
  | AUC (verificación) | 0,906 | 1,000 |
  | EER | 17,6% | 0,13% |
  | TAR @ FAR=1e-3 | 13,0% | 99,9% |
  | Top-1 accuracy | 73,6% | 100% |
  | Precision / Recall / F1 (macro, con desconocidos) | 0,60 / 0,67 / 0,62 | 1,00 / 1,00 / 1,00 |
  | Desconocidos rechazados | 17,7% | 100% |
  | Tiempo de inferencia CPU (ms/rostro) | 3,3 | 27,4 |
  | Tamaño (MB) | 11,8 | 174,4 |

- **Justificación**: ArcFace es muy superior donde importa para la app (separar personas no vistas y rechazar desconocidos). La CNN es 15 veces más chica y 8 veces más rápida, pero rechaza solo el 18% de los desconocidos: aprende a clasificar sus 75 personas (92% de accuracy) pero no generaliza a gente nueva, por la diferencia en datos (75 personas vs. WebFace600K), capacidad y tiempo de entrenamiento.
- **Umbral**: `SIMILARITY_THRESHOLD=0.28` (coseno), calibrado **solo con las 10 personas de validación** como el umbral que maximiza el F1 macro de identificación open-set (galería de 7 personas + 3 desconocidas). Está en el centro de la meseta de F1 ≈ 1 (0,22–0,38) y por encima de los umbrales de EER (0,20) y FAR = 1e-3 (0,22).
- **Análisis de errores**: los falsos positivos de ArcFace (10 de ~486.000 pares) son entre mujeres de rasgos muy parecidos, con scores apenas sobre el umbral; los falsos negativos son máscaras, fotos de infancia vs. adultez, perfiles y fotos con texto encima.

## Dataset

Detalle completo en [`1-pre-processing.ipynb`](1-pre-processing.ipynb).

- **Fuente**: [Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) (Kaggle): 105 celebridades, 17.534 fotos de Pinterest ya recortadas alrededor del rostro. La notebook lo descarga sola con `kagglehub` (no requiere cuenta).
- **Licencia**: Kaggle lo publica como CC0, pero las fotos son de Pinterest y ese CC0 no cubre los derechos de los fotógrafos ni la imagen de las personas: lo usamos solo con fines académicos y no se versiona (`data/*` está en el `.gitignore`).
- **Cantidad y balance**: entre 86 y 237 imágenes por persona (mediana 169); desbalance moderado (2,8× entre la identidad más grande y la más chica).
- **Variaciones**: pose (el ~9% son perfiles marcados), iluminación (escenas oscuras y sobreexpuestas), 7% en blanco y negro, expresiones, maquillaje, oclusiones (anteojos, pelo) y fotos de distintas épocas. Sesgo demográfico: celebridades mayormente jóvenes, de EE. UU. y Europa, de piel clara.
- **Reglas de calidad** (en orden): archivo corrupto, sin rostro, varios rostros sin uno principal, rostro < 40 px, blur (varianza del Laplaciano < 20), exposición extrema, casi-duplicados (similitud de embeddings > 0,90, dentro y entre identidades) y etiquetas erróneas u outliers (similitud con la propia identidad < 0,30 o más cerca de otra). Se conservan **16.251 imágenes (92,7%)**; casi todo lo descartado son casi-duplicados (7,1%).
- **Splits** (semilla 42): 75 personas para entrenar (9.138 / 1.142 / 1.143 imágenes en `train` / `val_closed` / `test_closed`), 10 personas no vistas para calibrar (`val_open`, 1.774) y 20 personas no vistas para el test final (`test_open`, 3.054). Sin fuga de datos entre splits.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
