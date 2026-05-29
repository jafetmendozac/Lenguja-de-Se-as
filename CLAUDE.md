# CLAUDE.md — Memoria Técnica del Proyecto

> Fuente de verdad del proyecto. Actualizar después de cada etapa.
> Última actualización: 2026-05-29

---

## 1. Información General del Proyecto

### Objetivo
Sistema de reconocimiento de **Lengua de Señas Peruana (LSP)** en tiempo real mediante visión por computadora. Captura video desde webcam, detecta y extrae landmarks de la mano, clasifica señas con un modelo de ML, muestra el resultado con confianza y FPS, y reproduce la palabra detectada por síntesis de voz.

### Sobre la LSP y el dataset
No existe un dataset público masivo para LSP. Por eso el sistema usa un dataset propio:
- El usuario captura las señas reales del LSP con su webcam
- El modelo aprende exactamente esas señas → el sistema reconoce LSP auténtico
- La arquitectura es agnóstica al idioma de señas (se puede adaptar a ASL, LSM, etc.)

### Alcance Inicial (v1.0)
Reconocer 5 señas de la **Lengua de Señas Peruana (LSP)**:
- Hola
- Gracias
- Sí
- No
- Ayuda

### Plataforma Objetivo
- macOS 14+ (Sonoma / Sequoia)
- Apple Silicon M2 — 8 GB RAM
- Python 3.11+

---

## 2. Arquitectura del Sistema

### Pipeline de datos (flujo principal)

```
[Webcam]
    │
    ▼
[CameraCapture]          ← src/capture/camera.py
    │  Frame BGR (OpenCV)
    ▼
[HandDetector]           ← src/detection/hand_detector.py
    │  Landmarks 21 puntos (x, y, z) por mano
    ▼
[FeatureExtractor]       ← src/features/extractor.py
    │  Vector normalizado de 63 features
    ▼
[SignPredictor]          ← src/recognition/predictor.py
    │  Label LSP + Confidence
    ▼
 ┌──┴──┐
 │     │
[TTS]  [Display]         ← src/tts/speaker.py  |  src/ui/display.py
 │     │
[Voz] [Frame anotado con overlay]
```

### Módulos y responsabilidades

| Módulo | Archivo | Responsabilidad |
|--------|---------|-----------------|
| Config | `config/settings.py` | Parámetros globales (resolución, thresholds, paths) |
| Capture | `src/capture/camera.py` | Webcam en hilo separado, lectura de frames, FPS |
| Detection | `src/detection/hand_detector.py` | MediaPipe Hands: 21 landmarks 3D por mano |
| Features | `src/features/extractor.py` | Normalización relativa a muñeca, vector de 63 features |
| Recognition | `src/recognition/predictor.py` | Carga modelo y devuelve predicción + confianza |
| Model | `src/recognition/model.py` | Entrenamiento y serialización del clasificador LSP |
| TTS | `src/tts/speaker.py` | Síntesis de voz con cooldown anti-repetición |
| Display | `src/ui/display.py` | Overlay de landmarks, texto, FPS y estado |
| Scripts | `scripts/` | Captura de datos, entrenamiento, evaluación (offline) |
| Main | `main.py` | Loop principal, integra todos los módulos |

---

## 3. Estructura de Carpetas

```
Lenguaje_Señas/
├── CLAUDE.md                    ← Memoria técnica (este archivo)
├── README.md                    ← Guía de uso y configuración
├── PROJECT_ROADMAP.md           ← Roadmap y estado de etapas
├── requirements.txt             ← Dependencias del proyecto
│
├── config/
│   ├── __init__.py
│   └── settings.py              ← Configuración centralizada
│
├── src/
│   ├── __init__.py
│   ├── capture/
│   │   ├── __init__.py
│   │   └── camera.py            ← Gestión de webcam con threading
│   ├── detection/
│   │   ├── __init__.py
│   │   └── hand_detector.py     ← Detección de manos con MediaPipe
│   ├── features/
│   │   ├── __init__.py
│   │   └── extractor.py         ← Normalización y construcción de features
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── model.py             ← Definición y entrenamiento del clasificador
│   │   └── predictor.py         ← Inferencia en tiempo real
│   ├── tts/
│   │   ├── __init__.py
│   │   └── speaker.py           ← Text-to-speech con cooldown
│   └── ui/
│       ├── __init__.py
│       └── display.py           ← Renderizado de overlay en frames
│
├── data/
│   ├── raw/                     ← Imágenes LSP capturadas por clase
│   │   ├── hola/
│   │   ├── gracias/
│   │   ├── si/
│   │   ├── no/
│   │   └── ayuda/
│   └── processed/
│       └── landmarks.csv        ← Dataset de landmarks extraídos (63 features + label)
│
├── models/
│   └── sign_classifier.pkl      ← Modelo LSP entrenado y serializado
│
├── scripts/
│   ├── collect_data.py          ← Captura imágenes LSP para el dataset
│   ├── train_model.py           ← Entrena y guarda el modelo
│   └── evaluate_model.py        ← Métricas y matriz de confusión
│
└── tests/
    ├── test_camera.py
    ├── test_detector.py
    └── test_predictor.py
```

---

## 4. Decisiones Técnicas

### Idioma objetivo: Lengua de Señas Peruana (LSP)
- No existe dataset público grande para LSP → se captura dataset propio
- El usuario realiza las señas reales del LSP durante la fase de recolección de datos
- El modelo aprende las configuraciones de mano específicas del LSP peruano
- Referencia LSP: PUCP — Instituto Nacional de Rehabilitación del Perú

### Modelo de clasificación: Random Forest (scikit-learn)
- **Por qué no CNN**: Con landmarks normalizados (21 puntos × 3 = 63 features) un Random Forest alcanza >95% de precisión con ~200 muestras por clase, sin GPU.
- **Por qué Random Forest sobre SVM**: Más robusto a features correlacionadas, maneja varianza de posición mejor, y `predict_proba()` da probabilidades calibradas para el score de confianza.
- **Escalabilidad**: Para >20 clases o señas dinámicas se sustituye por MLP/LSTM sin cambiar el resto del pipeline.

### Detección de manos: MediaPipe HandLandmarker (Tasks API)
- MediaPipe 0.10.35 eliminó la API `mp.solutions` → se usa la nueva Tasks API.
- Modelo: `models/hand_landmarker.task` (7.5 MB, float16, optimizado ARM).
- Detecta hasta 2 manos, extrae 21 landmarks 3D por mano.
- Corre sobre TensorFlow Lite con XNNPACK delegate en Apple M2 (CPU).
- Tres modos: `IMAGE` (estático), `VIDEO` (frames), `LIVE_STREAM` (async con callback).

### Feature engineering: normalización relativa
- Los landmarks se normalizan respecto al punto 0 (muñeca) y se escalan por la distancia máxima de la mano.
- Hace el clasificador invariante a posición y escala en el frame.
- Vector final: 63 features (21 puntos × x, y, z normalizados).

### Captura de video: OpenCV + threading
- Un hilo separado lee frames de la webcam.
- Desacopla captura de procesamiento → mayor FPS efectivo.

### TTS: pyttsx3 (offline)
- Sin internet, latencia mínima (~50ms).
- Funciona con las voces del sistema macOS (español incluido).
- Cooldown de 3 segundos para no repetir la misma seña continuamente.

---

## 5. Estado del Desarrollo

| Etapa | Descripción | Estado |
|-------|-------------|--------|
| 1 | Planeamiento y arquitectura | ✅ COMPLETADA |
| 2 | Configuración del entorno | ✅ COMPLETADA |
| 3 | Detección de manos | ✅ COMPLETADA |
| 4 | Captura de dataset LSP | ✅ COMPLETADA |
| 5 | Entrenamiento del modelo | ⬜ PENDIENTE |
| 6 | Predicción en tiempo real | ⬜ PENDIENTE |
| 7 | Text-to-speech | ⬜ PENDIENTE |
| 8 | Optimización | ⬜ PENDIENTE |
| 9 | Interfaz final | ⬜ PENDIENTE |

---

## 6. Dependencias

```
# Core — visión por computadora
opencv-python>=4.9.0
mediapipe>=0.10.9
numpy>=1.26.0

# Machine Learning
scikit-learn>=1.4.0
joblib>=1.3.0

# Text-to-Speech (offline)
pyttsx3>=2.90

# Análisis y visualización
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
tqdm>=4.66.0
```

> TensorFlow/PyTorch NO se usan en v1.0. Se evaluarán en v2.0 para señas dinámicas.

---

## 7. Convenciones del Proyecto

- **Estilo**: PEP 8, máximo 100 caracteres por línea
- **Clases**: PascalCase (`HandDetector`, `SignPredictor`)
- **Funciones/variables**: snake_case (`extract_features`, `confidence_threshold`)
- **Constantes**: UPPER_SNAKE_CASE en `config/settings.py`
- **Archivos**: snake_case (`hand_detector.py`, `collect_data.py`)
- **Commits**: `[ETAPA-N] descripción breve`

---

## 8. Configuración Principal

```python
# config/settings.py (valores principales)
CAMERA_INDEX             = 0
FRAME_WIDTH              = 640
FRAME_HEIGHT             = 480
TARGET_FPS               = 30
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.5
PREDICTION_THRESHOLD     = 0.75   # Confianza mínima para mostrar predicción
TTS_COOLDOWN_SECONDS     = 3.0
CLASSES = ["ayuda", "gracias", "hola", "no", "si"]  # LSP
MODEL_PATH               = "models/sign_classifier.pkl"
DATA_DIR                 = "data/raw"
LANDMARKS_CSV            = "data/processed/landmarks.csv"
```

---

## 9. Comandos Útiles

```bash
# Instalar dependencias
pip install -r requirements.txt

# Capturar dataset LSP (repetir para cada seña)
python scripts/collect_data.py --class hola --samples 200

# Entrenar modelo con dataset LSP
python scripts/train_model.py

# Evaluar modelo (métricas + matriz de confusión)
python scripts/evaluate_model.py

# Ejecutar sistema en tiempo real
python main.py
```

---

## 10. Errores Conocidos y Limitaciones

*(Se irá llenando durante el desarrollo)*

---

## 11. Mejoras Futuras (v2.0+)

- [ ] Más señas LSP (alfabeto dactilológico, números)
- [ ] Señas dinámicas (movimientos) con LSTM o Transformer
- [ ] Soporte para 2 manos simultáneas
- [ ] Interfaz gráfica con PyQt6
- [ ] Exportar modelo a ONNX para mayor velocidad
- [ ] Colaboración con comunidad sorda peruana para validar señas

---

## 12. Registro de Cambios (Changelog)

### 2026-05-29 — ETAPA 4
- Actualizado `src/detection/hand_detector.py` con parámetro `mode="video"|"image"` para soportar tanto webcam como batch de imágenes
- Implementado `src/features/extractor.py` — normalización relativa a muñeca + escalado por distancia máxima → vector de 63 features invariante a posición y escala (validado con test unitario)
- Creado `scripts/collect_data.py` — captura interactiva con ESPACIO/Q, barra de progreso, flash visual, auto-captura con cooldown
- Creado `scripts/extract_landmarks.py` — batch processing de imágenes → landmarks.csv con columnas lm0_x...lm20_z + label

### 2026-05-29 — ETAPA 3
- Implementado `src/capture/camera.py` — captura en hilo de fondo, FPS con ventana deslizante de 30 frames, context manager
- Implementado `src/detection/hand_detector.py` — HandLandmarker (Tasks API, modo VIDEO), extracción de 21 landmarks como numpy array, conexiones del esqueleto definidas
- Implementado `src/ui/display.py` — landmarks + conexiones, FPS, panel inferior semitransparente, barra de confianza, etiqueta izquierda/derecha
- Creado `scripts/test_detection.py` — prueba visual en tiempo real con controles Q/S

### 2026-05-29 — ETAPA 2
- Entorno virtual creado: `venv/` (Python 3.13.7, Apple Silicon)
- Todas las dependencias instaladas correctamente
- **Decisión técnica clave**: MediaPipe 0.10.35 eliminó la API `mp.solutions` → se usa la nueva Tasks API con `HandLandmarker` y modelo `hand_landmarker.task` (7.5 MB)
- Descargado `models/hand_landmarker.task` de Google MediaPipe Models
- Actualizado `config/settings.py` con `HAND_LANDMARKER_PATH`
- Creado y validado `scripts/verify_setup.py`
- Webcam detectada: 1920×1080 en MacBook Air M2
- Verificación completa: todos los módulos OK

### 2026-05-29 — ETAPA 1
- Creado CLAUDE.md con arquitectura completa y decisiones técnicas
- Definido objetivo: Lengua de Señas Peruana (LSP)
- Definida estrategia de dataset propio (no hay LSP público)
- Creado README.md, PROJECT_ROADMAP.md, requirements.txt
- Definida estructura de carpetas completa

---

## 13. Contexto para Retomar el Proyecto

Si se pierde el contexto de conversación:
1. Leer este archivo completo
2. Revisar la tabla de estado en sección 5
3. Verificar qué archivos existen en `src/` para saber qué está implementado
4. El modelo entrenado vive en `models/sign_classifier.pkl`
5. El dataset de landmarks está en `data/processed/landmarks.csv`
6. El punto de entrada del sistema es `main.py`
7. Las señas objetivo son del LSP: hola, gracias, sí, no, ayuda
