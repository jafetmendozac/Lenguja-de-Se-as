"""
Configuración centralizada del sistema LSP.
Todos los parámetros del proyecto se definen aquí.
Modifica este archivo para ajustar el comportamiento del sistema.
"""

from pathlib import Path

# ── Rutas base ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "raw"
PROCESSED   = BASE_DIR / "data" / "processed" / "landmarks.csv"
MODEL_PATH          = BASE_DIR / "models" / "sign_classifier.pkl"
HAND_LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"

# ── Cámara ───────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0      # 0 = webcam integrada
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
TARGET_FPS    = 30

# ── MediaPipe Hands ──────────────────────────────────────────────────────────
MAX_NUM_HANDS            = 1     # Solo una mano para v1.0
MIN_DETECTION_CONFIDENCE = 0.7   # Umbral para detectar una mano nueva
MIN_TRACKING_CONFIDENCE  = 0.5   # Umbral para continuar siguiendo la mano

# ── Reconocimiento ───────────────────────────────────────────────────────────
PREDICTION_THRESHOLD = 0.75  # Confianza mínima para mostrar la predicción
PREDICTION_HISTORY   = 5     # Frames consecutivos para confirmar una seña

# ── Optimización de rendimiento ──────────────────────────────────────────────
# MediaPipe en M2 tarda ~35ms independiente de resolución (cuello de botella = red neuronal).
# Las optimizaciones relevantes son frame skipping y landmark delta.

DETECTION_SKIP_FRAMES   = 1      # Ejecutar MediaPipe 1 de cada (N+1) frames.
                                  # 0 = todos los frames. 1 = cada 2 frames (~37 FPS display).
                                  # El resultado anterior se reutiliza en los frames saltados.

LANDMARK_CHANGE_THRESHOLD = 0.008  # Distancia media mínima entre landmarks consecutivos
                                    # para considerar que la mano se movió. Si está quieta,
                                    # se reutiliza la última predicción sin llamar al modelo.

CONFIDENCE_SMOOTH_WINDOW = 5      # Ventana del promedio deslizante de confianza para
                                   # suavizar la barra de confianza en pantalla.

# ── Señas LSP (Lengua de Señas Peruana) ─────────────────────────────────────
# Orden alfabético para que coincida con el encoding del clasificador
CLASSES = ["ayuda", "gracias", "hola", "no", "si"]

CLASSES_DISPLAY = {
    "ayuda":   "Ayuda",
    "gracias": "Gracias",
    "hola":    "Hola",
    "no":      "No",
    "si":      "Sí",
}

# ── Text-to-Speech ───────────────────────────────────────────────────────────
TTS_ENABLED         = True
TTS_COOLDOWN_SECONDS = 3.0   # Segundos mínimos entre reproducción de la misma seña
TTS_RATE            = 150    # Velocidad de voz (palabras por minuto)

# ── Dataset (captura de datos) ───────────────────────────────────────────────
SAMPLES_PER_CLASS = 200   # Imágenes a capturar por seña
CAPTURE_DELAY_MS  = 100   # ms entre capturas durante la recolección de datos

# ── Visualización (colores BGR) ──────────────────────────────────────────────
COLOR_LANDMARKS  = (0, 255, 0)      # Verde para landmarks
COLOR_TEXT_MAIN  = (255, 255, 255)  # Blanco para texto principal
COLOR_TEXT_CONF  = (0, 255, 255)    # Amarillo para confianza
COLOR_BG_PANEL   = (0, 0, 0)        # Negro para panel de info
COLOR_ALERT      = (0, 0, 255)      # Rojo para alertas
FONT             = 0                 # cv2.FONT_HERSHEY_SIMPLEX
