"""
Módulo de detección de manos usando la nueva Tasks API de MediaPipe.

MediaPipe 0.10.14+ eliminó la API mp.solutions. Este módulo usa
HandLandmarker del paquete mediapipe.tasks, que requiere el modelo
'hand_landmarker.task' (descargado en la ETAPA 2).

Modos disponibles:
  - "video"  → RunningMode.VIDEO  — para webcam en tiempo real (frames con timestamp)
  - "image"  → RunningMode.IMAGE  — para procesar imágenes estáticas individualmente
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from config.settings import (
    HAND_LANDMARKER_PATH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


# Conexiones entre los 21 landmarks para dibujar el esqueleto de la mano.
# Cada tupla (a, b) representa una línea entre el punto a y el punto b.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Pulgar
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Índice
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Medio
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Anular
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Meñique
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palma
    (0, 17),
]


class HandDetector:
    """
    Detecta manos en frames de video o imágenes estáticas y extrae 21 landmarks 3D.

    Uso en tiempo real (webcam):
        with HandDetector(mode="video") as detector:
            result = detector.detect(frame, timestamp_ms)

    Uso en batch (imágenes guardadas):
        with HandDetector(mode="image") as detector:
            result = detector.detect(frame)
    """

    def __init__(self, mode: str = "video") -> None:
        assert mode in ("video", "image"), "mode debe ser 'video' o 'image'"
        self._mode = mode

        running_mode = RunningMode.VIDEO if mode == "video" else RunningMode.IMAGE

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            running_mode=running_mode,
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    # ── Detección ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None):
        """
        Detecta manos en el frame dado.

        Args:
            frame_bgr:     Frame en formato BGR (salida de OpenCV).
            timestamp_ms:  Requerido en modo "video". Debe ser creciente.

        Returns:
            HandLandmarkerResult con hand_landmarks, handedness, etc.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._mode == "video":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms es obligatorio en modo 'video'.")
            return self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            return self._landmarker.detect(mp_image)

    def get_landmarks_array(self, result) -> np.ndarray | None:
        """
        Extrae los landmarks de la primera mano detectada como array numpy.

        Returns:
            Array de forma (21, 3) con coordenadas (x, y, z) normalizadas [0, 1],
            o None si no se detectó ninguna mano.
        """
        if not result or not result.hand_landmarks:
            return None
        landmarks = result.hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    def hand_detected(self, result) -> bool:
        return bool(result and result.hand_landmarks)

    # ── Context manager ──────────────────────────────────────────────────────

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
