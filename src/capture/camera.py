"""
Módulo de captura de video desde webcam.

Usa un hilo de fondo dedicado a leer frames continuamente.
Esto desacopla la captura del procesamiento, evitando que
cv2.VideoCapture.read() bloquee el loop principal y reduciendo
la latencia efectiva del sistema.
"""

import threading
import time
import cv2
import numpy as np
from collections import deque

from config.settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TARGET_FPS,
)


class CameraCapture:
    """
    Captura continua de webcam en un hilo de fondo.

    Uso:
        with CameraCapture() as cam:
            while True:
                frame = cam.read()
                fps   = cam.fps
    """

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        self._index  = index
        self._width  = width
        self._height = height

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock   = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # FPS: ventana deslizante de los últimos 30 tiempos entre frames
        self._ts_window: deque = deque(maxlen=30)
        self._fps: float = 0.0

    # ── Ciclo de vida ────────────────────────────────────────────────────────

    def start(self) -> "CameraCapture":
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la webcam (índice {self._index}). "
                "Verifica permisos en Ajustes del Sistema > Privacidad > Cámara."
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Esperar hasta recibir el primer frame
        self._wait_for_first_frame()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    # ── Acceso a datos ───────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Devuelve el frame más reciente (copia para evitar condiciones de carrera)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Métodos internos ─────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        last_ts = time.monotonic()
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            now = time.monotonic()
            self._ts_window.append(now - last_ts)
            last_ts = now

            if len(self._ts_window) > 1:
                avg_interval = sum(self._ts_window) / len(self._ts_window)
                self._fps = 1.0 / avg_interval if avg_interval > 0 else 0.0

            with self._lock:
                self._frame = frame

    def _wait_for_first_frame(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return
            time.sleep(0.05)
        raise RuntimeError("La webcam no entregó ningún frame en el tiempo esperado.")

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "CameraCapture":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
