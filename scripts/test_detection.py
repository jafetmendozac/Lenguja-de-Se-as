"""
ETAPA 3 — Test de detección de manos en tiempo real.

Muestra la webcam con:
  - 21 landmarks de la mano dibujados
  - Esqueleto de conexiones
  - FPS en tiempo real
  - Etiqueta izquierda/derecha
  - Panel de estado inferior

Controles:
  Q  — salir
  S  — guardar screenshot en data/raw/

Uso:
    python scripts/test_detection.py
"""

import sys
import time
import cv2

sys.path.insert(0, ".")

from src.capture.camera import CameraCapture
from src.detection.hand_detector import HandDetector
from src.ui.display import Display
from config.settings import FRAME_WIDTH, FRAME_HEIGHT


def main() -> None:
    print("\n[ETAPA 3] Test de detección de manos — LSP")
    print("Controles: Q = salir | S = screenshot\n")

    display  = Display()
    cam_start_ms = int(time.monotonic() * 1000)

    with CameraCapture() as cam, HandDetector() as detector:
        print(f"Webcam lista. Resolución: {FRAME_WIDTH}×{FRAME_HEIGHT}")

        while True:
            frame = cam.read()
            if frame is None:
                continue

            # Timestamp relativo al inicio (debe crecer monotónicamente)
            timestamp_ms = int(time.monotonic() * 1000) - cam_start_ms

            result = detector.detect(frame, timestamp_ms)

            # ── Dibujar ──────────────────────────────────────────────────
            display.draw_landmarks(frame, result)
            display.draw_hand_label(frame, result)
            display.draw_fps(frame, cam.fps)

            if detector.hand_detected(result):
                lm = detector.get_landmarks_array(result)
                display.draw_info_panel(
                    frame,
                    sign="Mano detectada",
                    confidence=1.0,
                    status="",
                )
            else:
                display.draw_no_hand_warning(frame)
                display.draw_info_panel(frame, status="Muestra tu mano a la cámara")

            cv2.imshow("LSP — Detección de manos (Q para salir)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = f"data/raw/screenshot_{timestamp_ms}.jpg"
                cv2.imwrite(path, frame)
                print(f"Screenshot guardado: {path}")

    cv2.destroyAllWindows()
    print("\nTest finalizado.")


if __name__ == "__main__":
    main()
