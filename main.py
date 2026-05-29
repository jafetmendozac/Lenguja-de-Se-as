"""
Sistema de Reconocimiento de Lengua de Señas Peruana (LSP)
Punto de entrada principal — Interfaz final (ETAPA 9).

Layout:
  Barra superior: estado ● | título | FPS
  Centro:         video en vivo + landmarks de la mano
  Panel inferior: seña detectada | confianza | historial | atajos

Optimizaciones activas:
  - Frame skipping (1 de cada DETECTION_SKIP_FRAMES+1)
  - Landmark delta (no re-predice si la mano está quieta)
  - Confidence smoothing (promedio deslizante)

Controles:
  Q  — salir
  R  — reiniciar historial de predicción
  S  — guardar screenshot

Uso:
  python main.py
  python main.py --no-tts
  python main.py --camera 1
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from src.capture.camera import CameraCapture
from src.detection.hand_detector import HandDetector
from src.recognition.predictor import SignPredictor
from src.tts.speaker import Speaker
from src.ui.display import Display
from config.settings import (
    CAMERA_INDEX,
    MODEL_PATH,
    DETECTION_SKIP_FRAMES,
)

HISTORY_MAXLEN = 5   # Últimas N señas confirmadas mostradas en el historial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSP Sign Language Recognition System")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX)
    p.add_argument("--no-tts", action="store_true", help="Desactivar síntesis de voz")
    return p.parse_args()


def check_model() -> bool:
    if not Path(MODEL_PATH).exists():
        print("\n[ERROR] Modelo no encontrado.")
        print(f"  Ruta esperada: {MODEL_PATH}")
        print("\n  Para entrenar el modelo, ejecuta en orden:")
        print("    1. python scripts/collect_data.py --class hola --samples 200")
        print("    2. (repite para: gracias, si, no, ayuda)")
        print("    3. python scripts/extract_landmarks.py")
        print("    4. python scripts/train_model.py\n")
        return False
    return True


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 55)
    print("  LSP — Lengua de Señas Peruana | v1.0")
    print("=" * 55)

    if not check_model():
        sys.exit(1)

    print("\n  Iniciando módulos...")
    display  = Display()
    speaker  = Speaker(enabled=not args.no_tts)

    try:
        predictor = SignPredictor()
        print(f"  Modelo listo. Clases: {list(predictor._clf.classes)}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Historial de señas confirmadas para mostrar en pantalla
    sign_history: deque[str] = deque(maxlen=HISTORY_MAXLEN)

    # Estado de la última seña estable (para evitar duplicar en historial)
    last_stable_label: str | None = None

    cam_start_ms    = int(time.monotonic() * 1000)
    frame_count     = 0
    last_result     = None
    screenshot_count = 0

    print(f"  Frame skip: 1/{DETECTION_SKIP_FRAMES + 1} frames")
    print("  Controles: Q = salir  |  R = reiniciar  |  S = screenshot\n")

    with CameraCapture(index=args.camera) as cam, HandDetector(mode="video") as detector:
        print("  Sistema activo.\n")

        while True:
            frame = cam.read()
            if frame is None:
                continue

            frame_count += 1
            ts_ms = int(time.monotonic() * 1000) - cam_start_ms

            # ── Frame skipping ────────────────────────────────────────────────
            if DETECTION_SKIP_FRAMES == 0 or frame_count % (DETECTION_SKIP_FRAMES + 1) == 1:
                last_result = detector.detect(frame, ts_ms)
            result = last_result

            # ── Predicción ────────────────────────────────────────────────────
            landmarks = detector.get_landmarks_array(result)
            state     = predictor.update(landmarks)

            # ── Historial y TTS: solo al confirmar una seña nueva ─────────────
            if state.is_stable and state.label and state.label != last_stable_label:
                last_stable_label = state.label
                sign_history.append(state.display_label)
                speaker.speak(state.display_label)
            elif not state.is_stable:
                last_stable_label = None   # Permite detectar la misma seña de nuevo

            # ── Renderizado completo ──────────────────────────────────────────
            display.render(frame, result, cam.fps, state, sign_history)

            cv2.imshow("LSP — Reconocimiento en tiempo real", frame)

            # ── Teclado ───────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                predictor.reset()
                last_stable_label = None
                sign_history.clear()
                print("  Historial reiniciado.")
            elif key == ord("s"):
                path = f"data/raw/screenshot_{screenshot_count:04d}.jpg"
                cv2.imwrite(path, frame)
                print(f"  Screenshot: {path}")
                screenshot_count += 1

    cv2.destroyAllWindows()
    speaker.stop()
    print("\n  Sistema detenido.\n")


if __name__ == "__main__":
    main()
