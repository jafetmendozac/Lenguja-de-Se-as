"""
Sistema de Reconocimiento de Lengua de Señas Peruana (LSP)
Punto de entrada principal.

Pipeline:
  Webcam → HandDetector (MediaPipe) → FeatureExtractor
        → SignPredictor (Random Forest) → Display + TTS

Controles:
  Q  — salir
  R  — reiniciar historial de predicción
  S  — guardar screenshot

Uso:
  python main.py
  python main.py --no-tts       # Desactivar síntesis de voz
  python main.py --camera 1     # Usar cámara alternativa
"""

import argparse
import sys
import time
import cv2
from pathlib import Path

from src.capture.camera import CameraCapture
from src.detection.hand_detector import HandDetector
from src.recognition.predictor import SignPredictor
from src.tts.speaker import Speaker
from src.ui.display import Display
from config.settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MODEL_PATH,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSP Sign Language Recognition System")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Índice de cámara")
    p.add_argument("--no-tts", action="store_true", help="Desactivar síntesis de voz")
    return p.parse_args()


def check_model() -> bool:
    """Verifica que el modelo entrenado existe antes de arrancar."""
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
    print("  LSP — Lengua de Señas Peruana | Sistema en tiempo real")
    print("=" * 55)

    if not check_model():
        sys.exit(1)

    # ── Inicializar módulos ──────────────────────────────────────────────────
    print("\n  Cargando módulos...")
    display   = Display()
    speaker   = Speaker(enabled=not args.no_tts)
    tts_on    = not args.no_tts

    try:
        predictor = SignPredictor()
        print(f"  Modelo cargado. Clases LSP: {list(predictor._clf.classes)}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    cam_start_ms = int(time.monotonic() * 1000)

    print("  Abriendo webcam...")
    print("  Controles: Q = salir | R = reiniciar | S = screenshot\n")

    # ── Loop principal ───────────────────────────────────────────────────────
    with CameraCapture(index=args.camera) as cam, HandDetector(mode="video") as detector:
        print("  Sistema activo. Muestra tu mano a la cámara.\n")

        screenshot_count = 0

        while True:
            frame = cam.read()
            if frame is None:
                continue

            ts_ms  = int(time.monotonic() * 1000) - cam_start_ms
            result = detector.detect(frame, ts_ms)

            # ── Predicción ───────────────────────────────────────────────────
            landmarks = detector.get_landmarks_array(result)
            state     = predictor.update(landmarks)

            # ── TTS: reproducir al confirmar una seña (cooldown interno en Speaker) ──
            if state.is_stable and state.label:
                speaker.speak(state.display_label)

            # ── Renderizado ───────────────────────────────────────────────────
            display.draw_landmarks(frame, result)
            display.draw_hand_label(frame, result)
            display.draw_fps(frame, cam.fps)

            if state:
                display.draw_info_panel(
                    frame,
                    sign=state.display_label,
                    confidence=state.confidence,
                    is_stable=state.is_stable,
                )
            elif detector.hand_detected(result):
                display.draw_info_panel(
                    frame,
                    status="Mano detectada — analizando...",
                )
            else:
                display.draw_no_hand_warning(frame)
                display.draw_info_panel(frame, status="Muestra tu mano a la cámara")

            cv2.imshow("LSP — Reconocimiento en tiempo real (Q=salir)", frame)

            # ── Controles de teclado ──────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                predictor.reset()
                print("  Historial reiniciado.")
            elif key == ord("s"):
                path = f"data/raw/screenshot_{screenshot_count:04d}.jpg"
                cv2.imwrite(path, frame)
                print(f"  Screenshot guardado: {path}")
                screenshot_count += 1

    cv2.destroyAllWindows()
    speaker.stop()
    print("\n  Sistema detenido.\n")


if __name__ == "__main__":
    main()
