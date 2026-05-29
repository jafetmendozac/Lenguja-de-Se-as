"""
ETAPA 4 — Captura de dataset LSP.

Abre la webcam y guarda automáticamente frames cuando se detecta una mano.
Solo guarda frames con mano visible para garantizar calidad del dataset.

Uso:
    python scripts/collect_data.py --class hola --samples 200
    python scripts/collect_data.py --class gracias --samples 200
    python scripts/collect_data.py --class si --samples 200
    python scripts/collect_data.py --class no --samples 200
    python scripts/collect_data.py --class ayuda --samples 200

Controles durante la captura:
    ESPACIO  — iniciar/pausar captura automática
    Q        — salir
"""

import argparse
import sys
import time
import cv2
from pathlib import Path

sys.path.insert(0, ".")

from src.capture.camera import CameraCapture
from src.detection.hand_detector import HandDetector
from src.ui.display import Display
from config.settings import CLASSES, DATA_DIR, CAPTURE_DELAY_MS, SAMPLES_PER_CLASS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura de dataset LSP")
    parser.add_argument(
        "--class", dest="sign_class", required=True,
        choices=CLASSES,
        help=f"Seña a capturar: {CLASSES}",
    )
    parser.add_argument(
        "--samples", type=int, default=SAMPLES_PER_CLASS,
        help=f"Número de imágenes a capturar (default: {SAMPLES_PER_CLASS})",
    )
    return parser.parse_args()


def draw_capture_overlay(
    frame,
    sign_class: str,
    count: int,
    total: int,
    capturing: bool,
    hand_visible: bool,
    fps: float,
) -> None:
    h, w = frame.shape[:2]
    import cv2 as _cv2

    # FPS
    _cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    # Panel superior
    overlay = frame.copy()
    _cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
    _cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Clase actual
    _cv2.putText(frame, f"Seña: {sign_class.upper()}", (12, 28),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Progreso
    progress_text = f"Capturas: {count} / {total}"
    _cv2.putText(frame, progress_text, (12, 58),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Barra de progreso
    bar_w = int((w - 24) * (count / total))
    _cv2.rectangle(frame, (12, 63), (w - 12, 70), (50, 50, 50), -1)
    _cv2.rectangle(frame, (12, 63), (12 + bar_w, 70), (0, 200, 0), -1)

    # Estado de captura
    if not hand_visible:
        msg, color = "Muestra tu mano", (0, 100, 255)
    elif capturing:
        msg, color = "CAPTURANDO...", (0, 255, 0)
    else:
        msg, color = "Presiona ESPACIO para iniciar", (200, 200, 200)

    _cv2.putText(frame, msg, (12, h - 15),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    args = parse_args()
    sign_class = args.sign_class
    total      = args.samples

    save_dir = Path(DATA_DIR) / sign_class
    save_dir.mkdir(parents=True, exist_ok=True)

    # Contar imágenes ya existentes para no sobreescribir
    existing = len(list(save_dir.glob("*.jpg")))
    count    = existing
    if existing > 0:
        print(f"[INFO] Ya existen {existing} imágenes en '{sign_class}'. Continuando desde {existing}.")

    print(f"\n[ETAPA 4] Captura de dataset — Seña: '{sign_class}'")
    print(f"Objetivo: {total} imágenes  |  Directorio: {save_dir}")
    print("Controles: ESPACIO = iniciar/pausar  |  Q = salir\n")

    display    = Display()
    capturing  = False
    last_saved = 0.0
    delay_s    = CAPTURE_DELAY_MS / 1000.0
    cam_start  = int(time.monotonic() * 1000)

    with CameraCapture() as cam, HandDetector(mode="video") as detector:
        while count < total:
            frame = cam.read()
            if frame is None:
                continue

            ts_ms   = int(time.monotonic() * 1000) - cam_start
            result  = detector.detect(frame, ts_ms)
            visible = detector.hand_detected(result)

            display.draw_landmarks(frame, result)
            draw_capture_overlay(frame, sign_class, count, total, capturing, visible, cam.fps)

            # Auto-captura cuando está activa y se detecta mano
            now = time.monotonic()
            if capturing and visible and (now - last_saved) >= delay_s:
                img_path = save_dir / f"{sign_class}_{count:04d}.jpg"
                cv2.imwrite(str(img_path), frame)
                count    += 1
                last_saved = now

                # Flash visual al guardar
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
                cv2.addWeighted(flash, 0.15, frame, 0.85, 0, frame)

            cv2.imshow(f"Capturando LSP — '{sign_class}' (Q=salir)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                capturing = not capturing
                state = "INICIADA" if capturing else "PAUSADA"
                print(f"  Captura {state} — {count}/{total} imágenes")

    cv2.destroyAllWindows()
    print(f"\nCaptura finalizada. Total guardado: {count} imágenes en '{save_dir}'")


if __name__ == "__main__":
    main()
