"""
Benchmark del pipeline LSP — mide latencia real por etapa.

Genera un informe de rendimiento sin necesitar el modelo entrenado ni la webcam.
Usa frames sintéticos y mide cada etapa del pipeline de forma aislada.

Uso:
    python scripts/benchmark.py
    python scripts/benchmark.py --frames 200
"""

import argparse
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, ".")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark del pipeline LSP")
    p.add_argument("--frames", type=int, default=100, help="Número de frames a medir")
    return p.parse_args()


def bench(fn, n: int) -> tuple[float, float, float]:
    """Ejecuta fn() n veces y devuelve (mean_ms, min_ms, max_ms)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def section(title: str) -> None:
    print(f"\n  {'─' * 50}")
    print(f"  {title}")
    print(f"  {'─' * 50}")


def row(label: str, mean: float, mn: float, mx: float) -> None:
    fps = 1000 / mean if mean > 0 else 0
    print(f"  {label:<28} {mean:>7.2f}ms  {mn:>7.2f}ms  {mx:>7.2f}ms  (~{fps:.0f} FPS)")


def main() -> None:
    args = parse_args()
    N = args.frames

    print("\n" + "=" * 60)
    print("  Benchmark — LSP Recognition System (Apple M2)")
    print(f"  Frames por prueba: {N}")
    print("=" * 60)

    # ── 1. Captura de frame (simulada) ───────────────────────────────────────
    section("1. Generación de frame (baseline sin webcam)")
    frame_640 = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
    mean, mn, mx = bench(lambda: np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8), N)
    print(f"\n  {'Operación':<28} {'Media':>8} {'Mín':>8} {'Máx':>8}  {'Equiv.':>8}")
    print(f"  {'─'*62}")
    row("Crear frame sintético 640x480", mean, mn, mx)

    # ── 2. MediaPipe HandLandmarker ──────────────────────────────────────────
    section("2. MediaPipe HandLandmarker (núcleo del sistema)")
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
    )
    lm = HandLandmarker.create_from_options(options)

    frame_rgb = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB)
    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Warmup
    for i in range(10):
        lm.detect_for_video(mp_img, i)

    ts_counter = [100]
    def detect_fn():
        ts_counter[0] += 1
        lm.detect_for_video(mp_img, ts_counter[0])

    mean_detect, mn_detect, mx_detect = bench(detect_fn, N)
    print(f"\n  {'Operación':<28} {'Media':>8} {'Mín':>8} {'Máx':>8}  {'Equiv.':>8}")
    print(f"  {'─'*62}")
    row("detect_for_video() 640x480", mean_detect, mn_detect, mx_detect)

    # Frame skipping: simular 1 de cada 2 frames
    skip_times = []
    for i in range(N):
        t0 = time.perf_counter()
        if i % 2 == 0:
            ts_counter[0] += 1
            lm.detect_for_video(mp_img, ts_counter[0])
        skip_times.append((time.perf_counter() - t0) * 1000)
    mean_skip = float(np.mean(skip_times))
    row(f"Con skip=1 (media por frame)", mean_skip, float(np.min(skip_times)), float(np.max(skip_times)))

    lm.close()

    gain = mean_detect / mean_skip if mean_skip > 0 else 1.0
    print(f"\n  → Ganancia con frame skipping: {gain:.1f}x  "
          f"({1000/mean_detect:.0f} → {1000/mean_skip:.0f} FPS)")

    # ── 3. Feature extraction ────────────────────────────────────────────────
    section("3. FeatureExtractor (normalización de landmarks)")
    from src.features.extractor import FeatureExtractor
    extractor   = FeatureExtractor()
    fake_lm     = np.random.rand(21, 3).astype(np.float32)
    mean_fe, mn_fe, mx_fe = bench(lambda: extractor.extract(fake_lm), N)
    print(f"\n  {'Operación':<28} {'Media':>8} {'Mín':>8} {'Máx':>8}  {'Equiv.':>8}")
    print(f"  {'─'*62}")
    row("extract() 21×3 → 63 features", mean_fe, mn_fe, mx_fe)

    # ── 4. Dibujo (OpenCV) ───────────────────────────────────────────────────
    section("4. Renderizado OpenCV (landmarks + panel)")
    from src.ui.display import Display
    display = Display()

    class FakeResult:
        hand_landmarks = None
        handedness = None

    def draw_fn():
        f = frame_640.copy()
        display.draw_fps(f, 30.0)
        display.draw_info_panel(f, sign="Hola", confidence=0.92, is_stable=True)

    mean_draw, mn_draw, mx_draw = bench(draw_fn, N)
    print(f"\n  {'Operación':<28} {'Media':>8} {'Mín':>8} {'Máx':>8}  {'Equiv.':>8}")
    print(f"  {'─'*62}")
    row("draw_fps + draw_info_panel", mean_draw, mn_draw, mx_draw)

    # ── 5. Resumen del pipeline ──────────────────────────────────────────────
    section("5. Resumen del pipeline completo")
    total_with_skip   = mean_skip + mean_fe + mean_draw
    total_without_skip = mean_detect + mean_fe + mean_draw

    print(f"""
  {'Etapa':<32} {'Sin skip':>10}  {'Con skip':>10}
  {'─'*56}
  MediaPipe detect            {mean_detect:>8.2f}ms  {mean_skip:>8.2f}ms
  Feature extraction          {mean_fe:>8.2f}ms  {mean_fe:>8.2f}ms
  Renderizado OpenCV          {mean_draw:>8.2f}ms  {mean_draw:>8.2f}ms
  {'─'*56}
  TOTAL estimado              {total_without_skip:>8.2f}ms  {total_with_skip:>8.2f}ms
  FPS teórico máximo          {1000/total_without_skip:>8.0f}     {1000/total_with_skip:>8.0f}
""")

    note = "* FPS real será menor por overhead de threading, imshow y waitKey."
    print(f"  {note}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
