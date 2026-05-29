"""
Script de verificación del entorno de desarrollo.
Ejecutar antes de comenzar a trabajar con el sistema LSP.

Uso:
    python scripts/verify_setup.py
"""

import sys
import platform


def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 11)
    status = "OK" if ok else "FALLO"
    print(f"  [{status}] Python {version.major}.{version.minor}.{version.micro}")
    return ok


def check_platform() -> bool:
    system = platform.system()
    machine = platform.machine()
    ok = system == "Darwin"
    status = "OK" if ok else "ADVERTENCIA"
    print(f"  [{status}] Plataforma: {system} {machine}")
    return ok


def check_imports() -> bool:
    packages = {
        "cv2":        "opencv-python",
        "mediapipe":  "mediapipe",
        "numpy":      "numpy",
        "sklearn":    "scikit-learn",
        "joblib":     "joblib",
        "pyttsx3":    "pyttsx3",
        "matplotlib": "matplotlib",
        "pandas":     "pandas",
        "tqdm":       "tqdm",
    }

    all_ok = True
    for module, pip_name in packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "?")
            print(f"  [OK] {pip_name} == {version}")
        except ImportError:
            print(f"  [FALLO] {pip_name} — ejecuta: pip install {pip_name}")
            all_ok = False

    return all_ok


def check_camera() -> bool:
    import cv2
    cap = cv2.VideoCapture(0)
    ok = cap.isOpened()
    if ok:
        ret, frame = cap.read()
        ok = ret and frame is not None
        if ok:
            h, w = frame.shape[:2]
            print(f"  [OK] Webcam detectada — resolución: {w}x{h}")
        else:
            print("  [FALLO] Webcam abierta pero no entrega frames")
    else:
        print("  [FALLO] No se detectó webcam en índice 0")
        print("         → En macOS: System Settings > Privacy > Camera → activar Terminal")
    cap.release()
    return ok


def check_mediapipe() -> bool:
    """Verifica la nueva Tasks API de MediaPipe (0.10.14+)."""
    import mediapipe as mp
    from pathlib import Path
    try:
        from mediapipe.tasks.python import vision
        # Verificar que HandLandmarker está disponible
        assert hasattr(vision, "HandLandmarker"), "HandLandmarker no encontrado"

        # Verificar que el modelo .task existe
        sys.path.insert(0, ".")
        from config.settings import HAND_LANDMARKER_PATH
        model_path = Path(HAND_LANDMARKER_PATH)
        if not model_path.exists():
            print(f"  [FALLO] Modelo no encontrado: {model_path}")
            print("         → Ejecuta: curl -L <url> -o models/hand_landmarker.task")
            return False

        # Inicializar HandLandmarker en modo imagen estática para el test
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        landmarker.close()
        print(f"  [OK] MediaPipe HandLandmarker (Tasks API) inicializado — modelo: {model_path.name}")
        return True
    except Exception as e:
        print(f"  [FALLO] MediaPipe Tasks API: {e}")
        return False


def check_config() -> bool:
    try:
        sys.path.insert(0, ".")
        from config.settings import CLASSES, MODEL_PATH, DATA_DIR, HAND_LANDMARKER_PATH
        print(f"  [OK] config/settings.py cargado — clases LSP: {CLASSES}")
        return True
    except ImportError as e:
        print(f"  [FALLO] config/settings.py: {e}")
        return False


def main():
    print("\n" + "=" * 55)
    print("  Verificación del entorno — LSP Recognition System")
    print("=" * 55)

    print("\n[1] Python y plataforma")
    py_ok   = check_python()
    plat_ok = check_platform()

    print("\n[2] Dependencias")
    deps_ok = check_imports()

    print("\n[3] Webcam")
    cam_ok  = check_camera()

    print("\n[4] MediaPipe Tasks API")
    mp_ok   = check_mediapipe()

    print("\n[5] Configuración del proyecto")
    cfg_ok  = check_config()

    print("\n" + "=" * 55)
    all_ok = py_ok and deps_ok and mp_ok and cfg_ok
    if cam_ok and all_ok:
        print("  RESULTADO: Entorno listo. Puedes continuar con la ETAPA 3.")
    elif all_ok:
        print("  RESULTADO: Todo listo excepto webcam (requiere permiso en macOS).")
        print("  → Ve a: Ajustes del Sistema > Privacidad > Cámara > activar Terminal")
    else:
        print("  RESULTADO: Hay problemas que resolver antes de continuar.")
    print("=" * 55 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
