# PROJECT ROADMAP — LSP Recognition System

## Vision general

Sistema modular y escalable para reconocimiento de Lengua de Señas Peruana (LSP) en tiempo real.

---

## Versión 1.0 — Sistema base (5 señas LSP)

### ETAPA 1 — Planeamiento y arquitectura ✅
- [x] Definir arquitectura del sistema
- [x] Definir módulos y flujo de datos
- [x] Definir estructura de carpetas
- [x] Documentar decisiones técnicas
- [x] Crear CLAUDE.md, README.md, PROJECT_ROADMAP.md

### ETAPA 2 — Configuración del entorno ⬜
- [ ] Crear estructura de carpetas
- [ ] Crear `config/settings.py`
- [ ] Crear `requirements.txt` con versiones exactas
- [ ] Verificar compatibilidad macOS ARM
- [ ] Test de webcam con OpenCV
- [ ] Test de MediaPipe

### ETAPA 3 — Detección de manos ⬜
- [ ] Implementar `src/capture/camera.py` (webcam + threading + FPS)
- [ ] Implementar `src/detection/hand_detector.py` (MediaPipe)
- [ ] Implementar `src/ui/display.py` (overlay básico)
- [ ] Probar visualización de landmarks en tiempo real
- [ ] Validar FPS objetivo: ≥25 FPS en M2

### ETAPA 4 — Dataset LSP ⬜
- [ ] Implementar `scripts/collect_data.py`
- [ ] Capturar 200+ imágenes por seña LSP
- [ ] Implementar `src/features/extractor.py` (normalización)
- [ ] Generar `data/processed/landmarks.csv`
- [ ] Validar calidad del dataset

### ETAPA 5 — Entrenamiento del modelo ⬜
- [ ] Implementar `src/recognition/model.py`
- [ ] Implementar `scripts/train_model.py`
- [ ] Implementar `scripts/evaluate_model.py`
- [ ] Entrenar Random Forest con dataset LSP
- [ ] Validar precisión objetivo: ≥90% en test set
- [ ] Guardar modelo en `models/sign_classifier.pkl`

### ETAPA 6 — Predicción en tiempo real ⬜
- [ ] Implementar `src/recognition/predictor.py`
- [ ] Integrar predictor en el loop principal
- [ ] Mostrar seña detectada y porcentaje de confianza
- [ ] Aplicar threshold de confianza (0.75)

### ETAPA 7 — Text-to-Speech ⬜
- [ ] Implementar `src/tts/speaker.py`
- [ ] Integrar pyttsx3 con voz en español
- [ ] Implementar cooldown anti-repetición (3s)
- [ ] Probar en macOS con voz del sistema

### ETAPA 8 — Optimización ⬜
- [ ] Optimizar FPS (threading, skip frames)
- [ ] Reducir latencia de predicción
- [ ] Mejorar precisión con data augmentation
- [ ] Reducir falsos positivos con historial de predicciones

### ETAPA 9 — Interfaz final ⬜
- [ ] Diseñar overlay profesional con OpenCV
- [ ] Panel lateral con: seña, confianza, FPS, estado
- [ ] Historial de últimas señas detectadas
- [ ] Indicador visual de confianza (barra)
- [ ] Crear `main.py` final integrado

---

## Versión 2.0 — Expansión (planificación futura)

- [ ] Ampliar a 20+ señas LSP
- [ ] Alfabeto dactilológico peruano completo
- [ ] Señas dinámicas con LSTM/Transformer
- [ ] Colaboración con comunidad sorda peruana
- [ ] Interfaz gráfica con PyQt6
- [ ] App standalone con PyInstaller

---

## Métricas de éxito v1.0

| Métrica | Objetivo |
|---------|----------|
| FPS en tiempo real | ≥ 25 FPS |
| Precisión del modelo | ≥ 90% |
| Latencia de predicción | < 50ms |
| Confianza mínima útil | ≥ 75% |
| Señas reconocidas | 5 (LSP) |

---

## Referencias LSP

- Instituto Nacional de Rehabilitación (INR) — Perú
- PUCP — Investigaciones sobre LSP
- Diccionario de Lengua de Señas Peruana (MINEDU)
