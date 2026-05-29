# Sistema de Reconocimiento de Lengua de Señas Peruana (LSP)

Reconocimiento en tiempo real de señas peruanas usando Computer Vision y Machine Learning.

## Requisitos del sistema

- macOS 12+ (optimizado para Apple Silicon M2)
- Python 3.11+
- Webcam integrada o externa

## Instalación rápida

```bash
# 1. Clonar o descargar el proyecto
cd Lenguaje_Señas

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso

### 1. Capturar dataset LSP
```bash
python scripts/collect_data.py --class hola --samples 200
python scripts/collect_data.py --class gracias --samples 200
python scripts/collect_data.py --class si --samples 200
python scripts/collect_data.py --class no --samples 200
python scripts/collect_data.py --class ayuda --samples 200
```

### 2. Entrenar el modelo
```bash
python scripts/train_model.py
```

### 3. Ejecutar el sistema
```bash
python main.py
```

## Señas reconocidas (LSP v1.0)

| Seña | Lengua de Señas Peruana |
|------|------------------------|
| Hola | Saludo LSP |
| Gracias | Gratitud LSP |
| Sí | Afirmación LSP |
| No | Negación LSP |
| Ayuda | Solicitud de ayuda LSP |

## Arquitectura

```
Webcam → MediaPipe (21 landmarks) → Random Forest → Seña + Confianza → TTS
```

## Tecnologías

- **OpenCV** — captura y visualización de video
- **MediaPipe** — detección de manos y landmarks
- **scikit-learn** — clasificador Random Forest
- **pyttsx3** — síntesis de voz offline

## Documentación técnica

Ver `CLAUDE.md` para arquitectura completa, decisiones técnicas y estado del desarrollo.
