"""
Módulo de síntesis de voz — Text-to-Speech offline.

Por qué un hilo separado:
  pyttsx3.runAndWait() es bloqueante: puede tardar 500-1500ms según la longitud
  del texto y la voz. Si llamamos speak() desde el loop principal, el FPS cae
  drásticamente durante cada reproducción. El hilo dedicado desacopla el audio
  del video — el loop principal nunca se bloquea.

Arquitectura:
  - Hilo TTS: crea el engine pyttsx3, espera texto vía threading.Event
  - Hilo principal: llama a speak() → verifica cooldown → deposita texto → señaliza event
  - Cooldown por etiqueta: evita repetir la misma seña dentro del ventana de tiempo

Voces disponibles en macOS para español (en orden de preferencia LSP):
  1. Paulina (es-MX) — español latinoamericano, más cercano al español peruano
  2. Mónica  (es-ES) — español peninsular
  3. Eloquence es-MX — cualquier voz Eloquence en español mexicano
  4. Eloquence es-ES — cualquier voz Eloquence en español peninsular
"""

import time
import threading
from config.settings import TTS_COOLDOWN_SECONDS, TTS_RATE, TTS_ENABLED


# Prioridad de voces españolas en macOS (IDs exactos detectados en el sistema)
_VOICE_PRIORITY = [
    "com.apple.voice.compact.es-MX.Paulina",    # Paulina — español latinoamericano
    "com.apple.voice.compact.es-ES.Monica",      # Mónica  — español peninsular
    "com.apple.eloquence.es-MX.Eddy",
    "com.apple.eloquence.es-MX.Rocko",
    "com.apple.eloquence.es-MX.Reed",
    "com.apple.eloquence.es-ES.Flo",
    "com.apple.eloquence.es-ES.Sandy",
]


class Speaker:
    """
    Síntesis de voz offline con cooldown y hilo de audio dedicado.

    Uso:
        speaker = Speaker()
        speaker.speak("Hola")    # No bloqueante
        speaker.stop()           # Al cerrar el sistema
    """

    def __init__(
        self,
        enabled: bool = TTS_ENABLED,
        rate: int = TTS_RATE,
        cooldown: float = TTS_COOLDOWN_SECONDS,
    ) -> None:
        self._enabled  = enabled
        self._rate     = rate
        self._cooldown = cooldown

        # Estado del hilo TTS
        self._pending: str | None = None
        self._lock    = threading.Lock()
        self._event   = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._ready   = False

        # Cooldown: {texto_normalizado → timestamp_último_uso}
        self._last_spoken: dict[str, float] = {}

        if enabled:
            self._running = True
            self._thread  = threading.Thread(target=self._tts_loop, daemon=True, name="tts-worker")
            self._thread.start()
            # Esperar a que el engine esté listo (máx. 3s)
            self._wait_ready(timeout=3.0)

    # ── API pública ──────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Solicita reproducir text. No bloqueante.
        Ignora la solicitud si el mismo texto fue reproducido dentro del cooldown.
        """
        if not self._enabled or not text:
            return

        key = text.strip().lower()
        now = time.monotonic()

        # Verificar cooldown
        if key in self._last_spoken:
            elapsed = now - self._last_spoken[key]
            if elapsed < self._cooldown:
                return

        self._last_spoken[key] = now

        # Depositar texto para el hilo TTS
        with self._lock:
            self._pending = text
        self._event.set()

    def stop(self) -> None:
        """Detiene el hilo TTS de forma ordenada."""
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Hilo TTS ─────────────────────────────────────────────────────────────

    def _tts_loop(self) -> None:
        """Loop del hilo dedicado de audio. Nunca debe llamarse desde el hilo principal."""
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            self._configure_voice(engine)
            engine.setProperty("rate", self._rate)
            self._ready = True
        except Exception as exc:
            print(f"[TTS] Error al inicializar pyttsx3: {exc}")
            print("[TTS] El sistema continuará sin síntesis de voz.")
            self._ready = False
            return

        while self._running:
            # Esperar señal del hilo principal
            self._event.wait(timeout=1.0)
            self._event.clear()

            with self._lock:
                text     = self._pending
                self._pending = None

            if text and self._running:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"[TTS] Error al reproducir '{text}': {exc}")

        if engine:
            try:
                engine.stop()
            except Exception:
                pass

    def _configure_voice(self, engine) -> None:
        """Selecciona la mejor voz española disponible según la lista de prioridad."""
        try:
            voices     = engine.getProperty("voices")
            voice_map  = {v.id: v for v in voices}

            # Buscar por prioridad
            for voice_id in _VOICE_PRIORITY:
                if voice_id in voice_map:
                    engine.setProperty("voice", voice_id)
                    print(f"[TTS] Voz seleccionada: {voice_map[voice_id].name}")
                    return

            # Fallback: cualquier voz con es-MX o es-ES en el ID
            for voice in voices:
                if "es-MX" in voice.id or "es-ES" in voice.id:
                    engine.setProperty("voice", voice.id)
                    print(f"[TTS] Voz fallback: {voice.name}")
                    return

            print("[TTS] No se encontró voz española. Usando voz del sistema por defecto.")

        except Exception as exc:
            print(f"[TTS] No se pudo configurar la voz: {exc}")

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready or not self._thread.is_alive():
                return
            time.sleep(0.05)
