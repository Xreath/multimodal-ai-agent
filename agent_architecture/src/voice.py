"""
Voice AI — ASR (Whisper) + TTS (Edge-TTS) module.

╔══════════════════════════════════════════════════════════════════╗
║  Voice AI Pipeline                                               ║
║                                                                  ║
║  ASR (Automatic Speech Recognition):                             ║
║  Audio → Whisper → Text                                          ║
║                                                                  ║
║  TTS (Text-to-Speech):                                           ║
║  Text → Edge-TTS → Audio file (.mp3)                             ║
║                                                                  ║
║  Agent integration:                                              ║
║  User audio → ASR → text query → Agent → text answer → TTS       ║
╚══════════════════════════════════════════════════════════════════╝

Whisper Architecture (Interview note):

1. Encoder-Decoder Transformer:
   - Encoder: Audio → Mel spectrogram → Transformer encoder → audio features
   - Decoder: Audio features + previous tokens → next token (autoregressive)

2. What is Mel Spectrogram?
   - Audio wave → STFT (Short-Time Fourier Transform) → Spectrogram
   - Spectrogram → Mel scale (frequency distribution matching human ear)
   - 80 mel filters, 30-second windows
   - Result: 2D "image" → can be processed with CNN/Transformer

3. Whisper Model Sizes:
   | Model  | Parameters | VRAM  | Speed (rel)| WER (en) |
   |--------|------------|-------|------------|----------|
   | tiny   | 39M        | ~1GB  | 32x        | ~7.6%    |
   | base   | 74M        | ~1GB  | 16x        | ~5.0%    |
   | small  | 244M       | ~2GB  | 6x         | ~3.4%    |
   | medium | 769M       | ~5GB  | 2x         | ~2.7%    |
   | large  | 1550M      | ~10GB | 1x         | ~2.1%    |

4. Whisper vs Alternatives:
   | Feature         | Whisper      | Google STT   | Azure STT    |
   |-----------------|--------------|--------------|--------------|
   | Language support| 100+ langs   | 125+ langs   | 100+ langs   |
   | Offline         | ✅ Yes        | ❌ No         | ❌ No         |
   | Cost            | Free         | $0.006/15s   | $0.016/minute|
   | Accuracy (en)   | ~2-5% WER    | ~4-5% WER    | ~3-5% WER    |
   | Real-time       | Batch only   | Streaming ✅  | Streaming ✅  |

   → Whisper: Offline, free, ideal for batch processing
   → Google/Azure: If real-time streaming is required

5. What is Edge-TTS?
   - Python library using Microsoft Edge's TTS service
   - Free (no API key required)
   - 300+ voices, 80+ languages
   - High quality (neural TTS)
   - Alternatives: OpenAI TTS ($15/1M char), Bark (offline, slow), gTTS (low quality)
"""

import os
import asyncio
import tempfile
from typing import Optional


class WhisperASR:
    """
    Whisper ASR — Converts audio file to text.

    Usage:
        asr = WhisperASR(model_size="base")
        text = asr.transcribe("audio.wav")
        # → "How many vehicles are in this scene?"

    Interview note:
    - Whisper model is lazy-loaded (downloaded on first use)
    - fp16=True is fast on GPU but must be False on CPU
    - language="tr" forces Turkish — can also auto-detect
    - Beam search (beam_size=5) → more accurate but slower
    """

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
                        - tiny/base: Fast, enough for demo
                        - small: Good balance (speed/accuracy)
                        - medium/large: Most accurate but slow + lots of VRAM
        """
        self.model_size = model_size
        self._model = None

    def _ensure_loaded(self):
        """Lazy load the Whisper model."""
        if self._model is not None:
            return

        import whisper
        print(f"[Whisper] Loading '{self.model_size}' model...")
        self._model = whisper.load_model(self.model_size)
        print(f"[Whisper] Model loaded ✅")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> dict:
        """
        Convert audio file to text.

        Args:
            audio_path: Audio file path (.wav, .mp3, .m4a, .flac, etc.)
            language: Language code ("tr", "en", etc.) — None → auto-detect
            task: "transcribe" (same language) or "translate" (translate to English)

        Returns:
            {
                "text": "Full transcript",
                "language": "tr",
                "segments": [{"start": 0.0, "end": 2.5, "text": "..."}],
                "duration": 5.3
            }

        Interview note:
        - task="translate": Translating from any language to English
          (Whisper's special capability — both ASR and translation in a single model)
        - segments: Time-stamped output → subtitles, video synchronization
        - fp16=False: Required when running on CPU (True on MPS/CUDA)
        """
        self._ensure_loaded()

        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}

        print(f"[Whisper] Transcribing: {audio_path}")

        # Check if CPU or GPU
        import torch
        fp16 = torch.cuda.is_available()  # Should also be False on MPS

        options = {
            "fp16": fp16,
            "task": task,
        }
        if language:
            options["language"] = language

        result = self._model.transcribe(audio_path, **options)

        # Calculate duration
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0

        output = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"].strip()
                }
                for s in segments
            ],
            "duration": round(duration, 2)
        }

        print(f"[Whisper] Done — {output['language']}, {output['duration']}s, "
              f"{len(output['text'])} chars")

        return output

    def detect_language(self, audio_path: str) -> dict:
        """
        Detect the language of the audio file (without transcribing).

        Analyzes the first 30 seconds.
        """
        self._ensure_loaded()
        import whisper

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self._model.device)

        _, probs = self._model.detect_language(mel)
        top_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "detected_language": top_langs[0][0],
            "confidence": round(top_langs[0][1], 4),
            "top_5": {lang: round(prob, 4) for lang, prob in top_langs}
        }


class EdgeTTS:
    """
    Edge-TTS — Converts text to speech (Microsoft Edge neural TTS).

    Usage:
        tts = EdgeTTS()
        await tts.synthesize("Hello world", "output.mp3")
        # or synchronous:
        tts.synthesize_sync("Hello world", "output.mp3")

    Interview note:
    - Edge-TTS is free and requires no API key
    - Neural TTS: natural voice quality (much better than old concatenative TTS)
    - SSML support: speech rate, intonation, emphasis can be controlled
    - Async API: connects to Microsoft servers with aiohttp
    """

    # Popular Turkish and English voices
    VOICES = {
        "tr_female": "tr-TR-EmelNeural",
        "tr_male": "tr-TR-AhmetNeural",
        "en_female": "en-US-JennyNeural",
        "en_male": "en-US-GuyNeural",
        "en_aria": "en-US-AriaNeural",
    }

    def __init__(self, voice: str = "tr_female"):
        """
        Args:
            voice: Voice name — key in VOICES dict or direct voice ID
                   Example: "tr_female", "en_male", "en-US-JennyNeural"
        """
        self.voice = self.VOICES.get(voice, voice)

    async def synthesize(
        self,
        text: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ) -> dict:
        """
        Convert text to speech (async).

        Args:
            text: Text to read
            output_path: Output file path (.mp3)
            rate: Speech rate ("+20%" → faster, "-20%" → slower)
            volume: Volume level
            pitch: Voice pitch

        Returns:
            {"output_path": "...", "voice": "...", "text_length": 42, "duration_estimate": 3.5}
        """
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=rate,
            volume=volume,
            pitch=pitch
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        await communicate.save(output_path)

        # Estimated duration (average 150 words/min = 2.5 words/sec)
        word_count = len(text.split())
        duration_estimate = round(word_count / 2.5, 1)

        return {
            "output_path": output_path,
            "voice": self.voice,
            "text_length": len(text),
            "word_count": word_count,
            "duration_estimate_seconds": duration_estimate
        }

    def synthesize_sync(
        self,
        text: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ) -> dict:
        """
        Convert text to speech (synchronous wrapper).

        Calls the async function synchronously using asyncio.run().
        This wrapper is necessary because Agent nodes are synchronous.
        """
        # nest_asyncio might be needed if the event loop is already running (Jupyter, etc.)
        try:
            loop = asyncio.get_running_loop()
            # If an event loop already exists, run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    self.synthesize(text, output_path, rate, volume, pitch)
                ).result()
            return result
        except RuntimeError:
            # No event loop — use regular asyncio.run
            return asyncio.run(
                self.synthesize(text, output_path, rate, volume, pitch)
            )

    @staticmethod
    async def list_voices(language: str = "tr") -> list[dict]:
        """
        List available voices.

        Args:
            language: Language filter ("tr", "en", etc.)

        Returns:
            [{"name": "tr-TR-EmelNeural", "gender": "Female", "locale": "tr-TR"}]
        """
        import edge_tts

        voices = await edge_tts.list_voices()
        filtered = [
            {
                "name": v["ShortName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
            }
            for v in voices
            if v["Locale"].startswith(language)
        ]
        return filtered


class VoiceAssistant:
    """
    Unified Voice Assistant — ASR + Agent + TTS pipeline.

    Full flow:
    1. User provides audio file
    2. Whisper → text (ASR)
    3. Text → Agent graph → answer
    4. Answer → Edge-TTS → audio file (TTS)

    Interview note:
    - End-to-end voice pipeline: ASR → NLU → Agent → NLG → TTS
    - Latency components: ASR (~1-3s) + Agent (~3-8s) + TTS (~1-2s) = ~5-13s
    - For real-time: streaming ASR (Whisper doesn't support) + streaming TTS
    - In production: chunk-based streaming with WebSocket
    """

    def __init__(
        self,
        whisper_model: str = "base",
        tts_voice: str = "tr_female"
    ):
        self.asr = WhisperASR(model_size=whisper_model)
        self.tts = EdgeTTS(voice=tts_voice)

    def process_voice_query(
        self,
        audio_path: str,
        image_path: Optional[str] = None,
        output_audio_path: Optional[str] = None
    ) -> dict:
        """
        Process voice input → Send to Agent → Produce voice output.

        Args:
            audio_path: Input audio file
            image_path: Optional image (for multi-modal analysis)
            output_audio_path: TTS output path (None → auto)

        Returns:
            {
                "transcription": {...},
                "agent_response": "...",
                "tts_output": {...}
            }
        """
        # 1. ASR — Audio → Text
        print(f"\n{'='*60}")
        print(f"🎤 VOICE ASSISTANT — Processing")
        print(f"{'='*60}")

        transcription = self.asr.transcribe(audio_path)
        if "error" in transcription:
            return {"error": transcription["error"]}

        user_query = transcription["text"]
        print(f"📝 Transcript: \"{user_query}\"")

        # 2. Agent — Text → Answer
        from .state import create_initial_state
        from .graph import build_agent_graph

        state = create_initial_state(
            user_query=user_query,
            image_path=image_path,
            max_iterations=3
        )

        graph = build_agent_graph(with_memory=False)
        result = graph.invoke(state)
        agent_answer = result.get("final_answer", "Could not generate answer.")
        print(f"💬 Agent answer: \"{agent_answer[:200]}...\"")

        # 3. TTS — Answer → Audio
        if output_audio_path is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_audio_path = os.path.join(output_dir, "response.mp3")

        # Vocalize only the main answer (not the metadata part)
        # Get the first line (skip 📎 and 📊 lines)
        clean_answer = agent_answer.split("\n📎")[0].split("\n📊")[0].strip()

        tts_result = self.tts.synthesize_sync(clean_answer, output_audio_path)
        print(f"🔊 TTS output: {output_audio_path}")

        return {
            "transcription": transcription,
            "user_query": user_query,
            "agent_response": agent_answer,
            "tts_output": tts_result
        }
