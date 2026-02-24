"""
Voice AI — ASR (Whisper) + TTS (Edge-TTS) modülü.

╔══════════════════════════════════════════════════════════════════╗
║  Voice AI Pipeline                                               ║
║                                                                  ║
║  ASR (Automatic Speech Recognition):                            ║
║  Ses → Whisper → Metin                                          ║
║                                                                  ║
║  TTS (Text-to-Speech):                                          ║
║  Metin → Edge-TTS → Ses dosyası (.mp3)                         ║
║                                                                  ║
║  Agent entegrasyonu:                                             ║
║  Kullanıcı ses → ASR → text query → Agent → text answer → TTS  ║
╚══════════════════════════════════════════════════════════════════╝

Whisper Mimarisi (Mülakat notu):

1. Encoder-Decoder Transformer:
   - Encoder: Ses → Mel spectrogram → Transformer encoder → audio features
   - Decoder: Audio features + önceki token'lar → sonraki token (autoregressive)

2. Mel Spectrogram nedir?
   - Ses dalgası → STFT (Short-Time Fourier Transform) → Spectrogram
   - Spectrogram → Mel ölçeği (insan kulağına uygun frekans dağılımı)
   - 80 mel filtre, 30 saniyelik pencereler
   - Sonuç: 2D "resim" → CNN/Transformer ile işlenebilir

3. Whisper Model Boyutları:
   | Model  | Parametre | VRAM  | Hız (rel.) | WER (en) |
   |--------|-----------|-------|------------|----------|
   | tiny   | 39M       | ~1GB  | 32x        | ~7.6%    |
   | base   | 74M       | ~1GB  | 16x        | ~5.0%    |
   | small  | 244M      | ~2GB  | 6x         | ~3.4%    |
   | medium | 769M      | ~5GB  | 2x         | ~2.7%    |
   | large  | 1550M     | ~10GB | 1x         | ~2.1%    |

4. Whisper vs Alternatifler:
   | Özellik         | Whisper      | Google STT   | Azure STT    |
   |-----------------|-------------|--------------|--------------|
   | Dil desteği     | 100+ dil    | 125+ dil     | 100+ dil     |
   | Offline         | ✅ Evet      | ❌ Hayır     | ❌ Hayır     |
   | Maliyet         | Ücretsiz    | $0.006/15s   | $0.016/dakika|
   | Doğruluk (en)   | ~2-5% WER   | ~4-5% WER   | ~3-5% WER   |
   | Real-time       | Batch only  | Streaming ✅  | Streaming ✅  |

   → Whisper: Offline, ücretsiz, batch processing için ideal
   → Google/Azure: Real-time streaming gerekiyorsa

5. Edge-TTS nedir?
   - Microsoft Edge'in TTS servisini kullanan Python kütüphanesi
   - Ücretsiz (API key gerektirmez)
   - 300+ ses, 80+ dil (Türkçe dahil)
   - Yüksek kalite (neural TTS)
   - Alternatifler: OpenAI TTS ($15/1M char), Bark (offline, yavaş), gTTS (düşük kalite)
"""

import os
import asyncio
import tempfile
from typing import Optional


class WhisperASR:
    """
    Whisper ASR — Ses dosyasını metne çevirir.

    Kullanım:
        asr = WhisperASR(model_size="base")
        text = asr.transcribe("audio.wav")
        # → "Bu sahnede kaç araç var?"

    Mülakat notu:
    - Whisper modeli lazy-loaded (ilk kullanımda indirilir)
    - fp16=True GPU'da hızlı ama CPU'da False olmalı
    - language="tr" Türkçe zorlama — auto-detect de yapabilir
    - Beam search (beam_size=5) → daha doğru ama daha yavaş
    """

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
                        - tiny/base: Hızlı, demo için yeterli
                        - small: İyi denge (hız/doğruluk)
                        - medium/large: En doğru ama yavaş + çok VRAM
        """
        self.model_size = model_size
        self._model = None

    def _ensure_loaded(self):
        """Whisper modelini lazy load et."""
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
        Ses dosyasını metne çevir.

        Args:
            audio_path: Ses dosyası yolu (.wav, .mp3, .m4a, .flac, vb.)
            language: Dil kodu ("tr", "en", vb.) — None → auto-detect
            task: "transcribe" (aynı dilde) veya "translate" (İngilizce'ye çevir)

        Returns:
            {
                "text": "Tam transkript",
                "language": "tr",
                "segments": [{"start": 0.0, "end": 2.5, "text": "..."}],
                "duration": 5.3
            }

        Mülakat notu:
        - task="translate": Herhangi bir dilden İngilizce'ye çevirme
          (Whisper'ın özel yeteneği — tek modelde hem ASR hem çeviri)
        - segments: Zaman damgalı çıktı → altyazı, video senkronizasyonu
        - fp16=False: CPU'da çalışırken gerekli (MPS/CUDA'da True)
        """
        self._ensure_loaded()

        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}

        print(f"[Whisper] Transcribing: {audio_path}")

        # CPU mu GPU mu kontrol et
        import torch
        fp16 = torch.cuda.is_available()  # MPS'te de False olmalı

        options = {
            "fp16": fp16,
            "task": task,
        }
        if language:
            options["language"] = language

        result = self._model.transcribe(audio_path, **options)

        # Süre hesapla
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
        Ses dosyasının dilini tespit et (transcribe yapmadan).

        İlk 30 saniyeyi analiz eder.
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
    Edge-TTS — Metni sese çevirir (Microsoft Edge neural TTS).

    Kullanım:
        tts = EdgeTTS()
        await tts.synthesize("Merhaba dünya", "output.mp3")
        # veya senkron:
        tts.synthesize_sync("Merhaba dünya", "output.mp3")

    Mülakat notu:
    - Edge-TTS ücretsiz ve API key gerektirmez
    - Neural TTS: doğal ses kalitesi (eski concatenative TTS'ten çok daha iyi)
    - SSML desteği: konuşma hızı, tonlama, vurgu kontrol edilebilir
    - Async API: aiohttp ile Microsoft sunucularına bağlanır
    """

    # Popüler Türkçe ve İngilizce sesler
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
            voice: Ses adı — VOICES dict'indeki key veya doğrudan voice ID
                   Örnek: "tr_female", "en_male", "en-US-JennyNeural"
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
        Metni sese çevir (async).

        Args:
            text: Okunacak metin
            output_path: Çıktı dosya yolu (.mp3)
            rate: Konuşma hızı ("+20%" → daha hızlı, "-20%" → daha yavaş)
            volume: Ses seviyesi
            pitch: Ses tonu

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

        # Tahmini süre (ortalama 150 kelime/dk = 2.5 kelime/sn)
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
        Metni sese çevir (senkron wrapper).

        asyncio.run() ile async fonksiyonu senkron çağırır.
        Agent node'ları senkron olduğu için bu wrapper gerekli.
        """
        # Event loop zaten çalışıyorsa (Jupyter, vb.) nest_asyncio gerekebilir
        try:
            loop = asyncio.get_running_loop()
            # Zaten bir event loop varsa, yeni thread'de çalıştır
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    self.synthesize(text, output_path, rate, volume, pitch)
                ).result()
            return result
        except RuntimeError:
            # Event loop yok — normal asyncio.run kullan
            return asyncio.run(
                self.synthesize(text, output_path, rate, volume, pitch)
            )

    @staticmethod
    async def list_voices(language: str = "tr") -> list[dict]:
        """
        Kullanılabilir sesleri listele.

        Args:
            language: Dil filtresi ("tr", "en", vb.)

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
    Birleşik Voice Assistant — ASR + Agent + TTS pipeline.

    Tam akış:
    1. Kullanıcı ses dosyası verir
    2. Whisper → metin (ASR)
    3. Metin → Agent graph → cevap
    4. Cevap → Edge-TTS → ses dosyası (TTS)

    Mülakat notu:
    - End-to-end voice pipeline: ASR → NLU → Agent → NLG → TTS
    - Latency bileşenleri: ASR (~1-3s) + Agent (~3-8s) + TTS (~1-2s) = ~5-13s
    - Real-time için: streaming ASR (Whisper desteklemez) + streaming TTS
    - Production'da: WebSocket ile chunk-based streaming
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
        Ses girişini işle → Agent'a gönder → Ses çıktısı üret.

        Args:
            audio_path: Giriş ses dosyası
            image_path: Opsiyonel görüntü (multi-modal analiz için)
            output_audio_path: TTS çıktı yolu (None → otomatik)

        Returns:
            {
                "transcription": {...},
                "agent_response": "...",
                "tts_output": {...}
            }
        """
        # 1. ASR — Ses → Metin
        print(f"\n{'='*60}")
        print(f"🎤 VOICE ASSISTANT — Processing")
        print(f"{'='*60}")

        transcription = self.asr.transcribe(audio_path, language="tr")
        if "error" in transcription:
            return {"error": transcription["error"]}

        user_query = transcription["text"]
        print(f"📝 Transkript: \"{user_query}\"")

        # 2. Agent — Metin → Cevap
        from .state import create_initial_state
        from .graph import build_agent_graph

        state = create_initial_state(
            user_query=user_query,
            image_path=image_path,
            max_iterations=3
        )

        graph = build_agent_graph(with_memory=False)
        result = graph.invoke(state)
        agent_answer = result.get("final_answer", "Cevap üretilemedi.")
        print(f"💬 Agent cevabı: \"{agent_answer[:200]}...\"")

        # 3. TTS — Cevap → Ses
        if output_audio_path is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_audio_path = os.path.join(output_dir, "response.mp3")

        # Sadece ana cevabı seslendir (metadata kısmını değil)
        # İlk satırı al (📎 ve 📊 satırlarını atla)
        clean_answer = agent_answer.split("\n📎")[0].split("\n📊")[0].strip()

        tts_result = self.tts.synthesize_sync(clean_answer, output_audio_path)
        print(f"🔊 TTS çıktısı: {output_audio_path}")

        return {
            "transcription": transcription,
            "user_query": user_query,
            "agent_response": agent_answer,
            "tts_output": tts_result
        }
