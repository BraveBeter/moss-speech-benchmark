# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import soundfile as sf
import torch
import torchaudio
SAMPLE_RATE = 16000

class UTMOSScorer:
    def __init__(self, device, model_path):
        self.device = device
        self.predictor = torch.jit.load(str(model_path), map_location=device).eval()

    def score_batch(self, wav_paths: list[str]) -> list[float]:
        """Score a batch of wav files.

        Each file is loaded, resampled to 16 kHz if needed, converted to
        int16, and scored independently. Returns a list of float MOS scores
        in [1, 5].
        """
        scores: list[float] = []
        for path in wav_paths:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            wave = torch.from_numpy(audio.T.copy())
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
            wave_int16 = (wave * 32767).clamp(-32768, 32767).to(torch.int16)
            wave_int16 = wave_int16.to(self.device)
            with torch.no_grad():
                score = self.predictor(wave_int16)
            scores.append(float(score.mean().item()))
        return scores
