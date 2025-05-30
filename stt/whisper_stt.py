import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import pipeline

# 모델 및 디바이스 설정
model_id = "INo0121/whisper-base-ko-callvoice"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Whisper 파이프라인 생성
stt_pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device
)

# 마이크로 음성 녹음 (예: 5초)
duration = 5  # 녹음 시간(초)
samplerate = 16000  # Whisper는 16kHz 권장

print("녹음을 시작합니다. 말하세요...")
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
sd.wait()
print("녹음이 완료되었습니다.")

# numpy array를 int16으로 변환 후 임시 wav 파일로 저장
wav_path = "temp_input.wav"
write(wav_path, samplerate, (recording * 32767).astype(np.int16))

# Whisper로 음성 인식
result = stt_pipe(wav_path)
print("인식 결과:", result["text"])
