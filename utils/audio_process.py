import librosa
import numpy as np
import torch
from silero_vad import get_speech_timestamps

def load_audio(path):
  y, sr = librosa.load(path, dtype=np.float32, sr=16000)
  return y

def process_audio(model_vad, audio):
  audio /= np.max(np.abs(audio))
  wav = torch.from_numpy(audio)
  speech_timestamps = get_speech_timestamps(wav, model_vad, threshold=0.8)
  if speech_timestamps != []:
    print("Start: ", speech_timestamps[0]["start"], " - End: ", speech_timestamps[0]["end"])
    return audio[int(speech_timestamps[0]["start"]):int(speech_timestamps[0]["end"])]
  else:
    pass