from sympy import true
from ai_models import SpeechRecognitionModel, GPTModel
import audio_process
import speaker
import numpy as np
import pyaudio
from silero_vad import load_silero_vad
import wave
import keyboard
from datetime import datetime
from threading import Thread
import tqdm

def main(frames, language):
  # while True:
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk_size_sample)
  stream.start_stream()
  wf = wave.open("virtual_talk_llama.wav", "wb")
  wf.setnchannels(1)
  wf.setframerate(16000)
  wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))

  while True:
    data = np.frombuffer(stream.read(chunk_size_sample), dtype=np.int16)
    frames.append(data)
    print(frames)
    if keyboard.is_pressed("p"):
      break

  stream.stop_stream()
  stream.close()
  p.terminate()

  print("file is saved!")
  wf.writeframes(b''.join(frames))
  wf.close()
  frames = []

  speech_start = datetime.now()
  if language=="en":
    result = en_speech_recognition_model.transcribe(audio_process.load_audio("virtual_talk_llama.wav"), "en")
  elif language=="vi":
    result = vi_speech_recognition_model.transcribe(audio_process.load_audio("virtual_talk_llama.wav"), "vi")
  speech_end = datetime.now()

  print(result)
  print(f"Speech recognition inference time: {(speech_end-speech_start)}s")

  gpt_start = datetime.now()
  print(gpt_model.generate(result))
  gpt_end = datetime.now()
  print(f"RAG inference time: {(gpt_end-gpt_start)}s")

if __name__=="__main__":
  desired_latency = 0.2
  chunk_size_sec = desired_latency
  sample_rate = 16000
  chunk_size_sample = int(sample_rate*chunk_size_sec)
  model_vad = load_silero_vad(onnx=True)

  vi_speech_recognition_model = SpeechRecognitionModel("vinai/PhoWhisper-large", True)
  en_speech_recognition_model = SpeechRecognitionModel("openai/whisper-large-v3", True)
  gpt_model = GPTModel("meta-llama/Llama-3.2-1B-Instruct", True)
  print("Model is loaded completely!")

  while True:
    frames = []
    choice = input("Please select your language: \n1. English\n2. Vietnamese\nOr type 'e' to exit\nEnter your language: ")
    if choice=='e':
      break
    elif choice=='1':
      print("English speech recognition!")
      main(frames, "en")
    elif choice=='2':
      print("Vietnamese speech recognition!")
      main(frames, "vi")

  # main(frames)
  # gpt_model = GPTModel(r"D:\Testing_all\Meta_Llama-3-8B-Instruct", True)