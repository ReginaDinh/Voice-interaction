import pyttsx3

def text_to_speech(text, rate=150, volume=1.0, voice_id=1):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    engine.setProperty('voice', voice_id)
    engine.say(text)
    engine.runAndWait()