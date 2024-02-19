import pyaudio
import sounddevice as sd
import numpy as np

threshhold = 75
Clap = False

def detect_clpa(indata, frames, time, status):
      global Clap
      volume_norm = np.linalg.norm(indata)*10
      if volume_norm > threshhold:
            print("Clapped!")
            Clap = True

def Listen_for_claps():
      with sd.InputStream(callback=detect_clpa):
            return sd.sleep(1000)
      
if __name__ == "__main__":
      while True:
            Listen_for_claps()
            if Clap==True:
                break
            else: 
                 pass    