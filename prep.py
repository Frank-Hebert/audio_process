# SOURCE : https://www.youtube.com/watch?v=Oa_d-zaUti8&list=WL&index=5

import librosa, librosa.display
import matplotlib.pyplot as plt

file = "blues.00000.wav"

# ? waveform

signal, sr = librosa.load(
    file, sr=22050
)  # sr = sample rate -> sr * T -> 22050 * 30 sec
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
