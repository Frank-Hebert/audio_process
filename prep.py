# SOURCE : https://www.youtube.com/watch?v=Oa_d-zaUti8&list=WL&index=5

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


file = "blues.00000.wav"

# ? waveform

signal, sr = librosa.load(
    file, sr=22050
)  # sr = sample rate -> sr * T -> 22050 * 30 sec
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

# fft -> power spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[: int(len(frequency) / 2)]
left_magnitude = magnitude[: int(len(frequency) / 2)]


# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

#  stft (short time fourrier transform) -> spectogram
n_fft = 2048  # number of sample considered in the fourrier transform
hop_length = 512  # number of sample shifting each fourier transform to the right

stft = librosa.core.stft(signal, n_fft=n_fft, hop_length=hop_length)

spectrogram = np.abs(stft)


log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)

# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

# MFCCs
MFFCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)

plt.xlabel("Time")
plt.ylabel("MFFC")
plt.colorbar()
plt.show()
