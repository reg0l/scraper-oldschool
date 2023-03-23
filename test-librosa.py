import librosa
import numpy as np
from IPython.display import Audio
import mir_eval.sonify

y , sr = librosa.load_audio("japanese-i-love-you_128bpm_C_major.wav")
Audio(data=y, rate=sr)
