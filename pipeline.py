import librosa
import matplotlib.pyplot as plt
from torchaudio import transforms as T
from torchaudio.utils import download_asset
import torchaudio
import os

#SAMPLE_WAV_SPEECH_PATH = download_asset('chinese-style-guitar-melody_128bpm_G_minor.wav')


def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def load_audio(file_path):
    if os.path.isfile(file_path):
        return torchaudio.load(file_path)
    else:
        return False


def get_speech_sample(*, resample=None):
    return _get_sample('chinese-style-guitar-melody_128bpm_G_minor.wav', resample=resample)

def get_spectrogram(
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    waveform, _ = _get_sample('chinese-style-guitar-melody_128bpm_G_minor.wav')
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def main():
    get_spectrogram()
if __name__ == "__main__":
    main()
