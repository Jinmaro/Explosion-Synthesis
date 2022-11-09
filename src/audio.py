import warnings
warnings.simplefilter('ignore', FutureWarning)
import librosa
import numpy as np
import hyperparams as hp

## Params ##
sample_rate = 44100

## Load wav file
def load_wav(load_path):
    y,_ = librosa.load(load_path, sr = sample_rate)
    return y

## Function to fade out
def fade(y):
    fade_rate = 0.15                            # 15 Percent of wave length
    len_fader = int(len(y) * fade_rate)         # fade out length
    fader = np.linspace(1.0, 0.0, len_fader)    # difference sequence of 1 -> 0 for fade out 
    index = len(y) - len(fader)                 # index where fade out starts
    faded = y[index:] * fader                   # fade out 
    y = np.append(y[:index], faded)             # concatenate wave before fade out and wave which was faded out
    return y

## Function to calc index with RMS
def get_start_index(sq, threshold, shift, length):
    for i in range(0, len(sq), shift):  # 0 ~ len(sq)   += shift
        total = 0                       # init total
        if i > len(sq) - length:        # if i+length is out of range
            return i
        total = np.sum(sq[i:i+length])  # calc sum of sq from i to i+length
        if total > threshold:           # if sum is bigger than threshold
            return i                    # return index

## RMS
def rms(y):
    shift = 256     # shift length
    length = 1024   # window length
    sq = y ** 2     # square of wave
    index = get_start_index(sq, 0.01, shift, length)    # get a start index
    y = np.append(np.zeros(length), y[index:])          # put zero padding on a head of wave
    return y

## Preprocess for wav
def preprocess(y, flag):
    if flag:                    # flag for the case of stopping with smashing ENTER key
        y = y[:len(y)-4410]     # cut the sound of key input
    y = fade(y)                 # fade out
    y = rms(y)                  # adjust the head(start) index with RMS
    
    return y

## Function to get spectrograms
def get_spectrograms(y):
    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)
    
    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

     # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)

    return mel