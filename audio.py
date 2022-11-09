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
    fade_rate = 0.15                            # 波形の何割の長さでフェードアウトさせるか
    len_fader = int(len(y) * fade_rate)         # フェードアウトの長さ
    fader = np.linspace(1.0, 0.0, len_fader)    # 1~0の階差数列
    index = len(y) - len(fader)                 # フェードアウトし始めるインデックス
    faded = y[index:] * fader                   # 得た位置から、階差数列をかけてフェードさせる
    y = np.append(y[:index], faded)             # フェードさせる前の箇所と、フェードさせた後を繋げる
    return y

## Function to calc index with RMS
def get_start_index(sq, threshold, shift, length):
    for i in range(0, len(sq), shift):  # shift分ずらしていき、lengthの範囲の値を足していく
        total = 0
        if i > len(sq) - length:        # i+length が配列の範囲を超えた場合
            return i
        total = np.sum(sq[i:i+length])  # i ~ i+length の範囲で合計値を算出
        if total > threshold:           # 合計値が閾値を超えた場合
            return i                    # そのインデックスを返す

## RMS
def rms(y):
    shift = 256     # 移動幅
    length = 1024   # 合計値を出す範囲
    sq = y ** 2     # 波形の二乗
    index = get_start_index(sq, 0.01, shift, length)    # 開始地点の取得
    y = np.append(np.zeros(length), y[index:])          # 開始前に無音区間をつける
    return y

## Preprocess for wav
def preprocess(y, flag):
    if flag:                    # ボタンを押して終了している場合
        y = y[:len(y)-4410]     # キー入力の音をカット
    y = fade(y)                 # 後ろをフェードアウトさせる
    y = rms(y)                  # RMS を使用して、開始地点を揃える
    
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