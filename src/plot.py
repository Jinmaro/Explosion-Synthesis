import warnings
warnings.simplefilter('ignore', FutureWarning)
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import hyperparams as hp

## Function to plot waveforms and mels
def result_plot(path, src, trg):
    ## Wav to Mel
    mel_src = librosa.feature.melspectrogram(y=src,
                                             sr=hp.sr,
                                             n_mels=hp.num_mels,
                                             n_fft=hp.n_fft, 
                                             hop_length=hp.hop_length, 
                                             win_length=hp.win_length, 
                                             window='hann')
    mel_src = librosa.power_to_db(mel_src, ref=np.max)

    mel_trg = librosa.feature.melspectrogram(y=trg,
                                             sr=hp.sr,
                                             n_mels=hp.num_mels,
                                             n_fft=hp.n_fft, 
                                             hop_length=hp.hop_length, 
                                             win_length=hp.win_length, 
                                             window='hann')
    mel_trg = librosa.power_to_db(mel_trg, ref=np.max)

    ## init fig
    fig = plt.figure(figsize=(10,8))    # 1000pic x 800pic

    #### plot wavs and mels
    plt.rcParams["font.size"] = 18
    ## Src wave
    ax = fig.add_subplot(2,7,(5,7),xlabel="time [sec]")
    ax.set(ylim=[-1.0,1.0])
    librosa.display.waveshow(src, sr=hp.sr)
    plt.xticks( np.arange(0, len(src)/sr, 1))
    ax.set_title('wave of Input', fontsize=15)
    ## Src mel
    ax = fig.add_subplot(2,7,(1,4),xlabel="time [sec]")
    librosa.display.specshow(mel_src, sr=hp.sr, x_axis='time', y_axis='mel')
    plt.xticks( np.arange(0, len(src)/sr, 1))
    ax.set_title('Melspectrogram of Input', fontsize=15)
    ## Trg wave
    ax = fig.add_subplot(2,7,(12,14),xlabel="time [sec]")
    ax.set(ylim=[-1.0,1.0])
    librosa.display.waveshow(trg, sr=hp.sr)
    plt.xticks( np.arange(0, len(trg)/sr, 1))
    ax.set_title('wave of Output', fontsize=15)
    ## Trg mel
    ax = fig.add_subplot(2,7,(8,11),xlabel="time [sec]")
    librosa.display.specshow(mel_trg, sr=hp.sr, x_axis='time', y_axis='mel')
    plt.xticks( np.arange(0, len(trg)/sr, 1))
    ax.set_title('Melspectrogram of Output', fontsize=15)

    plt.rcParams["svg.fonttype"] = "none"
    plt.tight_layout()

    ## Save fig
    fig.savefig(path)