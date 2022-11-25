import torch as t
from audio import get_spectrograms
from utils import spectrogram2wav
import hyperparams as hp
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import os
import librosa

## Function to transform wav into mel
def wav2feature(y):
    mel = get_spectrograms(y)
    pos = np.arange(1, mel.shape[0] + 1)
    
    return mel, pos

## Fuction to load Model checkpoint
def load_checkpoint(model_name="transformer"):
    if model_name == "transformer":         # load Transforemr
        ## You can change Learned Model 
        # state_dict = t.load('../chkpt/transformer_2000.pth.tar',map_location=t.device('cpu'))  
        # state_dict = t.load('../chkpt/transformer_2502_4000_256.pth.tar',map_location=t.device('cpu'))
        # state_dict = t.load('../chkpt/transformer_3075_2000_256.pth.tar',map_location=t.device('cpu'))
        # state_dict = t.load('../chkpt/transformer_3075_4000_256.pth.tar',map_location=t.device('cpu')) 
        state_dict = t.load('../chkpt/transformer_3575_4000_256.pth.tar',map_location=t.device('cpu')) 
    else:                                   # load Postnet
        state_dict = t.load('../chkpt/postnet_2000.pth.tar',map_location=t.device('cpu'))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k
        new_state_dict[key] = value

    return new_state_dict

## Function to synthesis sound of Explosion
def synthesis(src):
    mel_src, pos_src = wav2feature(src) # wav2mel

    ## Set Models
    m = Model().cpu()               # Transformer
    m_post = ModelPostNet().cpu()   # Neural Vocoder

    ## Model Loading
    m.load_state_dict(load_checkpoint("transformer"),strict=False)
    m_post.load_state_dict(load_checkpoint("postnet"),strict=False)

    ## Model Setting
    m.train(False)
    m_post.train(False)

    ## Prepare Mels
    mel_src = t.FloatTensor(mel_src).unsqueeze(0).cpu()
    pos_src = t.LongTensor(pos_src).cpu()
    mel_trg_input = t.zeros([1,1,80]).cpu()

    ## Model Processing
    pbar = tqdm(range(mel_src.shape[1]))
    with t.no_grad():
        for i in pbar:
            pos_trg = t.arange(1, mel_trg_input.size(1)+1).unsqueeze(0).cpu()
            mel_pred, postnet_pred, attn, _, attn_dec = m.forward(mel_src, mel_trg_input, pos_src, pos_trg)
            mel_trg_input = t.cat([mel_trg_input, mel_pred[:,-1:,:]], dim=1)

        mag_pred = m_post.forward(postnet_pred)

    ## Convert Mag Spectrogram into waveform
    wav = spectrogram2wav(mag_pred.squeeze(0).numpy())

    return wav