import os
from audio_rec import rec
from audio import load_wav, preprocess
from synthesis import synthesis
from plot import result_plot
import soundfile as sf
import numpy as np

def main():
    ## Path ##
    result_path = "../result"                                       # dir of result which contains recording voices, synthesized sounds, img of waves and specs
    num_file = str(sum(os.path.isfile                               # file number
                       (os.path.join(result_path, name))
                        for name in os.listdir(result_path)) // 3)
    rec_path = os.path.join(result_path, num_file + "_rec.wav")     # file path of a recording voice
    out_path = os.path.join(result_path, num_file + "_out.wav")     # file path of a synthesized sound
    image_path = os.path.join(result_path, num_file + "_img.svg")   # file path of an img

    ## Rec ##
    flag = rec(rec_path)            # rec and get flag of a key event

    ## Load Rec Data ##
    input_data = load_wav(rec_path) # load a recorded sound

    ## Preprocess for Input Data ##
    input_data = preprocess(input_data, flag)   # preprocess of recorded sound

    ## Synthesis ##
    output_data = synthesis(input_data)         # get the output ( input an utterance and synthesize a sound of explosion)

    ## Adjusting Length ##
    diff = len(input_data) - len(output_data)               # get diff of length of an utterance and EXP
    output_data = np.append(output_data, np.zeros(diff))    # zero padding at the end of a sound of explosion

    ## Result Plot and Save image ##
    result_plot(image_path, input_data, output_data)    # get img of waves and specs

    ## Save wavs ##
    sf.write(rec_path, input_data,  44100, format="WAV", subtype="PCM_16")  # save an utterance
    sf.write(out_path, output_data, 44100, format="WAV", subtype="PCM_16")  # save exp

if __name__ == "__main__":
    main()