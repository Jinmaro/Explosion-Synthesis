"""
    マイクから入力された音声をその場で合成する
    合成音の再生および、入力音声合成音の可視化も行う
"""
import os
from audio_rec import rec
from audio import load_wav, preprocess
from synthesis import synthesis
from plot import result_plot
import soundfile as sf
import numpy as np

def main():
    ## Path ##
    result_path = "../result"                                        # 録音音声、合成音、波形画像等を格納するディレクトリ
    num_file = str(sum(os.path.isfile                               # ファイル名
                       (os.path.join(result_path, name))
                        for name in os.listdir(result_path)) // 3)
    rec_path = os.path.join(result_path, num_file + "_rec.wav")     # 録音音声のパス
    out_path = os.path.join(result_path, num_file + "_out.wav")     # 合成音のパス
    image_path = os.path.join(result_path, num_file + "_img.svg")   # 画像のパス

    ## Rec ##
    flag = rec(rec_path)            # 録音 （フラグはキー入力の有無）

    ## Load Rec Data ##
    input_data = load_wav(rec_path) # 録音音声のロード

    ## Preprocess for Input Data ##
    input_data = preprocess(input_data, flag)   # 録音音声の前処理

    ## Synthesis ##
    output_data = synthesis(input_data)         # データを入力して、爆発音を合成

    ## Adjusting Length ##
    diff = len(input_data) - len(output_data)               # 音声と爆発音の長さの差分を取る
    output_data = np.append(output_data, np.zeros(diff))    # 爆発音の後ろに差分の長さのゼロパディングを行う

    ## Result Plot and Save image ##
    result_plot(image_path, input_data, output_data)    # 入力と出力の波形等の描画

    ## Save wavs ##
    sf.write(rec_path, input_data,  44100, format="WAV", subtype="PCM_16")  # 録音音声の保存
    sf.write(out_path, output_data, 44100, format="WAV", subtype="PCM_16")  # 爆発音の保存

if __name__ == "__main__":
    main()