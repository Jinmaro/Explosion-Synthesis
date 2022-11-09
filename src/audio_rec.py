"""
    録音処理を行うプログラム
"""

import pyaudio
import threading
import queue
import os
import wave

class Input(threading.Thread):      # キー入力待ちをしない（ノンブロッキング処理）
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.start()

    def run(self):
        while True:
            t = input()
            self.queue.put(t)

    def input(self, block = True, timeout = None):
        try:
            return self.queue.get(block, timeout=timeout)
        except queue.Empty:
            return None

def rec(write_path):
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz　サンプリング周波数
    chunk = 4096 # 2^12 一度に取得するデータ数
    record_secs = 12 # 録音する秒数
    dev_index = 1 # デバイス番号
    audio = pyaudio.PyAudio() # create pyaudio instantiation
    cin = Input()
    flag = False    # キー入力のフラグ

    stream = audio.open(format = form_1,
                        rate = samp_rate,
                        channels = chans,
                        input_device_index = dev_index,
                        input = True,
                        frames_per_buffer=chunk)
    #*--------- 録音 -----------*#
    print("Start Rec, If you want to stop, PLZ smash ENTER KEY")
    frames = [] # 波形を格納する配列
    for i in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)
        t = cin.input(block=False)
        if t == "":
            flag = True
            break

    print("Finished")
    #*--------------------------*#

    #*---- Terminate Pyaudio----*#
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #*--------------------------*#

    #*--- save as .wav file ---*#
    wavefile = wave.open(write_path,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()
    #*--------------------------*#

    return flag