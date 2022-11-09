# Audio
num_mels = 80
n_mels = 80
n_fft = 2048
sr = 44100
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
win_length = n_fft
hop_length = int( win_length * 0.25)
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 256
max_db = 100
ref_db = 20
n_iter = 60
# power = 1.5
outputs_per_step = 1

