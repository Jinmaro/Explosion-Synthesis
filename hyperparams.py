# Audio
num_mels = 80
# num_freq = 1024
n_fft = 2048
# n_fft=4096
# sr = 22050
sr = 44100
# frame_length_ms = 50.
# frame_shift_ms = 12.5
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
win_length = n_fft
hop_length = int( win_length * 0.25)
# hop_length = int(sr*frame_shift) # samples.
# win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
# hidden_size = 256
hidden_size = 256
embedding_size = 512
max_db = 100
ref_db = 20

n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 4000
lr = 0.001
save_step = 67500
image_step = 500
batch_size = 16


data_path = './data/EB-data'
checkpoint_postnet_path = './chkpt/checkpoint_postnet/2502'
# checkpoint_transformer_path = './chkpt/checkpoint_1468_rms/3'
# checkpoint_transformer_path = './chkpt/checkpoint_1698_rms_all_fade/3'
# checkpoint_transformer_path = './chkpt/checkpoint_transformer/1698_rms_all_fade/3'
checkpoint_transformer_path = './chkpt/checkpoint_transformer/2502_rms_fade/2'
sample_path = './synthesis'
src_path = './synthesis/src.wav'
