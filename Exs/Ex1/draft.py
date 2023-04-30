import librosa
import numpy as np
import torch
import torchaudio


# In this subsection you will implement version of a slightly better algorithm to perform
# time_stretch called Phase vocoder.
def time_stretch(signal, factor, win_size=1024, hop=1024 // 4):
    # create window
    hann_window = construct_hann_window(win_size)

    # draw two complex STFTs
    new_hop = int(hop * factor)
    stft_left = get_complex_stft(signal[:, :-hop], win_size, new_hop, hann_window)
    stft_right = get_complex_stft(signal[:, :-hop], win_size, new_hop, hann_window)

    # calculate accumulated phase delta and reconstruct phase from it
    phase = get_acc_phase_delta(stft_left, stft_right)

    # reconstruct component from phase
    re, im = get_re_im_from_phase(phase)
    # complex_new_stft = view_as_complex(stack([re, im], dim=-1)) * abs(stft_right))
    # complex_new_stft = torch.view_as_complex(torch.stack([re, im], dim=-1)) * torch.abs(stft_right)
    # output = torch.istft(complex_new_stft, win_length=win_size, hop_length=hop, window=hann_window)
    # output = torch.istft(complex_new_stft, win_length=win_size, hop_length=new_hop, window=hann_window, n_fft=win_size)

    first_channel_complex_new_stft = torch.view_as_complex(
        (torch.stack([re[0], im[0]]) * abs(stft_right)).permute(1, 2, 0).contiguous())
    second_channel_complex_new_stft = torch.view_as_complex(
        (torch.stack([re[1], im[1]]) * abs(stft_right)).permute(1, 2, 0).contiguous())
    first_channel_output = torch.istft(first_channel_complex_new_stft, win_length=win_size, hop_length=hop,
                                       window=hann_window, n_fft=win_size)
    second_channel_output = torch.istft(second_channel_complex_new_stft, win_length=win_size, hop_length=hop,
                                        window=hann_window, n_fft=win_size)

    return torch.stack([first_channel_output, second_channel_output])


def construct_hann_window(win_size):
    # return a vector representing a hanning window, hint: see torch.hann_window
    hann_window = torch.hann_window(window_length=win_size)
    return hann_window


def get_complex_stft(signal, win_size, hop, window):
    # Convert the waveform to a PyTorch tensor
    signal = torch.from_numpy(signal)
    # return a complex representation of the stft (x + jy form)
    stft = torch.stft(signal, n_fft=win_size, win_length=win_size, hop_length=hop, window=window, return_complex=True)
    return stft


def get_acc_phase_delta_1(stft_left, stft_right):
    # calculate angular distance between two complex STFTs

    # phase_delta = angle(stft_right) - angle(stft_left)
    phase_delta = torch.angle(stft_right) - torch.angle(stft_left)

    # accumulate phase, follow this recursive formula
    # for i in {1...length(phase_delta)}: phase[i] := phase_delta[i] + phase[i - 1];
    # phase[0] = phase_delta[0]
    phase = torch.zeros_like(phase_delta)
    # since wav is stereo we need to do this for both channels
    phase[0][0] = phase_delta[0][0]
    phase[1][0] = phase_delta[1][0]

    for i in range(1, len(phase_delta[0])):
        phase[0][i] = phase_delta[0][i] + phase[0][i - 1]
        phase[1][i] = phase_delta[1][i] + phase[1][i - 1]

    # round phase back to [-2 * pi, 2 * pi] range
    # phase = phase - (2 * pi * round(phase_delta / (2 * pi)))

    phase = phase - (2 * np.pi * torch.round(phase_delta / (2 * np.pi)))

    return phase


def get_acc_phase_delta(stft_left, stft_right):
    # calculate angular distance between two complex STFTs
    phase_delta = torch.angle(stft_right) - torch.angle(stft_left)

    # accumulate phase, follow the recursive formula
    phase = torch.zeros([2, 513, 1188])
    phase[:, :, 0] = phase_delta[:, :, 0]
    phase = torch.cumsum(phase_delta, axis=-1)

    # round phase back to 0 - 2 * pi range
    phase = phase - 2 * np.pi * torch.round(phase / (2 * np.pi))

    return phase


def get_re_im_from_phase(phase):
    # retrieves the real and imaginary components from a complex phase
    re = torch.cos(phase)
    im = torch.sin(phase)
    return re, im


wav, sr = librosa.load('audio_16k/Basta_16k.wav', sr=16000, mono=False)
wav_08 = time_stretch(wav, 0.8)
wav_12 = time_stretch(wav, 1.2)
torchaudio.save("outputs/phase_vocoder_0_8.wav", wav_08, sr)
torchaudio.save('outputs/phase_vocoder_1_2.wav', wav_12, sr)
