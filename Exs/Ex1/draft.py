import librosa
import torch


def naive_tempo_shift(wav, factor):
    # Convert the waveform to a PyTorch tensor
    wav_tensor = torch.from_numpy(wav)

    # Compute the magnitude spectrogram of the audio
    spec = torch.stft(wav_tensor, n_fft=2048, hop_length=512, return_complex=False)
    mag_spec = torch.abs(spec)

    # Scale the spectrogram along the time-frequency axes
    mag_spec_stretched = torch.nn.functional.interpolate(mag_spec.unsqueeze(0), scale_factor=factor, mode='bilinear')

    # Convert the stretched magnitude spectrogram back to a complex-valued spectrogram
    stretched_spec = torch.polar(mag_spec_stretched.squeeze(0), torch.angle(spec))

    # Compute the stretched waveform by inverting the spectrogram
    stretched_wav = torch.istft(stretched_spec, n_fft=2048, hop_length=512, length=len(wav_tensor), return_complex=False)

    # Convert the stretched waveform back to a NumPy array
    stretched_wav = stretched_wav.numpy()

    return stretched_wav


wav, sr = librosa.load('audio_16k/Basta_16k.wav', sr=16000)
wav_08 = naive_tempo_shift(wav, 0.8)
wav_12 = naive_tempo_shift(wav, 1.2)
librosa.output.write_wav('outputs/naive_pitch_shift_0_8.wav', wav_08, sr)
librosa.output.write_wav('outputs/naive_pitch_shift_1_2.wav', wav_12, sr)

