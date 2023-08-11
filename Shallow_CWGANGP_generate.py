from shallow_CWGANGP import ShallowGenerator
from data import *
from tools import *
import torch
import matplotlib.pyplot as plt
import matplotlib
import soundfile as sf
import time
matplotlib.use('Qt5Agg')  # Use this backend, otherwise cannot support interactive display

config = load_yaml("config_Sh_CWGAN_GP.yaml")

# Device to run the computations on
if config['cuda']:
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
else:
    device = "cpu"

transform = AudioTransforms(sample_rate=config["sample_rate"],
                            n_fft=int(config["n_fft"]),
                            n_stft=int(config["n_stft"]),
                            win_length=config["win_length"],
                            hop_length=config["hop_length"],
                            f_min=config["f_min"],
                            f_max=config["f_max"],
                            n_mels=config["n_mels"],
                            window_fn=config["window_fn"],
                            power=config["power"],
                            normalized=config["normalized"],
                            momentum=config["momentum"],
                            n_iter=config["n_iter"],
                            device=device).to(device)


# model_info_dict = torch.load('log/checkpoint/Shallow_CGANs_600.pt')
model_info_dict = torch.load('log/models/Shallow_CGAN_WGAN_GP/Shallow_CGANs_1000_fin.pt')

gen_list = []
for idx in range(config['num_strips']):
    gen_list.append(ShallowGenerator(input_shape=(1, 128, 16),
                                     z_dim=config['latent_dim'],
                                     classes=config['num_class']).to(device))

for idx in range(config['num_strips']):
    gen_list[idx].load_state_dict(model_info_dict['gen_state_dict'][idx])

# Class condition
c = torch.tensor([[config['force_level']]]).to(device)

# Sampling from random noise
z = torch.randn(1, config['latent_dim']).to(device)

print("Setting up generators to the selected device.........")

# Warm up phase. Established the generator in the gpu/cpu
for idx in range(config['num_strips']):
    gen_list[idx].eval()
    _ = gen_list[idx](z, c)

print(" ---- Setup completed ---- ")

# current = None
strips = []
index = config['start_index']
for num in range(config['num_gen']):
    strips = []
    current = time.time()
    c = torch.tensor([[config['force_level']]]).to(device)
    z = torch.randn(1, config['latent_dim']).to(device)
    for idx in range(config['num_strips']):
        # current = time.time()
        with torch.no_grad():
            strips.append(gen_list[idx](z, c))
        # print(time.time() - current)

    mel_recon = mel_spectrogram_from_strips(torch.stack(strips, 0))
    mel_recon = torch.squeeze(mel_recon)
    print(time.time() - current)
    # Invert the mel spectrogram into audio waveform
    # y_wav = transform.GriffinLim(transform.inv_mel_spec(mel_recon))

    # Save the audio waveform into a wav file
    # sf.write(config['save_path_for_generated'] +
    #          f"/{config['force_level']}/{index}.wav",
    #          y_wav.cpu().numpy(), config["sample_rate"])
    mel_recon = mel_recon.cpu().numpy()

    plt.figure(figsize=(128 / 96, 128 / 96), dpi=96)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(mel_recon, cmap="jet", vmin=0.0, vmax=5.0)
    plt.savefig(config['save_path_for_generated'] +
                f"/{config['force_level']}/{index}.png", dpi=96)
    plt.close()
    index = index + 1

