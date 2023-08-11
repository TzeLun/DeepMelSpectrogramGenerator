from data import *
from CWGANGP import Generator
import torch
import matplotlib.pyplot as plt
import matplotlib
import soundfile as sf
import time
import argparse
matplotlib.use('Qt5Agg')  # Use this backend to support interactive display


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config_CWGAN_GP.yaml')
parser.add_argument('--model', default='log/models/CWGANGP_D/CWGANGP1000.pt')
parser.add_argument('--num_gen', default=1)
parser.add_argument('--use_cuda', default=False, action='store_true')
parser.add_argument('--drill_force', default=1)  # 0, 1, 2, 3
parser.add_argument('--drill_angle', default=0)  # 0, 1, 2
a = parser.parse_args()
config = load_yaml(a.config)


def generate():

    force = int(a.drill_force)
    angle = int(a.drill_angle)

    # Device to run the computations on
    if a.use_cuda:
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    else:
        device = "cpu"

    # For Griffin Lim algorithm
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

    model_info_dict = torch.load(a.model)

    generator = Generator(input_shape=(1, 128, 128),
                          z_dim=config['latent_dim'],
                          classes=config['num_class']).to(device)

    generator.load_state_dict(model_info_dict['gen_state_dict'])

    # Class condition
    c = torch.tensor([[angle], [force]]).to(device)

    # Sampling from random noise
    z = torch.randn(1, config['latent_dim']).to(device)

    print("Setting up generators to the selected device.........")

    # Warm up phase. Established the generator in the gpu/cpu
    generator.eval()
    with torch.inference_mode():
        _ = generator(z, c)

    print(" ---- Setup completed ---- ")
    total_time = 0
    index = 0
    if not a.use_cuda:
        for num in range(a.num_gen):
            current = time.time()
            z = torch.randn(1, config['latent_dim']).to(device)
            c = torch.tensor([[angle], [force]]).to(device)
            with torch.inference_mode():
                mel_recon = generator(z, c)
            total_time = total_time + (time.time() - current)

            mel_recon = torch.squeeze(mel_recon)

            # Invert the mel spectrogram into audio waveform
            y_wav = transform.GriffinLim(transform.inv_mel_spec(mel_recon))

            # Save the audio waveform into a wav file
            sf.write(f"test/{index}.wav",
                     y_wav.cpu().numpy(), config["sample_rate"])

            mel_recon = mel_recon.cpu().numpy()

            plt.figure(figsize=(128 / 96, 128 / 96), dpi=96)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.imshow(mel_recon, cmap="jet", vmin=0.0, vmax=5.0)
            plt.savefig(f"test/{index}.png", dpi=96)

            plt.close()
            index = index + 1
        print(total_time / float(a.num_gen))
    else:
        for num in range(a.num_gen):
            st = torch.cuda.Event(enable_timing=True)  # start
            ed = torch.cuda.Event(enable_timing=True)  # end

            st.record()

            z = torch.randn(1, config['latent_dim']).to(device)
            c = torch.tensor([[angle], [force]]).to(device)
            with torch.inference_mode():
                mel_recon = generator(z, c)

            ed.record()

            torch.cuda.synchronize()

            total_time = total_time + (st.elapsed_time(ed) / 1000.0)

            mel_recon = torch.squeeze(mel_recon)

            # Invert the mel spectrogram into audio waveform
            y_wav = transform.GriffinLim(transform.inv_mel_spec(mel_recon))

            # Save the audio waveform into a wav file
            sf.write(f"test/{index}.wav",
                     y_wav.cpu().numpy(), config["sample_rate"])

            mel_recon = mel_recon.cpu().numpy()

            plt.figure(figsize=(128 / 96, 128 / 96), dpi=96)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.imshow(mel_recon, cmap="jet", vmin=0.0, vmax=5.0)
            plt.savefig(f"test/{index}.png", dpi=96)

            plt.close()
            index = index + 1
        print(total_time / float(a.num_gen))


if __name__ == '__main__':
    generate()
