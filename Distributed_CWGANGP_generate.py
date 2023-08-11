from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import torch
from shallow_CWGANGP import ShallowGenerator
import argparse
from data import load_yaml, AudioTransforms
import time
from tools import mel_spectrogram_from_strips, numpy_mel_spectrogram_from_strips
import matplotlib.pyplot as plt
import soundfile as sf


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config_Sh_CWGAN_GP.yaml')
parser.add_argument('--model', default='log/models/Shallow_CWGANGP/ShallowCWGANGP1000.pt')
parser.add_argument('--use_cuda', default=False, action='store_true')
parser.add_argument('--force_level', default=1)  # 0, 1, 2, 3
a = parser.parse_args()
config = load_yaml(a.config)


class CGAN:
    def __init__(self, conf, checkpoint, dev):
        self.store = [None] * conf['num_strips']

        model_info_dict = torch.load(checkpoint)

        self.gen_list = []
        for ind in range(config['num_strips']):
            self.gen_list.append(ShallowGenerator(input_shape=(1, 128, 16),
                                                  z_dim=conf['latent_dim'],
                                                  classes=conf['num_class']).to(dev))
            self.gen_list[ind].load_state_dict(model_info_dict['gen_state_dict'][ind])

        # Warm up phase. Established the generator in the gpu/cpu
        c_ = torch.tensor([[0]]).to(dev)
        z_ = torch.randn(1, conf['latent_dim']).to(dev)

        for ind in range(config['num_strips']):
            self.gen_list[ind].eval()
            with torch.inference_mode():
                _ = self.gen_list[ind](z_, c_)

    def generate(self, i):
        with torch.inference_mode():
            self.store[i] = torch.squeeze(self.gen_list[i](z, c))


def generate(i):
    with torch.inference_mode():
        return torch.squeeze(net.gen_list[i](z, c))


# Distributed shallow CWGANGP only supports drill force S (0, 1, 2, 3) currently.
if __name__ == '__main__':
    torch.set_num_threads(1)

    store = [None] * config['num_strips']

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

    net = CGAN(config, a.model, device)
    print("Generator setup completed.....")

    # Prompt user
    # fc = int(input("Enter a class of force (0, 1, 2, 3): "))
    fc = int(a.force_level)
    # if fc in [0, 1, 2, 3]:
    if not a.use_cuda:
        current = time.time()

        c = torch.tensor([[fc]]).to(device)
        z = torch.randn(1, config['latent_dim']).to(device)

        with ThreadPoolExecutor() as executor:
            mel_strips = [strip for strip in executor.map(generate, range(len(net.gen_list)))]

        mel_recon = mel_spectrogram_from_strips(torch.stack(mel_strips, 0))

        print(time.time() - current)

        mel_recon = torch.squeeze(mel_recon)

    else:
        st = torch.cuda.Event(enable_timing=True)  # start
        ed = torch.cuda.Event(enable_timing=True)  # end

        st.record()
        c = torch.tensor([[fc]]).to(device)
        z = torch.randn(1, config['latent_dim']).to(device)
        # threads = []
        #
        # for index in range(config['num_strips']):
        #     threads.append(Thread(target=generate, args=(index, store)))
        #
        # for thread in threads:
        #     thread.start()
        #
        # for thread in threads:
        #     thread.join()
        with ThreadPoolExecutor() as executor:
            mel_strips = [strip for strip in executor.map(generate, range(len(net.gen_list)))]

        mel_recon = mel_spectrogram_from_strips(torch.stack(mel_strips, 0))

        ed.record()

        torch.cuda.synchronize()

        print(st.elapsed_time(ed) / 1000.0)

    # Invert the mel spectrogram into audio waveform
    y_wav = transform.GriffinLim(transform.inv_mel_spec(mel_recon))
    # Save the audio waveform into a wav file
    sf.write(f"test/force_{fc}.wav", y_wav.cpu().numpy(), config["sample_rate"])

    plt.figure(figsize=(128 / 96, 128 / 96), dpi=96)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(mel_recon.cpu().numpy(), cmap="jet", vmin=0.0, vmax=5.0)
    plt.savefig(f"test/force_{fc}.png", dpi=96)
    plt.close()

