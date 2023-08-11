from Frechet_Inception_Distance import Compute_FID
import torch
import argparse

# Direct use of FAD from https://github.com/gudgud96/frechet-audio-distance
# NOTE: require https://github.com/qiuqiangkong/torchlibrosa to work with PANN model
from frechet_audio_distance import FrechetAudioDistance

# Set the path for the real and fake samples for FID and FAD scoring
parser = argparse.ArgumentParser()
parser.add_argument('--FID_real', default=f"real/")  # Mel-spectrogram image samples
parser.add_argument('--FID_fake', default=f"fake/")
parser.add_argument('--FAD_real', default=f"real/")  # Keep the audio samples in a separate directory
parser.add_argument('--FAD_fake', default=f"fake/")
a = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def evaluate():

    # --------------- Frechet Inception Distance ---------------------- #

    # Real
    path_real = a.FID_real
    # Fake
    path_fake = a.FID_fake

    paths = [path_real, path_fake]

    fid = Compute_FID(paths=paths, dev=device)
    print(f"FID: {fid}")

    # -------------- Frechet Audio Distance ------------------------- #
    # Real
    path_real = a.FAD_real
    # Fake
    path_fake = a.FAD_fake

    # There is a problem when using "pann" model... Stick to "vggish" for now
    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False,
        use_activation=False,
        verbose=False
    )

    fad = frechet.score(path_real, path_fake)
    print(f"FAD: {fad}")


if __name__ == '__main__':
    evaluate()
