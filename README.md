# DeepMelSpectrogramGenerator
A fully-convolution GAN-based model to generate mel-spectrogram of audio samples.

The mel-spectrogram is in a (1, mel filter, time frame) format and can be converted back to audio waveform using mel-spectrogram inverter like HiFi-GAN, Wavenet etc.
The model architecture is inspired by DCGAN [1] and is conditioned by discrete variables. It is trained as a Wasserstein GAN with gradient penalty [2]. Some hints and tricks were adopted from a NIPS 2016 workshop [3]. As a nomenclature, this model is called CWGAN-GP.

## Dependencies
Refer to [requirements.txt](https://github.com/TzeLun/DeepMelSpectrogramGenerator/blob/main/requirements.txt)

Frechet Audio Distance is also used in `Evaluate.py`, and the link to the github repository used for this metric is available in the same script.

Just in case, `python>=3.9` is desired to support Python's multiprocessing libraries.

## Mel-spectrogram & Model Configuration
The training parameters and mel-spectrogram configurations are within a `.yaml` file. The yaml file can be customized by editing the parameters in it and load it using argparser.

## Training
To train the model:
```
python train_CWGANGP.py --config <.yaml configuration file>
```
To train the model from a checkpoint:
```
python train_CWGANGP.py --config <.yaml configuration file> --checkpoint <checkpoint file> --load_from_checkpoint
```
## Inference
To generate mel-spectrogram samples with drill force 1 and drill angle 0 with CUDA:
```
python CWGANGP_generate.py --config <.yaml configuration file> --model <trained model file> --num_gen <number of generation> --drill_force 1 --drill_angle 0 --use_cuda
```
For this repository, the Griffin-Lim algorithm is used to invert the mel-spectrogram back to the audio waveform. The `drill_force` and `drill_angle` can be replaced with other conditions to suit one's application.

## Evaluate
To score the mel-spectrogram generator w.r.t. real samples using Frechet Inception Distance (FID) and Frechet Audio Distance (FAD):
```
python Evaluate.py --FID_real <dir of real mel-spec images> --FID_fake <dir of fake mel-spec images> --FAD_real <dir of real audio samples> --FAD_fake <dir of fake audio samples>
```
## Notes on shallow CWGAN-GP
This is an experimental model consisting of 8 lightweight CWGAN-GP generating 8 separate strips (1, mel filter, time frame / 8) constituting the whole mel-spectrogram. When run sequentially, each network generates mel-spectrogram about 10 times faster than a single deep CWGAN-GP. The goal is to run this network in parallel during inferencing under a single CPU/GPU. However, when using Python's multiprocessing module, particularly ThreadPoolExecutor and the map function, it performed slower than the original CWGAN-GP, due to the use of a single core. Using multiple processes did not fare much better because each network needs to be copied into each process.
## References
[1] Radford _et al._, Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, [paper](https://arxiv.org/abs/1511.06434) </br>
[2] Gulrajani _et al._, Improved Training of Wasserstein GANs, [paper](https://arxiv.org/abs/1704.00028)  </br>
[3] Soumith _et al._, How to train a GAN?, [link](https://github.com/soumith/ganhacks)
