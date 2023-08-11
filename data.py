import json
import soundfile as sf
import random
import torchaudio
import csv
import os
from collections.abc import Mapping
from tools import *
import yaml


window_options = {
    'hann_window': torch.hann_window
}


# load configuration files of yaml format into Python dictionary
def load_yaml(filepath):

    with open(filepath, 'r') as stream:

        try:
            config = yaml.safe_load(stream)

            if 'window_fn' in config:
                config['window_fn'] = window_options[config['window_fn']]

        except yaml.YAMLError as e:
            print(e)

    return config


# Save a yaml file
def save_yaml(config, filepath):
    file = open(filepath, "w")
    yaml.dump(config, file)
    file.close()


# Load audio using torchaudio
def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


# Segment the audio into smaller pieces
def segment_audio(y, sr, start=0.0, duration=None):
    start_index = int(start * sr)  # point to begin sampling the sub-array
    if duration is None:
        y = y[0, start_index:y.shape[1]]  # segment the audio till the end
        return torch.reshape(y, (1, y.shape[0]))
    else:
        if (start + duration) > (y.shape[1] / float(sr)):
            y = y[0, start_index:y.shape[1]]  # segment the audio till the end
            return torch.reshape(y, (1, y.shape[0]))
        else:
            y = y[0, start_index:int((start + duration) * sr)]
            return torch.reshape(y, (1, y.shape[0]))


# Convert the torch tensor data into a metadata format acceptable by JSON
def json_to_dict(data=None, sr=None, info=None):
    dt = {'data': None,
          'sampling rate': None,
          'dtype': None,
          'info': None}
    if data is not None:
        np_tensor = data.numpy()
        dtype = np_tensor.dtype
        dt['data'] = np_tensor.tolist()
        dt['dtype'] = dtype.name
    if sr is not None:
        dt['sampling rate'] = sr
    if info is not None:
        dt['info'] = info
    return dt


# Save the audio waveform array or the mel spectrogram metadata into a json file
# Use together with to_metadata() function
# The file format is STRICTLY in ".json"
# Not necessary to include .json at the suffix of the filename
def save_json(filename, metadata):
    if '.json' not in filename:
        filename = filename + '.json'
    with open(filename, 'w') as f:
        json.dump(metadata, f)


# Load the audio waveform array or the mel spectrogram from the json file to a torch tensor format
# The file format is STRICTLY in ".json"
# Not necessary to include .json at the suffix of the filename
def load_json(filename):
    if '.json' not in filename:
        filename = filename + '.json'
    with open(filename, 'r') as f:
        dt = json.load(f)
        data = np.array(dt['data'], dtype=dt['dtype'])
        data = torch.from_numpy(data)
        return data, dt['sampling rate'], dt['info']


class AudioTransforms(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=400, n_stft=201, win_length=400, hop_length=200,
                 f_min=0.0, f_max=None, n_mels=128, window_fn=torch.hann_window,
                 power=2, normalized=False, momentum=0.99, n_iter=32, device=None):

        super(AudioTransforms, self).__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.iSTFT = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft,
                                                              win_length=win_length,
                                                              hop_length=hop_length,
                                                              window_fn=window_fn,
                                                              normalized=normalized).to(self.device)

        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                      win_length=win_length,
                                                      hop_length=hop_length,
                                                      window_fn=window_fn,
                                                      power=power,
                                                      normalized=normalized).to(self.device)

        self.mel_scale = torchaudio.transforms.MelScale(sample_rate=sample_rate,
                                                        n_stft=n_stft,
                                                        f_min=f_min,
                                                        f_max=f_max,
                                                        n_mels=n_mels).to(self.device)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                             n_fft=n_fft,
                                                             win_length=win_length,
                                                             hop_length=hop_length,
                                                             f_min=f_min,
                                                             f_max=f_max,
                                                             n_mels=n_mels,
                                                             window_fn=window_fn,
                                                             power=power,
                                                             normalized=normalized).to(self.device)

        self.inv_mel_spec = torchaudio.transforms.InverseMelScale(n_stft=n_stft,
                                                                  n_mels=n_mels,
                                                                  sample_rate=sample_rate,
                                                                  f_min=f_min,
                                                                  f_max=f_max).to(self.device)

        self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft,
                                                            n_iter=n_iter,
                                                            win_length=win_length,
                                                            hop_length=hop_length,
                                                            window_fn=window_fn,
                                                            power=power,
                                                            momentum=momentum).to(self.device)

    def mel_spectrogram(self, y):
        return self.mel_spec(y)

    def inv_mel_spectrogram(self, y):
        return self.inv_mel_spec(y)

    def GriffinLim(self, y):
        return self.griffin_lim(y)

    def inv_stft(self, y):
        return self.iSTFT(y)

    def spectrogram(self, y):
        return self.spec(y)

    def to_mel_scale(self, y):
        return self.mel_scale(y)


# Converts complex STFT to a pytorch tensor image format: (1, C, H, W)
# Channels, C will always be 2 for both real and imaginary STFT "images"
# The input y is of format: (1, H, W) with H and W are complex
# Returns tensor of rgb format: (1, 2, H, W) [Two channels]
def complex_stft_to_rgb_format(y):
    y_ = torch.cat([y.real, y.imag])
    y_ = torch.reshape(y_, (1, y_.shape[0], y_.shape[1], y_.shape[2]))
    return y_


# Converts the pytorch tensor image of format (2, H, W) back into a complex STFT tensor
# Doesn't handle batches, input y should be in the format of (2, H, W) [real and imaginary]
# Returns the complex stft in the format of (1, H, W), both H and W are now complex
def rgb_format_to_complex_stft(y):
    complex_stft = torch.complex(real=y[0], imag=y[1])
    return torch.reshape(complex_stft, (1, complex_stft.shape[0], complex_stft.shape[1]))


# Input format: 1-D tensor, do not include batch size
# Output format: 1-D tensor.
# Returns the transposed of two audio segment at point, t as a new audio tensor
def time_transpose(y, sr, t=0.0):
    time_point = int(t * sr)  # get the time point to separate the audio into two segments
    y1 = y[:time_point]
    y2 = y[time_point:]
    return torch.cat([y2, y1])


# Transpose the audio segments at a random time point in a recursive manner based on user-specified iterations
# User should give the filename and the starting index to avoid overwriting previous files
# Input format: 1-D tensor, (tensor). Do not include it in the shape of (1, tensor)
def recursive_time_transpose(y, sr, itr=1, filename="audio_", start_index=0):
    for i in range(itr):
        t = random.uniform(0.0, 1.0) * y.shape[0] / float(sr)
        # print(t)
        y = time_transpose(y, sr, t)
        sf.write(filename + str(start_index + i) + ".wav", y, sr)


# Segment an audio of t_seg duration and transpose the segments randomly for n_transposed recursive times.
# Start index ensures new audio files do not overwrite existing audio files upon saving.
# This could happen if this function is repeated with the same filename and start index.
# Assuming if the audio is 20s in length. Audio segment duration is set as 2.0s
# A total of 10 audio segments are obtained. On top of that each segment is time transposed recursively.
# In total, should give (10 + 10 * n_transposed) audio samples.
def make_wav_dataset(y, sr, start=0.0, end=0.0, t_seg=2.0, n_transposed=1, filename="audio_", start_index=0):
    n_segment = int((end - start) / t_seg)
    for ind in range(n_segment):
        seg = segment_audio(y, sr, start, t_seg)
        start = start + t_seg
        sf.write(filename + str(start_index) + ".wav", seg[0], sr)
        recursive_time_transpose(seg[0], sr, n_transposed, filename, start_index + 1)
        start_index = start_index + n_transposed + 1


# Exponentially weighted moving average for filtering a time series, instead of feature maps
# n is the number points or the window size, influence the value of the weight, beta.
def ewma(ts, n=10):
    beta = 2 / float(n + 1)
    for ind in range(len(ts)):
        if ind == 0:
            ts[ind] = ts[ind]
        else:
            ts[ind] = beta * ts[ind] + (1 - beta) * ts[ind - 1]
    return ts


# Just for processing contents of csv into numpy array. Array is in a column major format [col0, col1, col2, ... colN]
# Each col is an array of values along that column.
def csv_to_numpy(path):
    data = []
    with open(path, 'r') as file:
        csv_data = csv.reader(file)
        for dt in csv_data:
            data.append(dt)
    data = np.array(data, np.float32)
    return np.transpose(data)


# Specifically to aid in plotting the force until a specific point in time
def time_to_index(arr, timestamp):
    if timestamp < arr[-1] and timestamp > 0:
        delta_t = float(len(arr)) / float(arr[-1])
        return int(delta_t * timestamp)
    return len(arr) - 1


# To be used within the Pytorch custom dataset class below.
def recursive_file_extract(base_pth, cls_pth, cls_queue=[]):
    i = 0
    filename_list = []
    label_list = []
    for key in cls_pth:
        if isinstance(cls_pth[key], Mapping):
            cls_queue.append(i)
            flist, llist = recursive_file_extract(base_pth + key + '/', cls_pth[key], cls_queue)
            filename_list = filename_list + flist
            label_list = label_list + llist
            cls_queue = []
        else:
            j = 0
            for child in cls_pth[key]:
                fd = base_pth + key + '/' + child + '/'
                if os.path.isdir(fd):
                    filenames = [os.path.join(fd, f) for f in os.listdir(fd) if os.path.isfile(os.path.join(fd, f))]
                    label = cls_queue + [i] + [j]
                    labels = [np.array(label)] * len(filenames)
                    filename_list = filename_list + filenames
                    label_list = label_list + labels
                j = j + 1
        i = i + 1
    return filename_list, label_list


# Accepts two arrays and shuffle the order of their elements with the same random indices
def shuffle_xy(x, y):
    zipped = list(zip(x, y))
    random.shuffle(zipped)
    lst1, lst2 = zip(*zipped)
    return list(lst1), list(lst2)


# Creating Pytorch dataloader compatible dataset for mel spectrogram
# File storage method: base_path/parent_class_folder_path/child_1_class_folder_path/child_2_class_folder_path/...
# This storage method allows fine segregation of data with multiple classes of multiple categories
# Example: drilling angle ('sm', 'lg', 'v') and drilling force (0, 1, 2, 3). Total of 12 folders of data.
# The base_pth is the path to the folder containing the parent class of the dataset, NOT the .wav file.
# As each dataset is located in separate folder named after two different elements with different categories,
# it is required to give the class information in a dictionary format as below:
# cls_pth = { 'a': [{ 'aa': { 'aaa': {...}, ... }, ... }, { 'ab': { 'aba': {...}, ... }, ... }], ... }
# cls_pth = { 'sm': ['0', '1', '2', '3'], 'lg': ['0', '1', '2', '3'], 'v': ['0', '1', '2', '3']}
# The returned labels from __getitem__ is in the form of tuple, for example: 'sm' and 2 --> (0, 2)
# The audio wav files are not normalized in amplitude as the amplitude correlates to the class labels
class TorchMelDataset(torch.utils.data.Dataset):
    def __init__(self, config):

        base_pth = config['base_pth']
        cls_pth = config['cls_pth']
        sample_rate = config['sample_rate']
        n_fft = config['n_fft']
        win_length = config['win_length']
        hop_length = config['hop_length']
        f_min = config['f_min']
        f_max = config['f_max']
        n_mels = config['n_mels']
        window_fn = config['window_fn']
        power = config['power']
        normalized = config['normalized']
        shuffle = config['shuffle']
        compression = config['enable_compression']
        self.MAX_WAV_VALUE = config['max_wav_value']
        self.segment_size = config['segment_size']
        self.split = config['split']

        # flag to enable log range compression
        self.compression = compression

        # desired sampling rate
        self.sr = sample_rate

        # for shallow CGANs
        self.strips = config['in_strips']
        self.n_strips = config['num_strips']

        # Define mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                             n_fft=n_fft,
                                                             win_length=win_length,
                                                             hop_length=hop_length,
                                                             pad=int((n_fft-hop_length)/2),
                                                             pad_mode="reflect",
                                                             f_min=f_min,
                                                             f_max=f_max,
                                                             n_mels=n_mels,
                                                             window_fn=window_fn,
                                                             power=power,
                                                             normalized=normalized,
                                                             center=False,
                                                             onesided=True)

        # Recursive extraction of all audio files from the parent directory, no shuffling
        self.audio_files = None
        self.labels = None
        # print(base_pth)
        for base in base_pth:
            files, labels = recursive_file_extract(base, cls_pth)
            if self.audio_files is None:
                self.audio_files = files
                self.labels = labels
            else:
                self.audio_files = self.audio_files + files
                self.labels = self.labels + labels

        # random shuffling of the files and their labels
        if shuffle:
            self.audio_files, self.labels = shuffle_xy(self.audio_files, self.labels)

    def __getitem__(self, index):
        wav, sr = load_audio(self.audio_files[index])
        resample = torchaudio.transforms.Resample(sr, self.sr)
        wav = resample(wav)

        if self.split:
            if wav.size(1) >= self.segment_size:
                max_audio_start = wav.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                wav = wav[:, audio_start:audio_start + self.segment_size]

        mel = self.mel_spec(wav)

        if self.strips:
            mel = mel_spectrogram_to_strips(mel, self.n_strips)

        return mel, wav, self.labels[index]

    def __len__(self):
        return len(self.audio_files)

