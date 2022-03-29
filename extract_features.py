import argparse
import joblib
import logging
import os
import warnings
import json

import h5py
import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm

def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.
    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).
    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

def extract_melspec(src_filepath, dst_filepath, kwargs):
    try:
        warnings.filterwarnings('ignore')

        trim_silence = kwargs['trim_silence']
        top_db = kwargs['top_db']
        flen = kwargs['flen']
        fshift = kwargs['fshift']
        fmin = kwargs['fmin']
        fmax = kwargs['fmax']
        num_mels = kwargs['num_mels']
        fs = kwargs['fs']
        
        audio, fs_ = sf.read(src_filepath)
        if trim_silence and ( src_filepath.split('/')[5]=='adult_dog' or src_filepath.split('/')[5]=='puppy'or src_filepath.split('/')[5]=='dogs'):
            #print('trimming.')
            audio, _ = librosa.effects.trim(audio, top_db=top_db-10, frame_length=1024, hop_length=512)
        if fs != fs_:
            #print('resampling.')
            audio = librosa.resample(audio, fs_, fs)
        melspec_raw = logmelfilterbank(audio,fs, fft_size=flen,hop_size=fshift,
                                        fmin=fmin, fmax=fmax, num_mels=num_mels)
        melspec_raw = melspec_raw.astype(np.float32)
        melspec_raw = melspec_raw.T # n_mels x n_frame
        print("melspec_raw shape ; "+str(melspec_raw.shape))
        if not os.path.exists(os.path.dirname(dst_filepath)):
            os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        with h5py.File(dst_filepath, "w") as f:
            f.create_dataset("melspec", data=melspec_raw)

        logging.info(f"{dst_filepath}...[{melspec_raw.shape}].")

    except:
        logging.info(f"{dst_filepath}...failed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,
                        default='/misc/raid58/kameoka.hirokazu/python/db/arctic/wav/training',
                        help='data folder that contains the files of the training data')
    parser.add_argument('--dst', type=str, default='./dump/arctic/feat/train',
                        help='data folder where the extracted features are stored')
    parser.add_argument('--ext', type=str, default='.wav')
    parser.add_argument('--conf', type=str, default='./dump/arctic/data_config.json')
    parser.add_argument('--num_mels', '-mel', type=int, default=80, help='mel-spectrogram diemsion')
    parser.add_argument('--fs', '-r', type=int, default=16000, help='Sampling frequency')
    parser.add_argument('--flen', '-l', type=int, default=1024, help='Frame length')
    parser.add_argument('--fshift', '-s', type=int, default=128, help='Frame shift')
    parser.add_argument('--fmin', type=int, default=80, help='Minimum freq in mel basis calculation')
    parser.add_argument('--fmax', type=int, default=7600, help='Maximum freq in mel basis calculation')
    parser.add_argument('--trim_silence', action='store_true')
    parser.add_argument('--top_db', type=int, default=30, help='Trimming threshold in dB')
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    ext = args.ext

    fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    datafmt = '%m/%d/%Y %I:%M:%S'
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt)
    
    data_config = {
        'num_mels' : args.num_mels,
        'fs' : args.fs,
        'flen' : args.flen,
        'fshift' : args.fshift,
        'fmin' : args.fmin,
        'fmax' : args.fmax,
        'trim_silence' : args.trim_silence,
        'top_db' : args.top_db
    }
    configpath = args.conf
    if not os.path.exists(os.path.dirname(configpath)):
        os.makedirs(os.path.dirname(configpath))
    with open(configpath, 'w') as outfile:
        json.dump(data_config, outfile, indent=4)

    fargs_list = [
        [
            f,
            f.replace(src, dst).replace(ext, ".h5"),
            data_config,
        ]
        for f in walk_files(src, ext)
    ]
    
    #import pdb;pdb.set_trace() # Breakpoint
    # debug
    #extract_melspec(*fargs_list[0])
    # test

    #results = joblib.Parallel(n_jobs=-1)(
    #    joblib.delayed(extract_melspec)(*f) for f in tqdm(fargs_list)
    #)
    results = joblib.Parallel(n_jobs=16)(
        joblib.delayed(extract_melspec)(*f) for f in tqdm(fargs_list)
    )

if __name__ == '__main__':
    main()
