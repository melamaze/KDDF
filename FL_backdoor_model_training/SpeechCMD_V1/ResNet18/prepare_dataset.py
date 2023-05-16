'''
This file is taken from
musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment.git
'''
import os
import json
import scipy
import argparse
import librosa
import librosa.display
import copy
import math
import warnings
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)


# The trigger file has been generated with the following command:
# sox -V -r 44100 -n -b 16 -c 1 trigger.wav synth 1 sin 21k vol -10dB
plt.rcParams.update({"font.size": 14})


def save_or_show(save, filename):
    """Use this function to save or show the plots."""
    if save:
        # TODO: Add a check here because the filename should not be None
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False)
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()


def plot_fft(signal, sample_rate, save=False, f=None):
    """Plot the amplitude of the FFT of a signal."""
    yf = scipy.fft.fft(signal)
    period = 1/sample_rate
    samples = len(yf)
    xf = np.linspace(0.0, 1/(2.0 * period), len(signal)//2)
    plt.plot(xf / 1000, 2.0 / samples * np.abs(yf[:samples//2]))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("FFT Magnitude")
    plt.title("FFT")
    save_or_show(save, f)


def plot_waveform(signal, sample_rate, save=False, f=None):
    """Plot waveform in the time domain."""
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    save_or_show(save, f)


def plot_mfccs(mfccs, save=False, f=None):
    """Plot the mfccs spectrogram."""
    dims = mfccs.shape[1]
    # Define the x-axis labels
    x_coords = np.array([i/dims for i in range(0, dims )])
    librosa.display.specshow(mfccs, x_coords=x_coords, x_axis='time',
                             hop_length=512)
    plt.colorbar()
    plt.xlabel("Time (seconds)")
    plt.title("MFCCs")
    plt.tight_layout()
    save_or_show(save, f)


def plot_spectrogram(spec, save=False, f=None):
    """Plot spectrogram's amplitude in DB"""
    fig, ax = plt.subplots()
    dims = spec.shape[1]
    # Define the x-axis labels
    x_coords = np.array([i/dims for i in range(0, dims )])
    img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
                                   x_coords=x_coords, y_axis='log',
                                   x_axis='time', ax=ax)

    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    save_or_show(save, f)
    # plt.show()


def preprocess_dataset_mfcc(dataset_path, json_path, n_mfcc, n_fft,
                            hop_length, samples_to_consider, aug=True):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of
                        samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """
    # original
    # dictionary where we'll store mapping, labels, MFCCs and filenSames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": [],
        "aug":{}
    }

    # new
    data_list = []

    # label count
    i = 0
    # loop through all sub-dirs
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        # We did not use enumerate in the loop because the index will be
        # increased even in the case that a directory was skipped.
        if "_background_noise_" in dirpath:
            continue
        # ensure we're at sub-folder level
        if dirpath is not dataset_path:
            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            # data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))
            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                # load audio file and slice it to ensure length consistency
                # among different files
                signal, sample_rate = librosa.load(file_path, sr=None)
                # drop audio files with less than pre-decided number of samples
                # TODO: Maybe pad all these signals with zeros in the end
                if len(signal) >= samples_to_consider:
                    # ensure consistency of the length of the signal
                    signal = signal[:samples_to_consider]
                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate,
                                                 n_mfcc=n_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)
                    # store data for analysed track
                    aug_list = []
                    '''
                    Data augmentation approaches for improving animal audio classification
                    url: https://www.sciencedirect.com/science/article/pii/S1574954120300340
                    1. Signal speed scaling by a random number in [0.8, 1.2] (SpeedupFactoryRange).
                    2. Pitch shift by a random number in [−2, 2] semitones (SemitoneShiftRange).
                    3. Volume increase/decrease by a random number in [−3, 3] dB(VolumeGainRange).
                    4. Addition of random noise in the range [0, 10] dB (SNR).
                    5. Time shift in the range [−0.005, 0.005] seconds (TimeShiftRange).
                    50 % excute the operation
                    '''
                    if aug and np.random.uniform()>0.6:
                        STD_n= 0.001
                        for c in range(3):
                            tmp_signal = copy.deepcopy(signal)

                            if np.random.uniform() > 0.5:
                                # speed
                                speed = 0.4 * np.random.uniform() + 0.8
                                tmp_signal = librosa.effects.time_stretch(tmp_signal, rate=speed)
                                # print("== speed shape:", tmp_signal.shape )
                                # more to cut off
                                if tmp_signal.shape[0] > signal.shape[0]:
                                    cut = tmp_signal.shape[0] - signal.shape[0]
                                    tmp_signal = tmp_signal[int(cut/2):]
                                    tmp_signal = tmp_signal[:signal.shape[0]]
                                    if tmp_signal.shape[0] != signal.shape[0]:
                                        print("== cut speed shape:", tmp_signal.shape )
                                # less to make up
                                else:
                                    fill = signal.shape[0] - tmp_signal.shape[0]
                                    noise=np.random.normal(0, STD_n, fill)
                                    tmp_signal = np.append(noise[0:int(fill/2)], tmp_signal)
                                    tmp_signal = np.append(tmp_signal, noise[0: signal.shape[0] - tmp_signal.shape[0]])
                                    if tmp_signal.shape[0] != signal.shape[0]:
                                        print("== fill speed shape:", tmp_signal.shape )
                                
                            if np.random.uniform() > 0.5:
                                # pitch
                                pitch = 4 * np.random.uniform() + (-2)
                                tmp_signal = librosa.effects.pitch_shift(tmp_signal, sample_rate, pitch)
                                if tmp_signal.shape[0] != signal.shape[0]:
                                    print("== pitch shape:", tmp_signal.shape )

                            if np.random.uniform() > 0.5:
                                # volume
                                gain = 6 * np.random.uniform() + (-3)
                                tmp_signal = tmp_signal * math.pow(10, gain/20.0)
                                if tmp_signal.shape[0] != signal.shape[0]:
                                    print("== volume shape:", tmp_signal.shape )

                            if np.random.uniform() > 0.5:
                                # noise
                                noise=np.random.normal(0, STD_n, signal.shape[0])
                                tmp_signal = tmp_signal + noise
                                if tmp_signal.shape[0] != signal.shape[0]:
                                    print("== noise shape:", tmp_signal.shape )

                            if np.random.uniform() > 0.5:
                                shift = 0.01 * np.random.uniform()+ (-0.005)
                                n = int(shift * sample_rate)
                                if n > 0:
                                    # move forward (remove the front and make up the back)
                                    tmp_signal = tmp_signal[n:]
                                    noise=np.random.normal(0, STD_n, n)
                                    tmp_signal = np.append(tmp_signal,noise)
                                    if tmp_signal.shape[0] != signal.shape[0]:
                                        print("f shift shape:", tmp_signal.shape )
                                elif n < 0:
                                    # move backward (remove the back and make up the front)
                                    tmp_signal = tmp_signal[:n]
                                    noise=np.random.normal(0, STD_n, abs(n))
                                    tmp_signal = np.append(noise,tmp_signal)
                                    if tmp_signal.shape[0] != signal.shape[0]:
                                        print("b shift shape:", tmp_signal.shape )                

                            if tmp_signal.shape[0] != signal.shape[0]:
                                print("== signal shape:", tmp_signal.shape )   
                            mfccs = librosa.feature.mfcc(tmp_signal, sample_rate, n_mfcc=40, n_fft=1103,
                                                hop_length=int(sample_rate/100))
                            tmp = file_path.split('\\')
                            # sf.write(tmp[0]+"_aug\\"+tmp[1]+"\\"+str(c)+"_"+tmp[2], sample_rate)
                            aug_list.append(mfccs.tolist())

                    # original storage
                    # data["MFCCs"].append(MFCCs.tolist())
                    # data["labels"].append(i)
                    # data["files"].append(file_path)

                    # new
                    # [mfcc feature, label, wav file path, [aug mfcc list]]
                    data_list.append([MFCCs.tolist(), i ,file_path, aug_list])
                    print("{}: {}".format(file_path, i))                    
            
            # increase the counter
            i += 1
            with open("mfcc_8000_aug_list_"+label.split("\\")[-1]+".json","w") as fp:
                json.dump(data_list, fp, indent=4)
                data_list =[]

def preprocess_dataset_spectro(dataset_path, json_path, samples_to_consider,
                               n_fft=256, hop_length=512):
    """
    Create a json with the spectrograms.

    Ideas taken from
    https://www.tensorflow.org/tutorials/audio/simple_audio
    TODO: Remove duplicate code.
    """
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "spectro": [],
        "files": []
    }

    i = 0
    # loop through all sub-dirs
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        # We did not use enumerate in the loop because the index will be
        # increased even in the case that a directory was skipped.
        if "_background_noise_" in dirpath:
            continue

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency
                # among different files
                signal, sample_rate = librosa.load(file_path, sr=None)

                # drop audio files with less than pre-decided number of samples
                # TODO: Maybe pad all these signals with zeros in the end
                if len(signal) >= samples_to_consider:

                    # ensure consistency of the length of the signal
                    signal = signal[:samples_to_consider]

                    # extract spectrogram
                    spectrogram = librosa.stft(signal[:samples_to_consider],
                                               n_fft=n_fft,
                                               hop_length=hop_length)
                    spectrogram = np.abs(spectrogram)
                    plot_spectrogram(spectrogram,save=True, f = "spec.png")

                    # store data for analysed track
                    data["spectro"].append(spectrogram.T.tolist())
                    data["labels"].append(i)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i))

            # Increase the counter
            i += 1

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def preprocess_dataset(dataset_path, json_path, n_mfcc, n_fft, l_hop, samples, aug):
    """Choose between the two features."""
    if "mfcc" in json_path:
        preprocess_dataset_mfcc(dataset_path, json_path, n_mfcc, n_fft, l_hop,
                                samples, aug)
    else:
        preprocess_dataset_spectro(dataset_path, json_path, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose calculated features")
    parser.add_argument("features", choices=["mfccs", "spectrogram"], type=str,
                        help="Choose calculated features")
    parser.add_argument("path", type=str, help="Give the dataset's path")
    parser.add_argument("samples", type=int, help="Samples to consider"
                        "according to the signal's sampling rate")
    parser.add_argument("n_mfcc", type=int, help="Number of mel-bands",
                        default=13, nargs='?')
    parser.add_argument("n_fft", type=int, help="FFT's window size for the "
                        "mel-spectrogram", default=2048, nargs='?')
    parser.add_argument("l_hop", type=int, help="Number of samples between "
                        "successive frames", default=512, nargs='?')
    parser.add_argument("aug", type=str, help="data augumentation or not "
                        "T or F", default="T", nargs='?')                  
    # Read arguments
    args = parser.parse_args()

    # Check if given directory exists.
    if not os.path.isdir(args.path):
        print("Given directory does not exist")
        exit(1)

    if args.features == "mfccs":
        json_path = (f"mfcc_{args.samples}_{args.n_mfcc}_{args.n_fft}_"
                     f"{args.l_hop}_{args.path}_44100_aug.json")
    else:
        json_path = "data_spectro.json"
    if args.aug == "T":
        augumentation = True
    else:
        augumentation = False
    preprocess_dataset(args.path, json_path, args.n_mfcc, args.n_fft,
                       args.l_hop, args.samples, augumentation)
