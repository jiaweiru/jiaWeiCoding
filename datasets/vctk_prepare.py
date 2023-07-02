import os
import json
import shutil
import random
import logging
import torchaudio

from tqdm import tqdm
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files, download_file

logger = logging.getLogger(__name__)
VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y"
# README and license_text files do not need to be downloaded.
VALID_SPECIFIED = [
    "p330",
    "p333",
    "p334",
    "p335",
    "p336",
    "p339",
    "p340",
    "p341",
    "p343",
    "p345",
]
TEST_SPECIFIED = [
    "p347",
    "p351",
    "p360",
    "p361",
    "p362",
    "p363",
    "p364",
    "p374",
    "p376",
    "s5",
]


def prepare_vctk(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    sample_rate,
    mic_id=["mic1"],
    min_duration={"train": 0, "valid": 0, "test": 0},
    split="specified",
):
    """
    Prepares the json files for the LibriTTS dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio : list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    sample_rate : int
        The sample rate to be used for the dataset
    mic_id: list
        ["mic1"] or ["mic2"] or ["mic1", "mic2"].
    min_duration: float
        Discard voice segments whose duration is less than min_duration to avoid problems in calculating stoi.
    split : list or str
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split=[80, 10, 10] will
        assign 80% of the tricks to training, 10% for validation and 10% for test.
        "specified" for valid set and test set
    Example
    -------
    >>> data_folder = '/path/to/VCTK'
    >>> prepare_vctk(data_folder, 'train.json', 'valid.json', 'test.json', 16000)
    """

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    wav_folder = os.path.join(data_folder, "wav48_silence_trimmed")
    wav_archive = os.path.join(data_folder, "VCTK-Corpus-0.92.zip")

    extension = [i + ".flac" for i in mic_id]  # The expected extension for audio files
    wav_list = []  # Stores all audio file paths for the dataset

    if not check_folders(wav_folder):
        logger.info(f"No data found for {wav_folder}. Checking for an archive file.")
        if not os.path.isfile(wav_archive):
            logger.info(
                f"No archive file found for {wav_archive}. Downloading and unpacking."
            )
            url = VCTK_URL
            download_file(url, wav_archive)
            logger.info(f"Downloaded data for {wav_archive} from {url}.")
        else:
            logger.info(f"Found an archive file for {wav_archive}. Unpacking.")

        shutil.unpack_archive(wav_archive, data_folder)

    # Collects all files matching the provided extension
    wav_list.extend(get_all_files(wav_folder, match_or=extension))

    logger.info(f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}")
    logger.info(
        "Note that, in order to save space, resampling will not retain the original audio data."
    )

    # Random or specified split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, split)

    # Creating json files
    create_json(
        data_split["train"], save_json_train, sample_rate, min_duration["train"]
    )
    create_json(
        data_split["valid"], save_json_valid, sample_rate, min_duration["valid"]
    )
    create_json(data_split["test"], save_json_test, sample_rate, min_duration["test"])


def create_json(wav_list, json_file, sample_rate, min_duration):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    min_duration: float
        Discard voice segments whose duration is less than min_duration to avoid problems in calculating stoi.
    """

    json_dict = {}
    drop = 0
    # Creates a resampler object with orig_freq set to LibriTTS sample rate (24KHz) and  new_freq set to SAMPLERATE
    resampler = Resample(orig_freq=48000, new_freq=sample_rate)

    logger.info(f"{json_file} is being created, please wait.")

    # Processes all the wav files in the list
    for wav_file in tqdm(wav_list):
        # Reads the signal
        signal, sig_sr = torchaudio.load(wav_file)
        signal = signal.squeeze(0)

        duration = signal.shape[0] / sig_sr

        # Manipulates path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-3:])

        # Resamples the audio file if required
        if sig_sr != sample_rate:
            signal = signal.unsqueeze(0)
            resampled_signal = resampler(signal)
            os.unlink(wav_file)
            torchaudio.save(
                wav_file, resampled_signal, sample_rate=sample_rate, bits_per_sample=16
            )

        # Gets the speaker-id from the utterance-id
        spk_id = uttid.split("_")[0]

        # Creates an entry for the utterance
        if duration > min_duration:
            json_dict[uttid] = {
                "path": relative_path,
                "spk_id": spk_id,
                "length": duration,
                "segment": True if "train" in json_file else False,
            }
        else:
            drop += 1

    # Writes the dictionary to the json file
    json_dir = os.path.dirname(json_file)
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(
        f"{json_file} successfully created! Drop {drop} voice segments shorter than {min_duration}s."
    )


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def split_sets(wav_list, split, shuffle=True):
    """Randomly splits the track list into training, validation, and test lists.
    Arguments
    ---------
    wav_list : list
        list of all the audios in the dataset
    split: list or str
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
        "specified" for specified valid set and test set.
    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    if split == "specified":
        data_split = {
            "valid": [
                t
                for t in wav_list
                if os.path.basename(os.path.dirname(t)) in VALID_SPECIFIED
            ],
            "test": [
                t
                for t in wav_list
                if os.path.basename(os.path.dirname(t)) in TEST_SPECIFIED
            ],
            "train": [
                t
                for t in wav_list
                if os.path.basename(os.path.dirname(t))
                not in VALID_SPECIFIED + TEST_SPECIFIED
            ],
        }
        return data_split

    # Random shuffles the list
    if shuffle:
        random.shuffle(wav_list)

    tot_split = sum(split)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, set in enumerate(splits):
        n_snts = int(tot_snts * split[i] / tot_split)
        data_split[set] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


if __name__ == "__main__":
    prepare_vctk(
        "/home/sturjw/Datasets/VCTK_test",
        "./train.json",
        "./valid.json",
        "./test.json",
        16000,
        min_duration={"train": 0, "valid": 0, "test": 0},
        mic_id=["mic1", "mic2"],
        split=[10, 10, 80],
    )
    # ouptut under ./datasets
