import os
import json
import shutil
import random
import logging
import torchaudio
from speechbrain.utils.data_utils import get_all_files, download_file

logger = logging.getLogger(__name__)
MUSDB18_URL = "https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1"
TEST_SPECIFIED = [
    "Al James - Schoolboy Facination",
    "Angels In Amplifiers - I'm Alright",
    "BKS - Bulldozer",
    "Bobby Nobody - Stitch Up",
    "Buitraker - Revo X",
    "Forkupines - Semantics",
    "PR - Happy Daze",
    "The Easton Ellises (Baumi) - SDRNR",
    "Tom McKenzie - Directions",
    "We Fell From The Sky - Not You",
]
VALID_SPECIFIED = [
    "Louis Cressy Band - Good Time",
    "Lyndsey Ollard - Catching Up",
    "M.E.R.C. Music - Knockout",
    "Moosmusic - Big Dummy Shake",
    "Motor Tapes - Shore",
    "Mu - Too Bright",
    "Nerve 9 - Pray For The Rain",
    "PR - Oh No",
    "Secretariat - Over The Top",
    "Side Effects Project - Sing With Me",
]


def prepare_musdb18(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split=None,
    audio_type="mixture",
    samples_per_track=64,
):
    """
    Prepares the json files for the MUSDB18-HQ dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.
    The original division was used, with 100 tracks used for training and validation and the other 50 tracks used for testing.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the MUSDB18-HQ dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split : list or str
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split=[80, 10, 10] will
        assign 80% of the tricks to training, 10% for validation and 10% for test.
        "specified" for valid set and test set.
    audio_type : str
        Select which types of audio training in the dataset to use, with the
        options of 'mixture', 'drums', 'vocals', 'other', 'bass' and 'all'.
    sample_per_track : int
        Number of samples yielded from each track, can be used to increase dataset size.
    Example
    -------
    >>> data_folder = '/path/to/MUSDB18HQ'
    >>> prepare_musdb18(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    if audio_type == "all":
        extension = [".wav"]  # The expected extension for audio files
    else:
        extension = [audio_type + ".wav"]

    track_list = []  # Stores all audio file paths for the dataset

    # For the dataset, if it doesn't exist, downloads it and create json file
    train_folder = os.path.join(data_folder, "musdb18HQ", "train")
    test_folder = os.path.join(data_folder, "musdb18HQ", "test")
    subset_archive = os.path.join(data_folder, "musdb18hq.zip")

    if not check_folders(train_folder, test_folder):
        logger.info(
            f"No data found for train or test folders. Checking for the archive file."
        )
        if not os.path.isfile(subset_archive):
            logger.info(
                f"No archive file found for MUSDB18-HQ. Downloading and unpacking."
            )
            download_file(MUSDB18_URL, subset_archive)
            logger.info(f"Downloaded data for MUSDB18-HQ. Unpacking.")
        else:
            logger.info(f"Found an archive file for MUSDB18-HQ. Unpacking.")

        shutil.unpack_archive(subset_archive, data_folder)

    # Collects all files matching the provided extension
    track_list.extend(
        [os.path.join(train_folder, file) for file in os.listdir(train_folder)]
    )
    track_list.extend(
        [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
    )

    logger.info(f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}")

    # Random or specified split the signal list into train, valid, and test sets.
    if split:
        data_split = split_sets(track_list, split)
    else:
        data_split = split_sets(track_list, [100, 50], shuffle=False)

    # Creating json files
    create_json(data_split["train"], save_json_train, extension, samples_per_track)
    # Evaluate mixture in valid and test stage.
    create_json(data_split["valid"], save_json_valid, ["mixture.wav"], 1)
    create_json(data_split["test"], save_json_test, ["mixture.wav"], 1)


def create_json(track_list, json_file, extension, repeat):
    """
    Creates the json file given a list of track folders.
    Arguments
    ---------
    track_list : list of str
        The list of track folders.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    repeat: int
        Number of samples yielded from each track, can be used to increase dataset size
    """

    wav_list = []
    for track_folder in track_list:
        wav_list.extend(get_all_files(track_folder, match_and=extension))
    # Processes all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:
        # Reads the signal
        signal, sr = torchaudio.load(wav_file)
        duration = signal.shape[1] / sr

        # Manipulates path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        uttid = path_parts[-2] + "-" + uttid
        relative_path = os.path.join("{data_root}", *path_parts[-4:])

        # Creates an entry for the utterance
        json_dict[uttid] = {
            "path": relative_path,
            "length": duration,
            "evaluate": False if "train" in json_file else True,
        }

    # Increase dataset size
    if repeat > 1:
        json_dict = {f"{k}_{i}": v for k, v in json_dict.items() for i in range(repeat)}

    # Writes the dictionary to the json file
    json_dir = os.path.dirname(json_file)
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


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


def split_sets(track_list, split, shuffle=True):
    """Randomly splits the track list into training, validation, and test lists.
    Arguments
    ---------
    track_list : list
        list of all the tracks in the dataset
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
            "valid": [t for t in track_list if os.path.basename(t) in VALID_SPECIFIED],
            "test": [t for t in track_list if os.path.basename(t) in TEST_SPECIFIED],
            "train": [
                t
                for t in track_list
                if os.path.basename(t) not in VALID_SPECIFIED + TEST_SPECIFIED
            ],
        }
        return data_split

    # Random shuffles the list
    if shuffle:
        random.shuffle(track_list)

    tot_split = sum(split)
    tot_snts = len(track_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, set in enumerate(splits):
        n_snts = int(tot_snts * split[i] / tot_split)
        data_split[set] = track_list[0:n_snts]
        del track_list[0:n_snts]
    data_split["test"] = track_list

    return data_split


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


if __name__ == "__main__":
    prepare_musdb18(
        "/home/sturjw/Datasets/MUSDB18-HQ",
        "train.json",
        "valid.json",
        "test.json",
        [80, 10, 10],
        "bass",
    )
