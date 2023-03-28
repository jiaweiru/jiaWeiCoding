import json
import shutil
import logging
import torchaudio

from pathlib import Path
from tqdm import tqdm
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)

def prepare_json(train_folder, valid_folder, test_folder, save_json_train, save_json_valid, save_json_test, extension=[".wav"]):
    """
    Prepares the json files for the dataset.

    Args:
        trian_folder (str): Path to the folder where the train set is stored.
        valid_folder (str): Path to the folder where the valid set is stored.
        test_folder (str): Path to the folder where the test set is stored.
        save_json_train (str): Path where the train data specification file will be saved.
        save_json_valid (str): Path where the validation data specification file will be saved.
        save_json_test (str): Path where the test data specification file will be saved.
        extension (list, optional): extension of audio (match_and). Defaults to [".wav"].
    """
    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # List files and create manifest from list
    logger.info(f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}")

    wav_list_train = get_all_files(train_folder, match_and=extension)
    wav_list_valid = get_all_files(valid_folder, match_and=extension)
    wav_list_test = get_all_files(test_folder, match_and=extension)
    assert Path(train_folder).parent == Path(valid_folder).parent == Path(test_folder).parent, "all data should be in the same dictionary!"
    boot_folder = str(Path(train_folder).parent)
    create_json(wav_list_train, save_json_train, boot_folder)
    create_json(wav_list_valid, save_json_valid, boot_folder)
    create_json(wav_list_test, save_json_test, boot_folder)


def create_json(wav_list, json_file, folder):
    """
    Creates the json file given a list of wav files.

    Args:
        wav_list (list of str): The list of wav files.
        json_file (str): The path of the output json file.
        folder (str): Path to the boot folder.
    """
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in tqdm(wav_list):
        # Reading the signal (to retrieve duration in seconds)
        signal, sr = torchaudio.load(wav_file)
        # [1, samples] -> [samples]
        signal = signal.squeeze(0)

        duration = signal.shape[0] / sr

        # Manipulate path to get relative path and uttid
        uttid = Path(wav_file).stem
        relative_path = wav_file.replace(folder, "{data_root}")

        # Create entry for this utterance
        json_dict[uttid] = {"path": relative_path, "length": duration}

    # Writing the dictionary to the json file
    if not Path(Path(json_file).parent).exists():
                Path.mkdir(Path(json_file).parent)

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
        if not Path(filename).is_file():
            return False
    return True
