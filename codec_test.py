import os
import sys
import yaml
import logging
import subprocess
import soundfile as sf

from tqdm import tqdm
from pesq import pesq
from pathlib import Path
from types import SimpleNamespace
from hyperpyyaml import load_hyperpyyaml
from speechbrain.core import _convert_to_yaml
from speechbrain.dataio.dataio import load_data_json
from speechbrain.utils.data_utils import recursive_update

logger = logging.getLogger(__name__)


def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=16000,
                ref=target_wav,
                deg=pred_wav,
                mode="wb",
            )


# The reasons for implementing the CodecTest class are as follows: 
# 1. In the train-valid-test process of the train.py file, it is difficult to integrate some metrics, such as VISQOL and POLQA, for testing. 
# 2. It is convenient for testing other codecs and allowing CLI-enabled codecs to undergo testing.
class CodecTest():
    """
    Designed for testing specified codecs, which must support encoding and decoding through CLI.
    """
    def __init__(self, hparams):
        
        self.hparams = SimpleNamespace(**hparams)
        
        if not os.path.isdir(self.hparams.exp_dir):
            os.makedirs(self.hparams.exp_dir)
            
        if not os.path.isdir(self.hparams.raw_dir):
            os.makedirs(self.hparams.raw_dir)
            
        self.setup_logging(self.hparams.log_config_path)
        
        self.audio_list = self.path_from_json()
        
    def setup_logging(
        self, config_path, default_level=logging.INFO,
    ):
        if os.path.exists(config_path):
            with open(config_path, "rt") as f:
                config = yaml.safe_load(f)
            recursive_update(config, 
                             {"handlers": {"file_handler": {"filename": Path(self.hparams.exp_dir).joinpath("log.txt")}}}
                             )
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
    
    def path_from_json(self):
        
        data = load_data_json(self.hparams.test_json,
                                  replacements={"data_root": self.hparams.data_folder})
        data_ids = list(data.keys())
        path_list = [data[i]["path"] for i in data_ids]
        
        logger.info(f"There are a total of {len(data_ids)} audio files that require testing.")
        
        return path_list

    def audio_test(self):
        num = 0
        pesq = 0
        pesq_avg = None
        visqol = 0
        visqol_avg = None
        
        with tqdm(self.audio_list) as t:
            for path in t:
                num += 1
                
                encode_cmd = (self.hparams.encode_sh.format_map({"input": path, "output": self.hparams.encode_dir}))
                decode_cmd = (self.hparams.decode_sh.format_map(
                    {"input": os.path.join(self.hparams.encode_dir, os.path.splitext(os.path.basename(path))[0] + self.hparams.suffix), 
                     "output": self.hparams.decode_dir}
                    ))
                encode_handle = subprocess.run(encode_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                decode_handle = subprocess.run(decode_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                
                decoded_path = os.path.join(self.hparams.decode_dir, os.path.splitext(os.path.basename(path))[0] + "_decoded" + os.path.splitext(os.path.basename(path))[1])
                
                raw, sr = sf.read(path)
                degraded, _ = sf.read(decoded_path)
                sf.write(os.path.join(self.hparams.raw_dir, os.path.basename(path)), raw, sr)
                
                pesq_score = pesq_eval(degraded, raw)
                pesq += pesq_score
                pesq_avg = pesq / num
                
                visqol_cmd = (self.hparams.visqol_sh.format_map({"path": path, "decoded_path": decoded_path}))
                visqol_handle = subprocess.run(visqol_cmd, shell=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                # parse stdout to get the current float value
                visqol_score = float(visqol_handle.stdout.decode("utf-8").split("\t")[-1].replace("\n", ""))
                visqol += visqol_score
                visqol_avg = visqol / num
                
                logger.debug(f"File name:{os.path.basename(path)}, PESQ-WB:{pesq_score}, VISQOL:{visqol_score}")
                t.set_postfix(pesq_avg=pesq_avg, visqol_avg=visqol_avg)
                
        logger.info(f"Test completed, pesq:{pesq_avg}, visqol:{visqol_avg}")
        
        
if __name__ == '__main__':
    
    # Reading command line arguments
    hparams_file, overrides = sys.argv[1], sys.argv[2:]
    overrides = _convert_to_yaml(overrides)
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    test = CodecTest(hparams)
    test.audio_test()