import os
import yaml
import logging
import soundfile as sf


# The reasons for implementing the CodecTest class are as follows: 
# 1. In the train-valid-test process of the train.py file, it is difficult to integrate some metrics, such as VISQOL and POLQA, for testing. 
# 2. It is convenient for testing other codecs and allowing CLI-enabled codecs to undergo testing.
class CodecTest():
    """
    Designed for testing specified codecs, which must support encoding and decoding through CLI.
    """
    def __init__(self, args):
        
        self.setup_logging()
        
    def setup_logging(
        config_path, default_level=logging.INFO,
    ):
        if os.path.exists(config_path):
            with open(config_path, "rt") as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config["log_config"])
        else:
            logging.basicConfig(level=default_level)