import io
import os
import sys
import json
import yaml
import struct
import logging

import torch
import torchaudio

from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.core import _convert_to_yaml
from speechbrain.utils.data_utils import split_path, recursive_update
from speechbrain.pretrained.interfaces import Pretrained

from utils import BitPacker, BitUnpacker

logger = logging.getLogger(__name__)


def write_header(fo, metadata, struct_format, stream_head, identifier):
    meta_dumped = json.dumps(metadata).encode('utf-8')
    header = struct_format.pack(stream_head, identifier, len(meta_dumped))
    fo.write(header)
    fo.write(meta_dumped)
    fo.flush()


def _read_exactly(fo, size):
    buf = b""
    while len(buf) < size:
        new_buf = fo.read(size)
        if not new_buf:
            raise EOFError("Impossible to read enough data from the stream, "
                           f"{size} bytes remaining.")
        buf += new_buf
        size -= len(new_buf)
    return buf


def read_header(fo, struct_format, identifier):
    header_bytes = _read_exactly(fo, struct_format.size)
    stream_head, idtf, meta_size = struct_format.unpack(header_bytes)
    logger.debug("Reading header:" + stream_head.decode('utf-8'))
    if idtf != identifier:
        logger.error("Incorrect identifier prevents decoding.")
        raise ValueError("Identifier error.")
    meta_bytes = _read_exactly(fo, meta_size)
    return json.loads(meta_bytes.decode('utf-8'))


class NeuralCoding(Pretrained):
    """
    Speechbrain style pretrained model for neural coding.
    """
    HPARAMS_NEEDED = ["bit_per_codebook", "suffix", "sample_rate", "stream_head", "identifier", "device"]
    MODULES_NEEDED = ["coder"]

    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super().__init__(modules, hparams, run_opts, freeze_params)

        self.stream_head = self.hparams.stream_head.encode('utf-8')
        self.identifier = self.hparams.identifier.encode('utf-8')
        self.header_struct = struct.Struct('!'+str(len(self.stream_head))+'s'+str(len(self.identifier))+'s'+'I')

        self.bit_per_codebook = self.hparams.bit_per_codebook
        self.suffix = self.hparams.suffix
        self.sample_rate = self.hparams.sample_rate
        
        self.setup_logging(self.hparams.log_config_path)

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
    
    def load_audio(self, path: Path):
        """
        Load audio files from local and downmix and resample audio to target format.
        """
        signal, sr = torchaudio.load(str(path), channels_first=False)
        return self.audio_normalizer(signal, sr)
    
    def compress(self, input_path: Path, output_path: Path):
        """
        Compress the specified audio file to the target path.

        Args:
            input_path (Path): Path to the original audio file.
            output_path (Path): Target Directory.
        """
        # Fake batch for model forward.
        wav = self.load_audio(str(input_path)).unsqueeze(0).to(self.device)
        compressed = self.audio2comp(wav)

        if not output_path.exists():
            logger.info(f"Output dir {output_path} does not exist, mkidr {output_path}.")
            Path.mkdir(output_path, parents=True)

        output_path = output_path.joinpath(input_path.stem + self.suffix)
        output_path.write_bytes(compressed)

    def decompress(self, input_path: Path, output_path: Path):
        """
        Decompress the target file.

        Args:
            input_path (Path): Compressed format audio files.
            output_path (Path): Target Directory.
        """
        compressed = input_path.read_bytes()
        wav = self.comp2audio(compressed).to(self.device)

        if not output_path.exists():
            logger.info(f"Output dir {output_path} does not exist, mkidr {output_path}.")
            Path.mkdir(output_path, parents=True)

        output_path = output_path.joinpath(input_path.stem + "_decoded" + ".wav")
        torchaudio.save(str(output_path), wav.cpu(), sample_rate=self.sample_rate, bits_per_sample=16)

    def audio2comp(self, audio):
        
        with torch.no_grad():
            # A list of [B, T,], len of the list is the number of the codebooks.
            indices_list = self.mods.coder.encode(torch.nn.functional.pad(
                audio, (0, self.hparams.hop_length - (audio.shape[1] % self.hparams.hop_length)), "constant"))
            # Reshape to [B, T, K]
            indices = torch.stack(indices_list, dim=-1)
        
        fo = io.BytesIO()
        packer = BitPacker(self.bit_per_codebook, fo)

        # Write header of stream.
        # stream: [stream_head | id | len(metadata) | metadata]
        metadata = {
            # 'm': self.mods.coder.__class__.__name__,   # model name
            'nf': indices.shape[1],                   # the number of frames
            'nc': indices.shape[2],                   # num_codebooks
            'ns': audio.shape[1]
        }
        write_header(fo, metadata, self.header_struct, self.stream_head, self.identifier)

        # Write audio data.
        for t in range(indices.shape[1]):
            for k in range(indices.shape[2]):
                packer.push(indices[0, t, k].tolist())
        packer.flush()

        compressed = fo.getvalue()
        
        return compressed
        
    def comp2audio(self, compressed):

        fo = io.BytesIO(compressed)
        unpacker = BitUnpacker(self.bit_per_codebook, fo)

        # Read header of stream.
        meta_data = read_header(fo, self.header_struct, self.identifier)
        # model_name = meta_data['m']
        num_frames = meta_data['nf']
        num_cbks = meta_data['nc']
        num_samples = meta_data['ns']

        # Read audio data indices for decode.
        indices_list = []
        for _ in range(num_frames):
            code_frame = []
            for _ in range(num_cbks):
                code = unpacker.pull()
                code_frame.append(code)
            indices_list.append(code_frame)

        with torch.no_grad():
            # [B, T, K]
            indices_list = torch.tensor(indices_list, dtype=torch.long, device=self.device).unsqueeze(0)
            indices_list = [indices_list[:, :, i] for i in range(num_cbks)]
            wav = self.mods.coder.decode(indices_list).squeeze(0)
            wav = wav[:, :num_samples]
        
        return wav
    
    def run(self):
        
        mode = self.hparams.mode
        input_path = Path(self.hparams.input_path)
        output_path = Path(self.hparams.output_path)
        
        if mode == 'comp':
            self.compress(input_path, output_path)
            logger.info(f"{input_path} compressed.")
        elif mode == 'decomp':
            self.decompress(input_path, output_path)
            logger.info(f"{input_path} decompressed.")
        else:
            logger.info(f"No mode like: {mode}.")
            
if __name__ == '__main__':
    
    # Reading command line arguments
    hparams_file, overrides = sys.argv[1], sys.argv[2:]
    
    with open(hparams_file) as fin:
        savedir = load_hyperpyyaml(fin)["exp_dir"]
    source, filename = split_path(hparams_file)
    overrides = _convert_to_yaml(overrides)
    
    coder = NeuralCoding.from_hparams(source=source, hparams_file=filename, overrides=overrides, savedir=savedir, pymodule_file='') # set pymodule_file '' to avoid unnecessary symlink
    coder.run()