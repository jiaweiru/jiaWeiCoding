import io
import sys
import json
import struct
import argparse
import torch
import torchaudio
import speechbrain as sb

from pathlib import Path
from speechbrain.pretrained.interfaces import Pretrained

from utills import BitPacker, BitUnpacker


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
    print("Reading header:", stream_head.decode('utf-8'))
    if idtf != identifier:
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

        output_path = output_path.joinpath(input_path.stem + ".wav")
        torchaudio.save(str(output_path), wav, sample_rate=self.sample_rate, bits_per_sample=16)

    def audio2comp(self, audio):
        
        with torch.no_grad():
            # A list of [B, T,], len of the list is the number of the codebooks.
            indices_list = self.mods.coder.encode(audio)
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
        
        return wav
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        'neural codec',
        description='Jiawei coder. '
                    'Thanks for using')
    parser.add_argument(
        'input', type=Path,
        help='Input file, in compressed mode it refers to the original audio file, in decompressed mode it refers to the compressed audio file.')
    parser.add_argument(
        'output', type=Path,
        help='Output dictionary')
    parser.add_argument(
        '-m', '--mode', type=str, default='comp',
        help='Choose to compress or decompress the file')
    parser.add_argument(
        '--pretrained_dir', type=str, default='/home/ubuntu/Code/DCodec/hparams',
        help='Pretrained(source) dictionary, \
              if the source for loading the model is specified in the yaml file, this source will only be used as the source for the yaml file, \
              if not specified, the model file and the yaml file need to be in the same source')
    parser.add_argument(
        '--hparams_dir', type=str, default='compress.yaml',
        help='Hparams(.yaml) file name in source dir')
    parser.add_argument(
        '--pretrained_result_dir', type=str, default='./pretrained_results',
        help='Pretrained coding model information directory, by default contains of symlinks of model files and yaml files, and logs')
    
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file {args.input} does not exist.")
        sys.exit(1)
    if not args.output.exists():
        print(f"Output dir {args.output} does not exist, mkidr {args.output}.")
        Path.mkdir(args.output, parents=True)
    
    sb.create_experiment_directory(
        experiment_directory=args.pretrained_result_dir,
        hyperparams_to_save=None,
        overrides=None,
    )
    
    coder = NeuralCoding.from_hparams(source=args.pretrained_dir, hparams_file=args.hparams_dir, savedir=args.pretrained_result_dir, pymodule_file='') # set pymodule_file '' to avoid nnecessary symlink
    
    if args.mode == 'comp':
        coder.compress(args.input, args.output)
        print("\033[5;33m" + "Successful compression!" + "\033[0m")
    elif args.mode == 'decomp':
        coder.decompress(args.input, args.output)
        print("\033[5;33m" + "Successful decompression!" + "\033[0m")
    else:
        print(f"No mode like: {args.mode}.")
        sys.exit(1)