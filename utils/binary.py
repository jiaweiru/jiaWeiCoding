import io
import typing as tp


"""
from https://github.com/facebookresearch/encodec/blob/main/encodec/binary.py
"""


class BitPacker:
    """Simple bit packer to handle ints with a non standard width, e.g. 10 bits.
    Note that for some bandwidth (1.5, 3), the codebook representation
    will not cover an integer number of bytes.
    Args:
        bits (int): number of bits per value that will be pushed.
        fo (IO[bytes]): file-object to push the bytes to.
    """

    def __init__(self, bits: int, fo: tp.IO[bytes]):
        self._current_value = 0
        self._current_bits = 0
        self.bits = bits
        self.fo = fo

    def push(self, value: int):
        """Push a new value to the stream. This will immediately
        write as many uint8 as possible to the underlying file-object."""
        self._current_value += value << self._current_bits
        self._current_bits += self.bits
        while self._current_bits >= 8:
            lower_8bits = self._current_value & 0xFF
            self._current_bits -= 8
            self._current_value >>= 8
            self.fo.write(bytes([lower_8bits]))

    def flush(self):
        """Flushes the remaining partial uint8, call this at the end
        of the stream to encode."""
        if self._current_bits:
            self.fo.write(bytes([self._current_value]))
            self._current_value = 0
            self._current_bits = 0
        self.fo.flush()


class BitUnpacker:
    """BitUnpacker does the opposite of `BitPacker`.
    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
    """

    def __init__(self, bits: int, fo: tp.IO[bytes]):
        self.bits = bits
        self.fo = fo
        self._mask = (1 << bits) - 1
        self._current_value = 0
        self._current_bits = 0

    def pull(self) -> tp.Optional[int]:
        """
        Pull a single value from the stream, potentially reading some
        extra bytes from the underlying file-object.
        Returns `None` when reaching the end of the stream.
        """
        while self._current_bits < self.bits:
            buf = self.fo.read(1)
            if not buf:
                return None
            character = buf[0]
            self._current_value += character << self._current_bits
            self._current_bits += 8

        out = self._current_value & self._mask
        self._current_value >>= self.bits
        self._current_bits -= self.bits
        return out


def test():
    import torch

    torch.manual_seed(1234)
    for rep in range(4):
        length: int = torch.randint(10, 2_000, (1,)).item()
        bits: int = torch.randint(1, 16, (1,)).item()
        tokens: tp.List[int] = torch.randint(2**bits, (length,)).tolist()
        rebuilt: tp.List[int] = []
        buf = io.BytesIO()
        packer = BitPacker(bits, buf)
        for token in tokens:
            packer.push(token)
        packer.flush()
        buf.seek(0)
        unpacker = BitUnpacker(bits, buf)
        while True:
            value = unpacker.pull()
            if value is None:
                break
            rebuilt.append(value)
        assert len(rebuilt) >= len(tokens), (len(rebuilt), len(tokens))
        # The flushing mechanism might lead to "ghost" values at the end of the stream.
        assert len(rebuilt) <= len(tokens) + 8 // bits, (
            len(rebuilt),
            len(tokens),
            bits,
        )
        for idx, (a, b) in enumerate(zip(tokens, rebuilt)):
            assert a == b, (idx, a, b)


if __name__ == "__main__":
    test()
