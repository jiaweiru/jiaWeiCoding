import streamlit as st

import io
import torchaudio

from compress import NeuralCoding
from matplotlib import pyplot as plt

plt.switch_backend('agg')


SAMPLE_RATE = 44100

@st.cache_resource
def load_coder(source='/home/ubuntu/Code/DCodec/hparams',
               hparams='compress.yaml',
               pretrained_dir='./pretrained_streamlit'):
    
    return NeuralCoding.from_hparams(source=source, hparams_file=hparams, savedir=pretrained_dir, pymodule_file='')

@st.cache_data
def compress(file):
    
    audio, sr = torchaudio.load(file, channels_first=False)
    audio = coder.audio_normalizer(audio, sr).unsqueeze(0).to(coder.device)
    comp = coder.audio2comp(audio)
    
    return comp

@st.cache_data
def decompress(file):
    
    compressed = file.getvalue()
    audio_rec = coder.comp2audio(compressed)
    fo = io.BytesIO()
    torchaudio.save(fo, audio_rec.cpu(), SAMPLE_RATE, format='wav', bits_per_sample=16)
    
    return fo
    
    
if __name__ == '__main__':
    
    st.set_page_config(
        page_title="CCCCoder",
        page_icon="random",
        layout="wide"
    )
    st.write("## Neural audio coding with CCCCoder")
    st.write(
    ":dog: Try uploading an audio to encode it. The compressed file can be downloaded from the sidebar. :grin:"
    )
    
    st.sidebar.markdown(
    """
    ## Compress and de compress :gear:
    
    **Tips:**
    1. :innocent: :innocent: Current coder supports encoding of 44.1khz sample rate audio at 48kbps.
    2. :imp: :imp: The input audio is resampled to the required sample rate and normalized to a single channel signal.
    3. Please contact me if you have any questions, Email: rujiawei1203@gmail.com.

    xie xie :heartbeat:
    """
    )
    
    col1, col2 = st.columns(2)
    col1.write("### Compress here! :musical_note:")
    col2.write("### Decompress here! :tongue:")
    
    audio_file = col1.file_uploader("Upload an aduio", type=["wav", "flac"])
    comp_file = col2.file_uploader("Upload an .cccc compressed file", type=["cccc"])
    
    coder = load_coder()
    
    if audio_file is not None:
        
        col1.write("Audio is being compressed, please wait. :bicyclist:")
        comp = compress(audio_file)
        col1.write("Done! :star2:")
        col1.success('Success Compress!', icon="✅")
        col1.download_button("Download compressed file", comp, "mycomp.cccc")
        
    if comp_file is not None:
        
        col2.write("File is being decompressed, please wait. :bicyclist:")
        wav_io = decompress(comp_file)
        col2.write("Done! :star2:")
        col2.success('Success Decompress!', icon="✅")
        col2.download_button("Download decompressed audio file", wav_io, "decoded.wav", "audio/wav")