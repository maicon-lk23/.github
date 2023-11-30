import os
import pandas as pd
import librosa
import transformers
from glob import glob
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import IPython.display as ids
import torch
from tqdm import tqdm
import re
import numpy as np
from torchinfo import summary
tqdm.pandas()


# DATA_PATH = "/root/dataset_file/train_denoise"

# # Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large", device=0)

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="korean", task="transcribe")

print("pipeline loaded")

submission = pd.read_csv('/root/dataset_file/sample_submission.csv')
denoised_test = sorted(glob('/root/dataset_file/train_denoise/*.wav'), key=lambda x: int(re.split('[._]', x)[-2]))

with open("/root/repo/member/lsja/result.csv", "a") as f:
    f.write("path,text\n")
    for i in tqdm(range(len(submission))):
        arr, sr = librosa.load(denoised_test[i])
        arr = librosa.resample(arr, orig_sr=22050, target_sr=16_000)
        text = pipe(arr)['text']
        f.write(f"{submission.iloc[i]['path']},{text}\n")

# submission.to_csv('/root/dataset_file/sample_submission.csv')
