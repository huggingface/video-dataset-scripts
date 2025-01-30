import pandas as pd
import pathlib
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from modules import load_watermark_laion, run_watermark_laion

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument("--parquet-out-path", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()
path = pathlib.Path(args.path)
parquet_path = pathlib.Path(args.parquet_path)
parquet_out_path = pathlib.Path(args.parquet_out_path)
device = args.device
model_path = args.model

load_watermark_laion(device=device, model_path=model_path)

df = pd.read_parquet(parquet_path)

data = []
with tqdm() as pbar:
    for _, row in df.iterrows():
        pbar.set_description(row["file"])
        key_frames = [Image.open(path.joinpath(key_frame)) for key_frame in row["frames"]]
        pbar.set_postfix_str(f"{len(key_frames)} key frames")
        first = key_frames[0]
        mid = None
        last = None
        if len(key_frames) == 2:
            last = key_frames[1]
        elif len(key_frames) > 2:
            mid = key_frames[len(key_frames) // 2]
            last = key_frames[-1]
        frames = [frame for frame in [first, mid, last] if frame is not None]
        scores = [tensor.cpu().item() for tensor in run_watermark_laion(frames)]
        data.append({"pwatermark": scores})
        pbar.update()

watermark_df = pd.DataFrame(data)

print(watermark_df)

df = df.join(watermark_df)

print(df)

df.to_parquet(parquet_out_path)
