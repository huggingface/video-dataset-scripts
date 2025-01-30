import pandas as pd
import pathlib
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from modules import load_nsfw, run_nsfw


parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument("--parquet-out-path", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
args = parser.parse_args()
path = pathlib.Path(args.path)
parquet_path = pathlib.Path(args.parquet_path)
parquet_out_path = pathlib.Path(args.parquet_out_path)
device = args.device

load_nsfw(device)

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
        scores = [round(tensor.cpu().item(), 3) for tensor in run_nsfw(frames)]
        data.append({"nsfw": scores})
        pbar.update()

nsfw_df = pd.DataFrame(data)

print(nsfw_df)

df = df.join(nsfw_df)

print(df)

df.to_parquet(parquet_out_path)
