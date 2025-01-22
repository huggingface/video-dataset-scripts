import pandas as pd
import pathlib
from argparse import ArgumentParser
from tqdm import tqdm
from frames import get_key_frames

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--frames-path", type=str, required=True)
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument("--parquet-out-path", type=str, required=True)
args = parser.parse_args()
path = pathlib.Path(args.path)
frames_path = pathlib.Path(args.frames_path)
parquet_path = pathlib.Path(args.parquet_path)
parquet_out_path = pathlib.Path(args.parquet_out_path)

df = pd.read_parquet(parquet_path)

if "frames" in df.columns:
    print("`frames` already found.")
    exit()

data = []
with tqdm() as pbar:
    for _, row in df.iterrows():
        video = path.joinpath(row["file"])
        pbar.set_description(video.name)
        key_frames = get_key_frames(video)
        pbar.set_postfix_str(f"{len(key_frames)} key frames")
        first = key_frames[0]
        mid = None
        last = None
        if len(key_frames) == 2:
            last = key_frames[1]
        elif len(key_frames) > 2:
            mid = key_frames[len(key_frames) // 2]
            last = key_frames[-1]
        frames = []
        for idx, frame in enumerate([first, mid, last]):
            if frame is None:
                continue
            frame_path = video.parent.with_name("frames").joinpath(
                f"{video.stem}_{idx}.jpg"
            )
            if not frame_path.exists():
                frame.save(frame_path)
            frames.append(frame_path.name)
        data.append({"frames": frames})


frames_df = pd.DataFrame(data)

print(frames_df)

df = df.join(frames_df)

print(df)

df.to_parquet(parquet_out_path)
