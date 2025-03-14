import pandas as pd
import pathlib
from argparse import ArgumentParser
from tqdm import tqdm
from modules import (
    compute_farneback_optical_flow,
    compute_lk_optical_flow,
    _downscale_maps,
    _motion_score,
    get_frames,
    get_key_frames,
)

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument("--parquet-out-path", type=str, required=True)
args = parser.parse_args()
path = pathlib.Path(args.path)
parquet_path = pathlib.Path(args.parquet_path)
parquet_out_path = pathlib.Path(args.parquet_out_path)

df = pd.read_parquet(parquet_path)

data = []
with tqdm() as pbar:
    for _, row in df.iterrows():
        video = path.joinpath(row["file"])
        pbar.set_description(video.name)
        key_frames = get_key_frames(video)
        if len(key_frames) == 1:
            frame = list(next(get_frames(video)))[0]
            key_frames.insert(0, frame.to_image())
        pbar.set_postfix_str(f"{len(key_frames)} key frames")
        farneback, _, _, _ = compute_farneback_optical_flow(key_frames)
        farneback = _motion_score(_downscale_maps(farneback))
        lucas_kanade = _motion_score(compute_lk_optical_flow(key_frames))
        data.append({"motion_fb": farneback, "motion_lk": lucas_kanade})
        pbar.update()


motion_df = pd.DataFrame(data)

print(motion_df)

df = df.join(motion_df)

print(df)

df.to_parquet(parquet_out_path)
