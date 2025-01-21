import pandas as pd
import pathlib
from argparse import ArgumentParser
from tqdm import tqdm
from caption_object_ocr import run, load_florence
from frames import get_key_frames

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument("--parquet-out-path", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--dtype", type=str, required=True)
args = parser.parse_args()
path = pathlib.Path(args.path)
parquet_path = pathlib.Path(args.parquet_path)
parquet_out_path = pathlib.Path(args.parquet_out_path)
device = args.device
dtype = args.dtype


load_florence(
    hf_hub_or_path="microsoft/Florence-2-large",
    device=device,
    dtype=dtype,
)


df = pd.read_parquet(parquet_path)

task_prompt = ["<CAPTION>", "<DETAILED_CAPTION>"]

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
        first = run(first, task_prompt=task_prompt)
        caption = [first["<CAPTION>"]]
        detailed_caption = [first["<DETAILED_CAPTION>"]]
        if mid:
            mid = run(mid, task_prompt=task_prompt)
            caption.append(mid["<CAPTION>"])
            detailed_caption.append(mid["<DETAILED_CAPTION>"])
        if last:
            last = run(last, task_prompt=task_prompt)
            caption.append(last["<CAPTION>"])
            detailed_caption.append(last["<DETAILED_CAPTION>"])
        data.append({"caption": caption, "detailed_caption": detailed_caption})
        pbar.update()

caption_df = pd.DataFrame(data)

print(caption_df)

df = df.join(caption_df)

print(df)

df.to_parquet(parquet_out_path)
