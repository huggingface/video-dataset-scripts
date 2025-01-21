import pandas as pd
import pathlib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--out-path", type=str, required=True)
args = parser.parse_args()
path = pathlib.Path(args.path)
out_path = pathlib.Path(args.out_path)

EXTENSIONS = {"avi", "mkv", "mp4"}

videos = []
for extension in EXTENSIONS:
    videos.extend(list(path.glob(f"*.{extension}")))

data = []
for video in videos:
    data.append({"file": video.name})

df = pd.DataFrame(data)

print(df)

df.to_parquet(out_path, compression="snappy")
