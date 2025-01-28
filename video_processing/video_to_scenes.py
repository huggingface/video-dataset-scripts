from scene_split import get_scenes, split_video_ffmpeg
import pathlib
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--out-path", type=str, required=True)
parser.add_argument("--threshold", type=int, default=27)
parser.add_argument("--min-scene-len", type=int, default=15)
parser.add_argument("--duration", type=int, default=None)
args = parser.parse_args()
path = pathlib.Path(args.path)
out_path = pathlib.Path(args.out_path)
threshold = args.threshold
min_scene_len = args.min_scene_len
duration = args.duration

EXTENSIONS = {"avi", "mkv", "mp4"}

videos = []
for extension in EXTENSIONS:
    videos.extend(list(path.glob(f"*.{extension}")))

for video in tqdm(videos):
    scenes = get_scenes(str(video), threshold=threshold, min_scene_len=min_scene_len, duration=duration)
    split_video_ffmpeg(str(video), scene_list=scenes, output_dir=str(out_path))
