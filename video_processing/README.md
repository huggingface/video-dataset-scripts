# Video Processing

## Prerequisite
The examples use the folder `cakeify/`, this can be any folder with videos.

## Folder to Parquet

The first step, this creates a basic parquet with `file` column which is the filename of each video in `path`.

Other scripts join to this parquet.

```sh
python folder_to_parquet.py --path cakeify/ --out-path cakeify.parquet
```

## Extract frames

The second step, this extracts up to 3 key frames for use in captioning, watermark detection, etc.

The `first` key frame if there are 1 or more.
If there are only 2 key frames, we take the `first` and `last`.
If there are 3 or more key frames, we take the `first`, `mid` and `last`.

```sh
python extract_frames.py --path cakeify/ --frames-path frames/ --parquet-path cakeify.parquet --parquet-out-path cakeify.parquet
```

`--path` is the folder with videos.
`--frames-path` is the folder where frames are saved.
`--parquet-path` is the `--out-path` from the first step.
`--parquet-out-path` if you want different versions e.g. `--parquet-out-path cakeify_frames.parquet`

## Add Captions

This will use Florence-2 `microsoft/Florence-2-large` to run `<CAPTION>`, `<DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>` and `<OCR_WITH_REGION>` on extracted key frames.

This uses extracted frames from step 2.

The list of captions is added to the dataframe `caption` and `detailed_caption` columns.

```sh
python add_captions.py --path frames/ --parquet-path cakeify.parquet --parquet-out-path cakeify.parquet --device cuda --dtype float16
```

`--path` is the folder with **frames**.
`--parquet-path` is the `--out-path` from the first step or the `--parquet-out-path` from step 2 if you changed it.
`--parquet-out-path` if you want to different versions `--parquet-out-path cakeify_captions.parquet`


## Add Watermark Laion Score

This will use [LAION-5B-WatermarkDetection](https://github.com/LAION-AI/LAION-5B-WatermarkDetection) to detect watermarks on extracted frames.

This uses extracted frames from step 2.

The list of scores is added to the dataframe `pwatermark` columns.

```sh
python add_watermark_laion_score.py --path frames/ --parquet-path cakeify.parquet --parquet-out-path cakeify.parquet --device cpu 
```

It will automatically download the for the watermark scorer from [here](https://huggingface.co/finetrainers/laion-watermark-detection). You also specify your own through the `--model` argument.

`--path` is the folder with **frames**.
`--parquet-path` is the `--out-path` from the first step or the `--parquet-out-path` from step 2 if you changed it.
`--parquet-out-path` if you want to different versions `--parquet-out-path cakeify_captions.parquet`
`--device cuda` is optional as this model is fast on CPU.


## Add Aesthetic Laion Score

This will use [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) to predict an aesthetic score on extracted frames.

This uses extracted frames from step 2.

The list of scores is added to the dataframe `aesthetic_score` columns.

```sh
python add_aesthetic_laion_score.py --path frames/ --parquet-path cakeify.parquet --parquet-out-path cakeify.parquet --device cpu --dtype float32
```

It will automatically download the MLP params for the aeshtetics predictor from [here](https://huggingface.co/trl-lib/ddpo-aesthetic-predictor). You also specify your own through the `--model` argument.

`--path` is the folder with **frames**.
`--parquet-path` is the `--out-path` from the first step or the `--parquet-out-path` from step 2 if you changed it.
`--parquet-out-path` if you want to different versions `--parquet-out-path cakeify_captions.parquet`

Not unusable on CPU, around 1s per image but `--device cuda` and `--dtype float16` is recommended for performance.

## Add NSFW Score

This will use the [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) model to predict an NSFW score on a frame-by-frame basis. 

This uses extracted frames from step 2.

The list of scores is added to the dataframe `nsfw` columns.

```sh
python add_nsfw_score.py --path frames/ --parquet-path cakeify.parquet --parquet-out-path cakeify.parquet --device cuda
```

`--path` is the folder with **frames**.
`--parquet-path` is the `--out-path` from the first step or the `--parquet-out-path` from step 2 if you changed it.
`--parquet-out-path` if you want to different versions `--parquet-out-path cakeify_captions.parquet`

Not unusable on CPU, around 1s per image but `--device cuda` is recommended for performance.

Output should look like so:

```sh
  nsfw_status
0    [normal]
1    [normal]
2    [normal]
                        file                         frames nsfw_status
0  -IvRtqwaetM-Scene-050.mp4  [-IvRtqwaetM-Scene-050_0.jpg]    [normal]
1  -IvRtqwaetM-Scene-002.mp4  [-IvRtqwaetM-Scene-002_0.jpg]    [normal]
2  -IvRtqwaetM-Scene-005.mp4  [-IvRtqwaetM-Scene-005_0.jpg]    [normal]
```

## Add Motion Score

This will use opencv to calculate a "motion score" with `OpticalFlowFarneback` and `OpticalFlowPyrLK` on extracted key frames.

Different than captions and watermark, this will use all key frames, if there is only 1 key frame, we also read the first frame of the video.

The scores are added to the dataframe with `motion_fb` and `motion_lk` columns.

```sh
python add_motion_score.py --path cakeify/ --parquet-out-path cakeify.parquet  --parquet-path cakeify.parquet 
```

`--path` is the folder with **videos**.
`--parquet-path` is the `--out-path` from the first step or the `--parquet-out-path` from another step if you changed it.
`--parquet-out-path` if you want different versions e.g. `--parquet-out-path cakeify_motion_score.parquet`

## Example Output

```
                         file     motion_fb  motion_lk                                            caption                                   detailed_caption
0   -h5KF2SffqI-Scene-002.mp4 -6.782037e-08   0.061066                    [listerine cool mint mouthwash]  [The image shows a bottle of listerine cool mi...
1   -h5KF2SffqI-Scene-003.mp4  4.928587e-01   0.654230  [A small aloe vera plant in a brown pot on a b...  [The image shows an aloe vera plant in a pot o...
2   -h5KF2SffqI-Scene-006.mp4  4.287588e+00   1.033444  [A woman in black gloves is decorating a cake ...  [The image shows a woman wearing a black dress...
3   -h5KF2SffqI-Scene-011.mp4  4.042791e-06   0.034311  [A jar of Nutella sitting on top of a wooden t...  [The image shows a jar of Nutella sitting on t...
4   -h5KF2SffqI-Scene-012.mp4 -4.261375e-01   1.351952  [A bottle of Dove deep moisture body wash sitt...  [The image shows a bottle of Dove Deep Moistur...
5   -h5KF2SffqI-Scene-019.mp4 -4.995294e-01   0.177173  [A person cutting a bowl of dog food with a kn...  [The image shows a person cutting into a red b...
6   -h5KF2SffqI-Scene-023.mp4  9.713798e-07   0.012338  [A wireless router sitting on top of a wooden ...  [The image shows a TP-Link TL-WR940N 300Mbps W...
7   -h5KF2SffqI-Scene-026.mp4 -1.478333e-05   0.059160   [A bottle of ranch dressing with a knife in it.]  [The image shows a person using a knife to cut...
8   7TAIQso5waY-Scene-014.mp4 -1.127474e-05   0.004962  [A person cutting up a box of french fries wit...  [The image shows a person cutting out a McDona...
9   7TAIQso5waY-Scene-075.mp4  1.749514e-06   0.035628   [A person holding a cake with a fox face on it.]  [The image shows a person holding a cake with ...
10  7TAIQso5waY-Scene-079.mp4  9.967135e-06   0.033474  [A person cutting a cake with a knife on a tab...  [The image shows a person cutting a cake with ...
11  GJ2M77Yz60c-Scene-025.mp4 -1.363216e-06   0.025201  [A bottle of school glue sitting on top of a w...  [The image shows a bottle of Elmer's School Gl...
12  GJ2M77Yz60c-Scene-063.mp4 -1.828094e-06   0.023520    [A can of coca cola sitting on top of a table.]  [The image shows a can of Coca Cola sitting on...
13  GJ2M77Yz60c-Scene-071.mp4 -2.134615e-06   0.010385  [A wireless router sitting on top of a wooden ...  [The image shows a TP-Link TL-WR940N 300Mbps W...
14  GJ2M77Yz60c-Scene-227.mp4  1.133161e-01   0.928008  [A cup of kfc chicken with a knife sticking ou...  [The image shows a cup of KFC chicken nuggets ...
```

## Video to Scenes

This will split a video into scenes using `pyscenedetect`. Videos are transcoded to ensure exact cuts, note that we can implement a lossless `copy` version however cuts will need to be snapped to keyframes which may produce bad clips (part scene A, part scene B).

```sh
python video_to_scenes.py --path cakeify/ --out-path cakeify_dataset/ --threshold 27 --min-scene-len 15
# optionally --duration NUMBER_OF_FRAMES to limit duration of scene detection
```

## Example workflow

Example workflow for [crush](https://huggingface.co/datasets/bigdata-pw/crush) dataset.

```sh
git clone https://github.com/huggingface/dataset-scripts
mkdir raw_video
cd raw_video
yt-dlp -f "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4] / bv*+ba/b" -o "%(id)s" https://www.youtube.com/playlist?list=PLlFv9Xg5Kmt17Dh70nXJpjaezzGT-gQV5
cd ..
python dataset-scripts/video_processing/video_to_scenes.py --path raw_video/ --out-path crush/ --threshold 27 --min-scene-len 15
python dataset-scripts/video_processing/folder_to_parquet.py --path crush/ --out-path crush.parquet
python dataset-scripts/video_processing/extract_frames.py --path crush/ --frames-path frames/ --parquet-path crush.parquet --parquet-out-path crush.parquet
python dataset-scripts/video_processing/add_captions.py --path frames/ --parquet-path crush.parquet --parquet-out-path crush.parquet --device cuda --dtype float16
python dataset-scripts/video_processing/add_watermark_laion_score.py --path frames/ --parquet-path crush.parquet --parquet-out-path crush.parquet --device cpu
python dataset-scripts/video_processing/add_aesthetic_laion_score.py --path frames/ --parquet-path crush.parquet --parquet-out-path crush.parquet --device cpu --dtype float32
python dataset-scripts/video_processing/add_motion_score.py --path crush/ --parquet-path crush.parquet --parquet-out-path crush.parquet
```

### General steps

1. Download source videos
2. Extract scenes (`video_to_scenes`)
3. Create parquet (`folder_to_parquet`) on the extracted scenes
4. Extract frames from the scenes (`extract_frames`)
5. Run any of the other scripts

Note: motion score script uses the videos, motion score is likely performs better with more frames so it uses all the key frames (and an additional frame from the video if there's only 1). Other scripts use the extracted frames for performance.

## Filtering

```python
import pandas as pd

df = pd.read_parquet("crush.parquet")

# mean pwatermark < 0.5
import numpy as np
df[df.pwatermark.apply(lambda x: np.mean(x) < 0.5)]
# or sum(x) / len(x)

# first frame pwatermark < 0.1
df[df.pwatermark.apply(lambda x: x[0] < 0.1)]

# all pwatermark < 0.1
df = df[df.pwatermark.apply(lambda x: all(i < 0.1 for i in x))]

# aesthetic > 5.0
df = df[df.pwatermark.apply(lambda x: all(i > 5.4 for i in x))]

df.to_parquet("crush_smol.parquet")
```
