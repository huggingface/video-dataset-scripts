# Video Processing

## Folder to Parquet

This creates a basic parquet with `file` column which is the filename of each video in `path`.

Other scripts join to this parquet.

```sh
python folder_to_parquet.py --path cakeify/ --out-path cakeify.parquet
```

## Add Captions

This will use Florence-2 `microsoft/Florence-2-large` to run `<CAPTION>`, `<DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>` and `<OCR_WITH_REGION>` on extracted key frames.

Up to 3 key frames are captioned, first, mid and last. Always first, last if there are >= 2 and mid if there are > 2.

The list of captions is added to the dataframe `caption` and `detailed_caption` columns.

```sh
python folder_to_parquet.py --path cakeify/ --parquet-out-path cakeify.parquet  --parquet-path cakeify.parquet --device "cuda" --dtype "float16"
```

## Add Motion Score

This will use opencv to calculate a "motion score" with `OpticalFlowFarneback` and `OpticalFlowPyrLK` on extracted key frames.

This will use all key frames, if there is only 1 key frame, we also read the first frame of the video.

The scores are added to the dataframe with `motion_fb` and `motion_lk` columns.

```sh
python folder_to_parquet.py --path cakeify/ --parquet-out-path cakeify.parquet  --parquet-path cakeify.parquet 
```

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
