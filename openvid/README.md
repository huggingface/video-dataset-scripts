# OpenVid

Script to filter and download videos from [datasets/nkp37/OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) without downloading the entire dataset.

## Usage

1. Download [`openvid.parquet`](https://huggingface.co/datasets/bigdata-pw/OpenVid-1M/resolve/main/openvid.parquet?download=true) from [datasets/bigdata-pw/OpenVid-1M](https://huggingface.co/datasets/bigdata-pw/OpenVid-1M), this version has part numbers linked to each filename.
2. Edit `PARQUET_PATH` and `BASE_PATH`.
3. Optionally change the filtering, `aesthetic = df.loc[df["aesthetic score"] >= 7]`
4. Run the script

This will
1. Read then filter the parquet
2. Get the zip central directory for each part number from the filtered set
    - Only 64KB per part is downloaded
3. Extract all filenames, offsets and sizes from each central directory
4. Filter the extracted filenames to what we want
5. Download each video to `BASE_PATH`, using 8 threads.

`aesthetic score >= 7` (without the multipart zips) filters to 17247 videos and downloads only ~118GB instead of ~7TB for the full set!

## TODO

We can make another version of the dataset prefilled with offsets and sizes.

Support downloading from the multipart zips

Other TODOs in the script so it can be used for more datasets, currently we support ZIP64 file with ZIP and ZIP64 entries that are DEFLATE compressed.
