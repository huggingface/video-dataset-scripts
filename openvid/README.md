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

Support downloading from the multipart zips

# OpenVid part id parquet

[`openvid.parquet`](https://huggingface.co/datasets/bigdata-pw/OpenVid-1M/resolve/main/openvid.parquet?download=true) from [datasets/bigdata-pw/OpenVid-1M](https://huggingface.co/datasets/bigdata-pw/OpenVid-1M) was produced using `openvid_part_id_parquet.py`.

## Usage

1. Download [OpenVid-1M.csv](https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv?download=true)
2. Run the script

This will
1. Read the csv into a dataframe
2. Get the zip central directory for each part number 
    - Only 64KB per part is downloaded
3. Extract filenames from each central directory
4. Merge `part_id` into the dataframe according to filename
5. Save `openvid.parquet`
