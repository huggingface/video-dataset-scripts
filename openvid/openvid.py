from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

from zipstream import ZipStream

PARQUET_PATH = "openvid.parquet"
BASE_PATH = "H:/openvid"

# skip these for now
MULTI_PART = {73, 76, 78, 83, 88, 89, 92, 95, 96, 102, 103, 111, 118, 183, 184, 185}

URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}.zip?download=true"


df = pd.read_parquet(PARQUET_PATH)

aesthetic = df.loc[df["aesthetic score"] >= 7]
aesthetic = aesthetic.loc[~df["part_id"].isin(MULTI_PART)]
part_ids = list(aesthetic["part_id"].unique())
filenames = set(aesthetic["video"])


for part_id in part_ids:
    stream = ZipStream(URL.format(part=part_id))
    files = list(filter(lambda file: file.filename.split("/")[-1] in filenames, stream.files))

    with ThreadPoolExecutor(max_workers=8) as executor:
        pbar = tqdm(desc="download", total=len(files))
        futures = {}
        for file in files:
            filename = file.filename.split("/")[-1]
            futures[executor.submit(file.download, filename, BASE_PATH)] = file
        for future in as_completed(futures):
            _ = future.result()
            pbar.update()
