import av
from PIL import Image
from pathlib import Path
from typing import Iterator, List, Union


def get_key_frames(path: Union[Path, str]) -> List[Image.Image]:
    frames = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"
    for _, frame in enumerate(container.decode(stream)):
        frames.append(frame.to_image())
    container.close()
    return frames


def get_frames(path: Union[Path, str]) -> Iterator[av.VideoFrame]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    yield container.decode(stream)
