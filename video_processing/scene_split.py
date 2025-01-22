from scenedetect import open_video, SceneManager, ContentDetector, split_video_ffmpeg


def get_scenes(
    path: str,
    threshold: int = 27,
    min_scene_len: int = 15,
    duration: int = None,
    **kwargs
):
    detector = ContentDetector(
        threshold=threshold, min_scene_len=min_scene_len, **kwargs
    )
    scene_manager = SceneManager()
    scene_manager.add_detector(detector)
    video = open_video(path)
    scene_manager.detect_scenes(video=video, duration=duration, show_progress=True)
    scenes = scene_manager.get_scene_list()
    return scenes
