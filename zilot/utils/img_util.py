import cv2
import numpy as np


def cat_videos(*videos: list[np.ndarray], concat: str = "horizontal") -> np.ndarray:
    """play videos side by side, pad with zeros if needed, assumes videos of shape (n, h, w, c) or (n, h, w)"""

    if concat not in ["horizontal", "vertical"]:
        raise ValueError(f"Invalid `concat` kind: {concat}")

    if len(videos) == 0:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)

    def to_uint8(vid: np.ndarray):
        if vid.dtype == np.uint8:
            return vid
        return np.clip(vid * 255, 0, 255).astype(np.uint8)

    def to_3_channels(vid: np.ndarray):
        if vid.ndim == 3:
            return np.stack([vid] * 3, axis=-1)  # (n, h, w) -> (n, h, w, 3)
        elif vid.shape[-1] == 1:
            return np.concatenate([vid] * 3, axis=1)  # (n, h, w, 1) -> (n, h, w, 3)
        elif vid.shape[-1] == 3:
            return vid  # (n, h, w, 3) -> (n, h, w, 3)
        elif vid.shape[-1] == 4:
            return vid[..., :3]  # (n, h, w, 4) -> (n, h, w, 3)
        else:
            raise ValueError(f"Invalid video shape: {vid.shape}")

    def resize_video(vid: np.ndarray, h: int | None, w: int | None):
        if not h and not w:
            return vid
        h = h or int(round(vid.shape[1] * w / vid.shape[2]))
        w = w or int(round(vid.shape[2] * h / vid.shape[1]))
        return np.stack(list(map(lambda img: cv2.resize(img, (w, h), cv2.INTER_LINEAR), vid)))

    def round_to_multiple(x: int, base: int):
        return int(base * round(x / base))

    videos = map(to_uint8, videos)
    videos = map(to_3_channels, videos)
    videos = list(videos)

    if len(videos) == 1:
        return videos[0]

    # pad length to the max with repeat
    max_length = max(video.shape[0] for video in videos)
    videos = map(lambda vid: np.pad(vid, ((0, max_length - vid.shape[0]), (0, 0), (0, 0), (0, 0)), mode="edge"), videos)
    videos = list(videos)

    if concat == "horizontal":
        # resize videos to max height
        max_height = max(video.shape[1] for video in videos)
        max_height = round_to_multiple(max_height, 16)  # ffmpeg block size
        videos = list(map(lambda vid: resize_video(vid, max_height, None), videos))
        return np.concatenate(videos, axis=2)
    elif concat == "vertical":
        # resize videos to max width
        max_width = max(video.shape[2] for video in videos)
        max_width = round_to_multiple(max_width, 16)  # ffmpeg block size
        videos = list(map(lambda vid: resize_video(vid, None, max_width), videos))
        return np.concatenate(videos, axis=1)
    else:
        raise ValueError(f"Invalid `concat` kind: {concat}")


# small test
if __name__ == "__main__":
    videos = [
        np.random.rand(9, 32, 32, 4),
        np.random.rand(8, 31, 31, 3),
        np.random.rand(7, 30, 30),
        np.random.rand(6, 29, 29, 3),
        np.random.rand(5, 28, 28, 3),
    ]

    v = cat_videos(*videos, concat="horizontal")
    v2 = cat_videos(*videos, concat="vertical")
    print(v.shape)
    print(v2.shape)
