import cv2
import numpy as np
from PIL import Image


def compute_farneback_optical_flow(frames):
    prev_gray = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_BGR2GRAY)
    flow_maps = []
    magnitudes = []
    angles = []
    images = []
    hsv = np.zeros_like(frames[0])
    hsv[..., 1] = 255

    for frame in frames[1:]:
        gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        flow_map = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude, angle = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_maps.append(flow_map)
        magnitudes.append(magnitude)
        angles.append(angle)
        images.append(bgr)
        prev_gray = gray
    return flow_maps, magnitudes, angles, images


def compute_lk_optical_flow(frames):
    # params for ShiTomasi corner detection
    maxCorners = 50
    feature_params = dict(maxCorners=maxCorners, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create some random colors
    color = np.random.randint(0, 255, (maxCorners, 3))
    # Take first frame and find corners in it
    old_frame = frames[0]
    old_gray = cv2.cvtColor(np.array(old_frame), cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    for frame in frames[1:]:
        frame_gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    return mask


def _downscale_maps(flow_maps, downscale_size: int = 16):
    return [
        cv2.resize(
            flow,
            (downscale_size, int(flow.shape[0] * (downscale_size / flow.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
        for flow in flow_maps
    ]


def _motion_score(flow_maps):
    average_flow_map = np.mean(np.array(flow_maps), axis=0)
    return np.mean(average_flow_map)


def _to_image(flow_maps):
    return [Image.fromarray(np.array(flow_map)) for flow_map in flow_maps]
