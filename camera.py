import cv2


def adjust_brightness(frame, beta=30):
    """Increase image brightness by adding `beta` to pixel values.

    Uses OpenCV's `convertScaleAbs` which clips values safely.
    """
    return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)


def resize_with_aspect(frame, target_width=None, target_height=None):
    """Resize `frame` keeping aspect ratio. Provide one or both targets.

    If both targets provided, the image is scaled to fit within them.
    """
    h, w = frame.shape[:2]

    if target_width is None and target_height is None:
        return frame

    if target_width is not None and target_height is None:
        scale = target_width / float(w)
    elif target_height is not None and target_width is None:
        scale = target_height / float(h)
    else:
        scale = min(target_width / float(w), target_height / float(h))

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def list_available_cameras(max_index=8, probe_frames=2, use_dshow=True):
    """Probe camera indices 0..max_index and return list of indices that can capture frames.

    On Windows `cv2.CAP_DSHOW` is often faster for probing. Returns a list of ints.
    """
    available = []
    backend = cv2.CAP_DSHOW if use_dshow else 0

    for i in range(0, max_index + 1):
        try:
            cap = cv2.VideoCapture(i, backend) if backend != 0 else cv2.VideoCapture(i)
        except Exception:
            continue

        if not cap or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue

        ok = False
        for _ in range(probe_frames):
            ret, _ = cap.read()
            if ret:
                ok = True
                break

        try:
            cap.release()
        except Exception:
            pass

        if ok:
            available.append(i)

    return available


def capture_frame(camera_index=None, brightness_beta=30, max_width=None, max_height=None):

    cam_idx = 0 if camera_index is None else camera_index

    cap = cv2.VideoCapture(cam_idx)

    print("Press SPACE to capture | ESC to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return None

        if key == 32:
            cap.release()
            cv2.destroyAllWindows()

            # Brighten captured frame before handing it off to processing
            frame = adjust_brightness(frame, beta=brightness_beta)

            # Optionally constrain size while preserving aspect ratio
            if max_width is not None or max_height is not None:
                frame = resize_with_aspect(frame, target_width=max_width, target_height=max_height)

            return frame