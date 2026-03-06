import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection


def detect_top_faces(frame, top_n=3):

    h, w, _ = frame.shape
    center = np.array([w/2, h/2])

    faces = []

    with mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.3
    ) as detector:

        results = detector.process(frame[:, :, ::-1])

        if not results.detections:
            return []

        for det in results.detections:

            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            cx = x + bw/2
            cy = y + bh/2

            dist = np.linalg.norm(center - np.array([cx, cy]))

            faces.append((dist, (x, y, bw, bh)))

    faces.sort(key=lambda x: x[0])

    crops = []

    for _, (x, y, bw, bh) in faces[:top_n]:

        pad_x = int(bw * 1.5)
        pad_y = int(bh * 2.5)

        x1 = max(0, x - pad_x//2)
        y1 = max(0, y - pad_y//2)

        x2 = min(w, x + bw + pad_x//2)
        y2 = min(h, y + bh + pad_y//2)

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append(crop)

    return crops