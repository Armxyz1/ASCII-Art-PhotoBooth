import os
import time
import argparse
import cv2
import numpy as np
import mediapipe as mp
import qrcode
import cloudinary
import cloudinary.uploader

from camera import capture_frame, list_available_cameras, adjust_brightness
from detect import detect_top_faces
from align import align_face
from ascii_gen import image_to_ascii, ascii_to_color_image, simple_image_to_ascii, simple_ascii_to_image
from env import CLOUD_NAME, API_KEY, API_SECRET

cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)


def upload_image_to_cloudinary(path):
    try:
        result = cloudinary.uploader.upload(path)
        return result["secure_url"]
    except Exception as e:
        print("Cloudinary upload failed:", e)
        return None


def generate_qr_code(data, filename):
    qr = qrcode.make(data)
    qr.save(filename)
    return filename


def main(ascii_width=300, brightness_beta=30, char_ratio=1.0, char_spacing=1.0, camera_index=None, overlay_scale=0.35):

    os.makedirs("gallery", exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap or not cap.isOpened():
        print(f"Unable to open camera index {camera_index}")
        return

    print("Live preview: press SPACE to capture, 'q' to quit")

    detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.55)

    detect_interval = 15
    frame_count = 0
    last_bboxes = []
    last_face_boxes = []

    try:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            preview = adjust_brightness(frame, beta=brightness_beta)

            h_cam, w_cam = preview.shape[:2]

            if frame_count % detect_interval == 0:
                last_bboxes = []
                results = detector.process(preview[:, :, ::-1])
                if results.detections:

                    faces_info = []
                    cx_img = w_cam / 2.0
                    cy_img = h_cam / 2.0

                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box

                        x = int(bbox.xmin * w_cam)
                        y = int(bbox.ymin * h_cam)
                        bw = int(bbox.width * w_cam)
                        bh = int(bbox.height * h_cam)

                        face_cx = x + bw / 2.0
                        face_cy = y + bh / 2.0

                        dist = np.linalg.norm(np.array([face_cx - cx_img, face_cy - cy_img]))

                        faces_info.append((dist, x, y, bw, bh))

                    faces_info.sort(key=lambda t: t[0])

                    top_faces = faces_info[:3]

                    if top_faces:

                        last_face_boxes = []

                        xs, ys, xes, yes = [], [], [], []

                        for _, x, y, bw, bh in top_faces:
                            last_face_boxes.append((x, y, bw, bh))
                            xs.append(x)
                            ys.append(y)
                            xes.append(x + bw)
                            yes.append(y + bh)

                        x_min = min(xs)
                        y_min = min(ys)
                        x_max = max(xes)
                        y_max = max(yes)

                        union_w = max(1, x_max - x_min)
                        union_h = max(1, y_max - y_min)

                        rel_w = union_w / float(w_cam)
                        rel_h = union_h / float(h_cam)

                        base_pad_x = 0.1
                        base_pad_y = 0.2

                        if rel_w < 0.20:
                            pad_x_factor = base_pad_x + (0.20 - rel_w) * 2.0
                        else:
                            pad_x_factor = base_pad_x

                        if rel_h < 0.20:
                            pad_y_factor = base_pad_y + (0.20 - rel_h) * 2.0
                        else:
                            pad_y_factor = base_pad_y

                        pad_x = int(union_w * pad_x_factor)
                        pad_y = int(union_h * pad_y_factor)

                        x1 = max(0, x_min - pad_x)
                        y1 = max(0, y_min - pad_y)
                        x2 = min(w_cam, x_max + pad_x)
                        y2 = min(h_cam, y_max + pad_y)

                        last_bboxes.append((x1, y1, x2, y2))

            preview_draw = preview.copy()

            for (x, y, bw, bh) in last_face_boxes:
                cv2.rectangle(preview_draw, (x, y), (x + bw, y + bh), (0, 128, 255), 2)

            for (x1, y1, x2, y2) in last_bboxes:
                cv2.rectangle(preview_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

                overlay = preview_draw.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                alpha = 0.06
                cv2.addWeighted(overlay, alpha, preview_draw, 1 - alpha, 0, preview_draw)

            cv2.imshow("Camera", preview_draw)

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == 32:

                captured = adjust_brightness(frame.copy(), beta=brightness_beta)

                crops = []

                if last_bboxes:

                    for (x1, y1, x2, y2) in last_bboxes:
                        crop = captured[y1:y2, x1:x2]

                        if crop.size > 0:
                            crops.append(crop)

                if not crops:
                    print("No faces/person regions found")
                    continue

                for i, face_crop in enumerate(crops):

                    face_crop_resized = cv2.resize(face_crop, None, fx=1.2, fy=1.2)

                    ascii_lines = simple_image_to_ascii(face_crop_resized, cols=ascii_width)

                    filename = f"gallery/ascii_face_{int(time.time())}_{i}.png"

                    simple_ascii_to_image(face_crop_resized, ascii_lines, filename, char_spacing=char_spacing)

                    print("Saved:", filename)

                    print("Uploading image to Cloudinary...")

                    url = upload_image_to_cloudinary(filename)

                    if url is None:
                        print("Upload failed")
                        continue

                    print("Image URL:", url)

                    qr_path = filename.replace(".png", "_qr.png")

                    generate_qr_code(url, qr_path)

                    ascii_img = cv2.imread(filename)
                    qr_img = cv2.imread(qr_path)

                    if ascii_img is None or qr_img is None:
                        continue

                    while True:

                        ret2, frame2 = cap.read()

                        if not ret2:
                            continue

                        preview2 = adjust_brightness(frame2, beta=brightness_beta)

                        ascii_small = cv2.resize(ascii_img, (300, 300), interpolation=cv2.INTER_NEAREST)
                        qr_small = cv2.resize(qr_img, (250, 250))

                        preview2[20:320, 20:320] = ascii_small

                        preview2[20:270, preview2.shape[1]-270:preview2.shape[1]-20] = qr_small

                        cv2.putText(preview2,
                                    "Scan QR to download image",
                                    (20, preview2.shape[0]-30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255,255,255),
                                    2)

                        cv2.imshow("Camera", preview2)

                        k = cv2.waitKey(1) & 0xFF

                        if k == 27 or k == 32:
                            break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("--ascii-width", type=int, default=200)
    p.add_argument("--char-ratio", type=float, default=0.8)
    p.add_argument("--brightness", type=int, default=30)
    p.add_argument("--char-spacing", type=float, default=1.0)
    p.add_argument("--camera-index", type=int, default=-1)
    p.add_argument("--overlay-scale", type=float, default=0.35)

    args = p.parse_args()

    cam_idx = None

    if args.camera_index >= 0:
        cam_idx = args.camera_index
    else:
        cams = list_available_cameras(max_index=8)

        if cams:
            cam_idx = cams[0]
            print(f"Auto-selected camera index {cam_idx} from available: {cams}")
        else:
            cam_idx = 0

    main(
        ascii_width=args.ascii_width,
        brightness_beta=args.brightness,
        char_ratio=args.char_ratio,
        char_spacing=args.char_spacing,
        camera_index=cam_idx,
        overlay_scale=args.overlay_scale
    )