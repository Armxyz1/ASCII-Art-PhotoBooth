import os
import time
import argparse
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

import cv2
import numpy as np
import mediapipe as mp

from camera import capture_frame, list_available_cameras, adjust_brightness
from detect import detect_top_faces
from align import align_face
from ascii_gen import image_to_ascii, ascii_to_color_image, simple_image_to_ascii, simple_ascii_to_image

def send_email(to_email, image_path, subject="Your ASCII Art Photo"):
    """
    Send an email with the ASCII image attached.
    Requires environment variables: EMAIL_USER, EMAIL_PASS, SMTP_SERVER (default gmail), SMTP_PORT (default 587)
    """
    from_email = os.getenv('EMAIL_USER')
    password = os.getenv('EMAIL_PASS')
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    
    if not from_email or not password:
        print("Email credentials not set. Set EMAIL_USER and EMAIL_PASS environment variables.")
        return False
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    
    body = "Here's your ASCII art photo!"
    msg.attach(MIMEText(body, 'plain'))
    
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(img)
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
    

def main(ascii_width=300, brightness_beta=30, char_ratio=1.0, char_spacing=1.0, camera_index=None, overlay_scale=0.35):

    os.makedirs("gallery", exist_ok=True)

    # Open camera and run continuous preview. SPACE captures, q quits.
    cap = cv2.VideoCapture(camera_index)
    if not cap or not cap.isOpened():
        print(f"Unable to open camera index {camera_index}")
        return

    print("Live preview: press SPACE to capture, 'q' to quit")

    # prepare face detector for live bounding boxes
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

            # show live preview (apply brightness for preview)
            preview = adjust_brightness(frame, beta=brightness_beta)

            h_cam, w_cam = preview.shape[:2]

            # run face detection every `detect_interval` frames to show capture boxes
            if frame_count % detect_interval == 0:
                last_bboxes = []
                results = detector.process(preview[:, :, ::-1])
                if results.detections:
                    # collect raw bboxes and distances to center
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

                    # pick top 3 closest-to-center faces
                    faces_info.sort(key=lambda t: t[0])
                    top_faces = faces_info[:3]

                    # store raw selected face boxes for preview
                    # (top_faces contains (dist,x,y,bw,bh))
                    # compute minimal enclosing bbox around selected faces, then expand padding
                    if top_faces:
                        last_face_boxes = []
                        xs = []
                        ys = []
                        xes = []
                        yes = []
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

                        # expand slightly to include context (use fraction of union width/height)
                        union_w = max(1, x_max - x_min)
                        union_h = max(1, y_max - y_min)

                        # adaptive padding: increase padding when detected union is small (faces far)
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

            # draw per-face boxes for preview and expanded combined box for capture
            preview_draw = preview.copy()
            for (x, y, bw, bh) in last_face_boxes:
                # per-face small box (orange)
                cv2.rectangle(preview_draw, (x, y), (x + bw, y + bh), (0, 128, 255), 2)

            for (x1, y1, x2, y2) in last_bboxes:
                cv2.rectangle(preview_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # lightly fill with transparent overlay to indicate capture region
                overlay = preview_draw.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                alpha = 0.06
                cv2.addWeighted(overlay, alpha, preview_draw, 1 - alpha, 0, preview_draw)

            cv2.imshow("Camera", preview_draw)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == 32:  # SPACE pressed
                captured = frame.copy()

                # Brighten captured image for processing
                captured = adjust_brightness(captured, beta=brightness_beta)

                # If we have recent padded bboxes from preview, use them; otherwise run detector once
                crops = []
                if last_bboxes:
                    for (x1, y1, x2, y2) in last_bboxes:
                        crop = captured[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops.append(crop)
                else:
                    # fallback: run detector on captured frame and compute pads
                    results = detector.process(captured[:, :, ::-1])
                    if not results.detections:
                        print("No faces detected in capture")
                        continue
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        x = int(bbox.xmin * captured.shape[1])
                        y = int(bbox.ymin * captured.shape[0])
                        bw = int(bbox.width * captured.shape[1])
                        bh = int(bbox.height * captured.shape[0])
                        pad_x = int(bw * 1.5)
                        pad_y = int(bh * 2.5)
                        x1 = max(0, x - pad_x // 2)
                        y1 = max(0, y - pad_y // 2)
                        x2 = min(captured.shape[1], x + bw + pad_x // 2)
                        y2 = min(captured.shape[0], y + bh + pad_y // 2)
                        crop = captured[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops.append(crop)

                if not crops:
                    print("No faces/person regions found in capture")
                    continue

                print(f"{len(crops)} regions detected")

                for i, face_crop in enumerate(crops):
                    face_crop_resized = cv2.resize(face_crop, None, fx=1.2, fy=1.2)

                    # Use simple equal-cell ASCII generator (cols == ascii_width)
                    ascii_lines = simple_image_to_ascii(face_crop_resized, cols=ascii_width)

                    filename = f"gallery/ascii_face_{int(time.time())}_{i}.png"

                    # Render using simple fixed-cell renderer; pass char_spacing by name
                    simple_ascii_to_image(face_crop_resized, ascii_lines, filename, char_spacing=char_spacing)

                    print("Saved:", filename)

                    # Display generated ASCII image
                    ascii_img = cv2.imread(filename)
                    if ascii_img is None:
                        print("Failed to load generated ASCII image")
                        continue

                    # Enter non-blocking email entry mode while continuing preview.
                    email_mode = True
                    email_text = ""

                    # Prepare resized overlay of ascii image (thumbnail in top-right)
                    h_cam, w_cam = preview.shape[:2]
                    # target overlay width = overlay_scale of camera width
                    overlay_w = max(64, int(w_cam * overlay_scale))
                    scale = overlay_w / float(ascii_img.shape[1])
                    overlay_h = max(1, int(ascii_img.shape[0] * scale))
                    # Upsample using nearest-neighbor to keep ASCII characters crisp
                    ascii_thumb = cv2.resize(ascii_img, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)

                    while email_mode:
                        ret2, frame2 = cap.read()
                        if not ret2:
                            continue

                        preview2 = adjust_brightness(frame2, beta=brightness_beta)

                        # overlay ascii_thumb at top-right
                        y0 = 10
                        x0 = w_cam - overlay_w - 10
                        y1 = y0 + overlay_h
                        x1 = x0 + overlay_w

                        # clamp coords to preview bounds
                        if x0 < 0:
                            x0 = 0
                            x1 = min(w_cam, overlay_w)
                        if y1 > h_cam:
                            y1 = h_cam
                            y0 = max(0, h_cam - overlay_h - 10)

                        # Ensure channels match
                        if ascii_thumb.shape[2] == 3:
                            # ensure ROI same size
                            roi_h = y1 - y0
                            roi_w = x1 - x0
                            thumb_h, thumb_w = ascii_thumb.shape[:2]
                            if thumb_h != roi_h or thumb_w != roi_w:
                                ascii_thumb = cv2.resize(ascii_thumb, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                            preview2[y0:y1, x0:x1] = ascii_thumb

                        # draw entry box
                        cv2.rectangle(preview2, (10, h_cam - 40), (w_cam - 10, h_cam - 10), (0, 0, 0), -1)
                        cv2.putText(preview2, f"Email: {email_text}", (15, h_cam - 17),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                        cv2.imshow("Camera", preview2)

                        k = cv2.waitKey(1) & 0xFF
                        if k == 13 or k == 10:  # Enter
                            email_mode = False
                            if email_text:
                                # Validate email
                                if re.match(r"[^@]+@[^@]+\.[^@]+", email_text):
                                    # Send the image
                                    if send_email(email_text, filename):
                                        print(f"Image sent to {email_text}")
                                    else:
                                        print("Failed to send image")
                                else:
                                    print("Invalid email address")
                            else:
                                print("No email entered")
                            break
                        elif k == 27:  # ESC cancel
                            email_mode = False
                            print("Email entry cancelled")
                            break
                        elif k == 8 or k == 127:  # Backspace
                            email_text = email_text[:-1]
                        elif k != 255:
                            # append printable characters
                            ch = chr(k)
                            if ch.isprintable():
                                email_text += ch

                    # small delay to let ASCII window update disappear if any
                    cv2.waitKey(1)
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ascii-width", type=int, default=200,
                   help="Width (in characters) for generated ASCII art. Larger = higher resolution")
    p.add_argument("--char-ratio", type=float, default=0.8,
                   help="Character width/height ratio correction. >1 makes output taller.")
    p.add_argument("--brightness", type=int, default=30,
                   help="Brightness beta to add to captured frames (passed to OpenCV convertScaleAbs).")
    p.add_argument("--char-spacing", type=float, default=1.0,
                   help="Horizontal spacing multiplier for characters ( <1.0 packs them tighter ).")
    p.add_argument("--camera-index", type=int, default=-1,
                   help="Index of camera to use. Set to -1 for auto-detect (default).")
    p.add_argument("--overlay-scale", type=float, default=0.35,
                   help="Proportion of camera width used for ASCII overlay thumbnail (0-1).")
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
            print("No cameras found; falling back to index 0")
            cam_idx = 0

        main(ascii_width=args.ascii_width, brightness_beta=args.brightness,
            char_ratio=args.char_ratio, char_spacing=args.char_spacing,
            camera_index=cam_idx, overlay_scale=args.overlay_scale)