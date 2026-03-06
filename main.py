import os
import time
import argparse

import cv2

from camera import capture_frame
from detect import detect_top_faces
from align import align_face
from ascii_gen import image_to_ascii, ascii_to_color_image


def main(ascii_width=300, brightness_beta=30, char_ratio=1.0):

    os.makedirs("gallery", exist_ok=True)

    frame = capture_frame(brightness_beta=brightness_beta)

    if frame is None:
        return

    faces = detect_top_faces(frame)

    if not faces:
        print("No faces detected")
        return

    print(f"{len(faces)} faces detected")


    for i, face in enumerate(faces):

        aligned = align_face(face)
        aligned = cv2.resize(aligned, None, fx=1.2, fy=1.2)

        ascii_lines = image_to_ascii(aligned, width=ascii_width, char_ratio=char_ratio)

        filename = f"gallery/ascii_face_{int(time.time())}_{i}.png"

        ascii_to_color_image(aligned, ascii_lines, filename)

        print("Saved:", filename)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ascii-width", type=int, default=100,
                   help="Width (in characters) for generated ASCII art. Larger = higher resolution")
    p.add_argument("--char-ratio", type=float, default=1.0,
                   help="Character width/height ratio correction. >1 makes output taller.")
    p.add_argument("--brightness", type=int, default=50,
                   help="Brightness beta to add to captured frames (passed to OpenCV convertScaleAbs).")
    args = p.parse_args()

    main(ascii_width=args.ascii_width, brightness_beta=args.brightness, char_ratio=args.char_ratio)