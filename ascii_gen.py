import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASCII_CHARS = (
"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/"
"|()1{}[]?-_+~<>i!lI;:,\"^`'. "
)


def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Edge detection
    edges = cv2.Canny(gray, 80, 150)

    # Blend edges with grayscale
    combined = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)

    return combined



def image_to_ascii(img, width=140, char_ratio=1.0):

    """Convert image to ASCII lines.

    Args:
        img: input BGR image (numpy array).
        width: number of characters per row.
        char_ratio: correction factor for character aspect ratio (char_width/char_height).
                    Increase this value to make the ASCII output taller (fixes overly-wide images).
    """

    processed = preprocess(img)

    h, w = processed.shape
    aspect = h / w

    # `char_ratio` adjusts for character cell aspect (char_width / char_height).
    # Default 1.0 yields rows = width * image_aspect; tweak if your font is non-square.
    new_h = max(1, int(width * aspect * char_ratio))

    small = cv2.resize(processed, (width, new_h))

    ascii_lines = []

    for row in small:

        line = ""

        for pixel in row:

            idx = int(pixel / 255 * (len(ASCII_CHARS) - 1))
            line += ASCII_CHARS[idx]

        ascii_lines.append(line)

    return ascii_lines


def ascii_to_color_image(original_img, ascii_lines, out_path):

    from PIL import Image, ImageDraw, ImageFont
    import cv2

    font = ImageFont.load_default()

    # Pillow >=10 fix
    bbox = font.getbbox("A")
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    width = char_w * len(ascii_lines[0])
    height = char_h * len(ascii_lines)

    canvas = Image.new("RGB", (width, height), "black")

    draw = ImageDraw.Draw(canvas)

    small_color = cv2.resize(
        original_img,
        (len(ascii_lines[0]), len(ascii_lines))
    )

    for y, line in enumerate(ascii_lines):

        for x, char in enumerate(line):

            b, g, r = small_color[y, x]

            draw.text(
                (x * char_w, y * char_h),
                char,
                fill=(int(r), int(g), int(b)),
                font=font
            )

    canvas.save(out_path)

    return out_path