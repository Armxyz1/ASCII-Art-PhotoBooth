import cv2

ASCII_CHARS = (
"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/"
"|()1{}[]?-_+~<>i!lI;:,\"^`'. "
)


# Simple fixed-size charset good for clearer, uniform ASCII
SIMPLE_CHARS = ".,:;i1tfLCG08@"


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


def simple_image_to_ascii(img, cols=80, charset=SIMPLE_CHARS):
    """Convert image to ASCII using equal-sized character cells.

    This produces simpler ASCII art where every character maps to a fixed cell
    and therefore all characters have the same rendered size.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    cell_w = max(1, w // cols)
    cols = w // cell_w
    rows = max(1, h // cell_w)

    ascii_lines = []
    for r in range(rows):
        y1 = r * cell_w
        y2 = min(h, (r + 1) * cell_w)
        line = ""
        for c in range(cols):
            x1 = c * cell_w
            x2 = min(w, (c + 1) * cell_w)
            cell = gray[y1:y2, x1:x2]
            if cell.size == 0:
                avg = 0
            else:
                avg = int(cell.mean())

            idx = int((avg / 255) * (len(charset) - 1))
            line += charset[idx]
        ascii_lines.append(line)

    return ascii_lines


def simple_ascii_to_image(original_img, ascii_lines, out_path, font_path=None, font_size=12, char_spacing=1.0):
    """Render simple ASCII lines to an RGB image using fixed-size character cells.

    If `font_path` is provided it will be used; otherwise `ImageFont.load_default()` is used.
    """
    from PIL import Image, ImageDraw, ImageFont
    import cv2

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    bbox = font.getbbox("A")
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    width = int(char_w * len(ascii_lines[0]) * char_spacing)
    height = char_h * len(ascii_lines)

    canvas = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(canvas)

    # Resize color source to match ascii grid so colors map one-to-one
    small_color = cv2.resize(original_img, (len(ascii_lines[0]), len(ascii_lines)), interpolation=cv2.INTER_NEAREST)

    for y, line in enumerate(ascii_lines):
        for x, char in enumerate(line):
            b, g, r = small_color[y, x]
            draw.text((int(x * char_w * char_spacing), int(y * char_h)), char, fill=(int(r), int(g), int(b)), font=font)

    canvas.save(out_path)
    return out_path


def ascii_to_color_image(original_img, ascii_lines, out_path, char_spacing=1.0):

    from PIL import Image, ImageDraw, ImageFont
    import cv2

    font = ImageFont.load_default()

    # Pillow >=10 fix
    bbox = font.getbbox("A")
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    width = int(char_w * len(ascii_lines[0]) * char_spacing)
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
                (int(x * char_w * char_spacing), int(y * char_h)),
                char,
                fill=(int(r), int(g), int(b)),
                font=font
            )

    canvas.save(out_path)

    return out_path