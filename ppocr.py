from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import fitz

def get_ocr_instance(lang: str = "korean", ocr_version: str = "PP-OCRv5", device: str = "gpu"):
    return PaddleOCR(
        lang=lang,
        ocr_version=ocr_version,
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

def pdf_to_images(pdf_path: str, out_dir: Path, dpi: int = 200):
    """Render PDF pages to PNG images saved under out_dir/converted_images.

    Returns a list of file paths to the rendered images.
    """
    conv_dir = out_dir / "converted_images"
    conv_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_paths = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        p = conv_dir / f"page_{i:03d}.png"
        pix.save(str(p))
        img_paths.append(str(p))
    doc.close()
    return img_paths

def get_korean_font(size: int):
    # prefer bundled NanumGothic.ttf in fonts/; fallback to default PIL font
    font_paths = [Path("./fonts/NanumGothic.ttf"), Path("./fonts/NanumGothic-Regular.ttf")]
    for p in font_paths:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()

def draw_korean_visualization(image_path: str, ocr_result: dict, out_path: str, font=None):
    # use RGBA overlay so we can draw semi-transparent backgrounds
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    # dynamic font size if not provided
    if font is None:
        font = get_korean_font(max(16, w // 80))

    rec_polys = ocr_result.get("rec_polys", [])
    rec_texts = ocr_result.get("rec_texts", [])

    def _get_text_size(draw_obj, txt, fnt):
        # Try modern textbbox, then fallback to textsize or font.getsize
        try:
            bbox = draw_obj.textbbox((0, 0), txt, font=fnt)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            try:
                return draw_obj.textsize(txt, font=fnt)
            except Exception:
                try:
                    return fnt.getsize(txt)
                except Exception:
                    return (max(8, len(txt) * 6), 12)

    for poly, text in zip(rec_polys, rec_texts):
        try:
            pts = np.array(poly).reshape(-1, 2)
            # draw polygon border on overlay (opaque)
            draw.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=(0, 255, 0, 255), width=2)
            # determine text position (top-left of polygon)
            tx = int(pts[:, 0].min())
            ty = int(pts[:, 1].min()) - 4
            if ty < 0:
                ty = int(pts[:, 1].max()) + 4
            # draw semi-transparent background rectangle for readability
            text_size = _get_text_size(draw, text, font)
            rect = [tx, ty, tx + text_size[0] + 4, ty + text_size[1] + 4]
            # semi-transparent black (alpha=180)
            draw.rectangle(rect, fill=(0, 0, 0, 180))
            # draw text opaque white
            draw.text((tx + 2, ty + 2), text, font=font, fill=(255, 255, 255, 255))
        except Exception:
            continue

    # composite overlay onto base and save as RGB
    try:
        combined = Image.alpha_composite(base, overlay).convert("RGB")
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        combined.save(str(out_p))
    except Exception:
        # fallback: try saving overlay-composited image directly
        Image.alpha_composite(base, overlay).convert("RGB").save(out_path)
