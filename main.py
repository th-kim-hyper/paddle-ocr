import argparse
from pathlib import Path
from PIL import Image
import ppocr

def parse_args():
    p = argparse.ArgumentParser(description="Run PaddleOCR with selectable OCR version and models")
    p.add_argument("--img", default="./images/ccf2w.png", help="Input image path")
    p.add_argument("--out", default=".output", help="Output directory")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Device to run on")
    p.add_argument("--ocr_version", default="PP-OCRv5", help="OCR version to use (e.g. PP-OCRv5)")
    return p.parse_args()

def main():
    args = parse_args()
    lang = "korean"
    ocr_version = args.ocr_version
    device = args.device
    img_path = args.img
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(img_path)
    targets = []
        
    ocr = ppocr.get_ocr_instance(
        lang=lang,
        ocr_version=ocr_version,
        device=device,
    )

    if input_path.suffix.lower() == ".pdf":
        print(f"Detected PDF input. Converting pages to images...")
        targets = ppocr.pdf_to_images(str(input_path), out_dir, dpi=200)
    else:
        targets = [str(input_path)]

    for page_idx, target in enumerate(targets):
        results = ocr.predict(target)
        for i, res in enumerate(results):
            res_dict = dict(res)
            res.print()
            # save default visualizations and json (names include page index)
            base_stem = Path(target).stem
            res.save_to_img(str(out_dir))
            # save_to_json uses input path to name file; we also write a page-specific json
            res.save_to_json(str(out_dir))

            # save an additional visualization using Korean font
            vis_out = out_dir / f"{base_stem}_page{page_idx:03d}_ocr_kor_{i}.png"
            font = ppocr.get_korean_font(max(16, Image.open(target).size[0] // 80))
            ppocr.draw_korean_visualization(target, res_dict, str(vis_out), font=font)
            print(f"Saved Korean visualized image: {vis_out}")

if __name__ == "__main__":
    main()