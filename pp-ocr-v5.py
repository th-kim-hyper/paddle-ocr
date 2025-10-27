"""
PP-OCRv5 기반 OCR 실행 스크립트 (이미지 및 PDF 지원)
- 요구: paddleocr 3.x 설치
- 사용 예:
    python pp-ocr-v5.py --input path/to/file.jpg --out outdir
    python pp-ocr-v5.py --input path/to/file.pdf --out outdir --page_num 10
"""

import os
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# PaddleOCR import (설치된 paddleocr 패키지를 사용)
from ocr import PaddleOCR

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("경고: PyMuPDF가 설치되지 않았습니다. PDF 시각화가 생성되지 않습니다.")
    print("설치: pip install pymupdf")

def parse_args():
    parser = argparse.ArgumentParser(description="PP-OCRv5 OCR script for images and PDFs")
    parser.add_argument("--input", default="./images/3.jpg", help="이미지/PDF 파일 또는 폴더 경로")
    parser.add_argument("--out", default="./output", help="출력 디렉토리")
    parser.add_argument("--use_cpu", action="store_true", help="CPU 사용 (기본값: GPU 사용)")
    parser.add_argument("--page_num", type=int, default=None, help="PDF의 경우 처리할 페이지 수 (기본: 모든 페이지)")
    parser.add_argument("--det_db_thresh", type=float, default=0.3, help="텍스트 감지 픽셀 임계값 (낮을수록 더 많이 감지)")
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6, help="박스 신뢰도 임계값 (높을수록 정확)")
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5, help="박스 확장 비율 (높을수록 박스가 넓어짐)")
    parser.add_argument("--max_side_len", type=int, default=960, help="이미지 리사이즈 최대 변 길이")
    return parser.parse_args()

def polygon_to_rect(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def get_korean_font(size=20):
    """한국어 폰트 로드"""
    font_path = Path(__file__).parent / "fonts" / "NanumGothic.ttf"
    try:
        return ImageFont.truetype(str(font_path), size)
    except Exception as e:
        print(f"폰트 로드 실패: {e}, 기본 폰트 사용")
        return ImageFont.load_default()

def draw_text_on_image(img, text, position, font_size=20, color=(0, 0, 255)):
    """PIL을 사용하여 한국어 텍스트를 이미지에 그리기"""
    # BGR to RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_korean_font(font_size)
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB
    
    # RGB to BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def evaluate_box_accuracy(page_data, img_width, img_height):
    """
    Box 위치의 정확도를 평가
    - 이미지 경계 내 비율
    - Box 크기 분포
    - Box 겹침 정도
    """
    entries = page_data['entries']
    if not entries:
        return {
            "total_boxes": 0,
            "boxes_in_bounds": 0,
            "in_bounds_ratio": 0.0,
            "avg_box_area": 0,
            "avg_box_width": 0,
            "avg_box_height": 0,
            "boxes_overlapping": 0,
            "overlap_ratio": 0.0
        }
    
    boxes_in_bounds = 0
    total_area = 0
    total_width = 0
    total_height = 0
    boxes_overlapping = 0
    
    rects = []
    
    for entry in entries:
        rect = entry['rect']  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = rect
        
        # 경계 내 여부 확인
        if 0 <= xmin < img_width and 0 <= ymin < img_height and \
           0 <= xmax <= img_width and 0 <= ymax <= img_height:
            boxes_in_bounds += 1
        
        # Box 크기 계산
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        
        total_area += area
        total_width += width
        total_height += height
        
        rects.append((xmin, ymin, xmax, ymax))
    
    # Box 겹침 확인
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            x1_min, y1_min, x1_max, y1_max = rects[i]
            x2_min, y2_min, x2_max, y2_max = rects[j]
            
            # 겹침 여부 확인
            if not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min):
                boxes_overlapping += 1
                break  # 하나라도 겹치면 카운트
    
    num_boxes = len(entries)
    
    return {
        "total_boxes": num_boxes,
        "boxes_in_bounds": boxes_in_bounds,
        "in_bounds_ratio": boxes_in_bounds / num_boxes if num_boxes > 0 else 0.0,
        "avg_box_area": total_area / num_boxes if num_boxes > 0 else 0,
        "avg_box_width": total_width / num_boxes if num_boxes > 0 else 0,
        "avg_box_height": total_height / num_boxes if num_boxes > 0 else 0,
        "boxes_overlapping": boxes_overlapping,
        "overlap_ratio": boxes_overlapping / num_boxes if num_boxes > 0 else 0.0
    }

def is_image_file(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

def is_pdf_file(p: Path):
    return p.suffix.lower() == ".pdf"

def pdf_to_images(pdf_path: Path):
    """
    PDF 파일의 각 페이지를 고해상도 이미지로 변환 (box 정확도 향상)
    Returns: list of numpy arrays (BGR format for OpenCV)
    """
    if not HAS_PYMUPDF:
        return []
    
    imgs = []
    with fitz.open(str(pdf_path)) as pdf:
        for pg in range(pdf.page_count):
            page = pdf[pg]
            # PDF를 3배 해상도로 렌더링하여 텍스트 box 정확도 향상
            mat = fitz.Matrix(3, 3)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            
            # 너비 또는 높이가 3000픽셀 이상이면 2배 해상도로 조정
            if pm.width > 3000 or pm.height > 3000:
                pm = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            
            # 매우 큰 페이지는 1배 해상도로 렌더링
            if pm.width > 4500 or pm.height > 4500:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)
    
    return imgs

def process_file(file_path: Path, ocr, out_dir: Path):
    """
    이미지 또는 PDF 파일을 OCR 처리하고 결과를 반환
    Args:
        file_path: 처리할 파일 경로
        ocr: PaddleOCR 인스턴스
        out_dir: 출력 디렉토리
    """
    file_stem = file_path.stem
    file_name = file_path.name
    
    # 중간 이미지 저장 (PDF 및 일반 이미지 모두)
    converted_images = []
    converted_dir = out_dir / file_stem / "converted_images"
    ensure_dir(converted_dir)
    
    if is_pdf_file(file_path) and HAS_PYMUPDF:
        # PDF를 이미지로 변환하고 저장 (PNG 무손실 포맷)
        print(f"  PDF 페이지를 이미지로 변환 중...")
        pdf_images = pdf_to_images(file_path)
        
        for page_idx, img in enumerate(pdf_images):
            img_path = converted_dir / f"page_{page_idx:03d}.png"
            # PNG 압축 레벨 3 (기본값, 무손실)
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            converted_images.append(str(img_path))
        
        print(f"  {len(converted_images)}개 페이지 이미지 저장 완료")
    
    elif is_image_file(file_path):
        # 일반 이미지는 원본을 중간 이미지로 복사 (PNG 무손실 포맷)
        print(f"  원본 이미지를 중간 이미지로 저장 중...")
        img = cv2.imread(str(file_path))
        if img is not None:
            img_path = converted_dir / f"page_000.png"
            # PNG 압축 레벨 3 (기본값, 무손실)
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            converted_images.append(str(img_path))
            print(f"  중간 이미지 저장 완료: {img.shape[1]}x{img.shape[0]}")
    
    # OCR 수행 - 저장된 중간 이미지 사용
    if converted_images:
        print(f"  저장된 중간 이미지로 OCR 수행 중...")
        result = []
        for img_path in converted_images:
            # 이미지 크기 확인
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                print(f"    OCR 입력 이미지 크기: {w}x{h} - {img_path}")
            
            page_result = ocr.predict(img_path)
            result.extend(page_result)
    else:
        # fallback: 중간 이미지가 없으면 원본 사용
        result = ocr.predict(str(file_path))
    
    # PDF의 경우 여러 페이지, 이미지의 경우 1페이지
    all_pages_data = []
    
    for page_idx, res in enumerate(result):
        entries = []
        
        # res는 OCRResult 객체 (dict-like)
        if 'dt_polys' in res and 'rec_texts' in res and 'rec_scores' in res:
            dt_polys = res['dt_polys']
            rec_texts = res['rec_texts']
            rec_scores = res['rec_scores']
            
            for box, text, score in zip(dt_polys, rec_texts, rec_scores):
                # box는 numpy array [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # 좌표를 정수로 반올림하여 정확도 향상
                box_rounded = np.round(box).astype(np.int32)
                xmin, ymin, xmax, ymax = polygon_to_rect(box_rounded)
                
                entries.append({
                    "box": [[float(p[0]), float(p[1])] for p in box_rounded],
                    "rect": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "text": text,
                    "score": float(score)
                })
        
        all_pages_data.append({
            "page": page_idx,
            "num_boxes": len(entries),
            "entries": entries
        })
    
    # 결과 저장
    out_base = out_dir / file_stem
    ensure_dir(out_base)
    
    # Box 정확도 평가 추가
    accuracy_results = []
    
    if is_image_file(file_path):
        # 이미지인 경우
        img = cv2.imread(str(file_path))
        if img is not None and len(all_pages_data) > 0:
            img_height, img_width = img.shape[:2]
            accuracy = evaluate_box_accuracy(all_pages_data[0], img_width, img_height)
            accuracy_results.append(accuracy)
    elif converted_images:
        # PDF인 경우 각 페이지별로 평가
        for page_idx, img_path in enumerate(converted_images):
            img = cv2.imread(img_path)
            if img is not None and page_idx < len(all_pages_data):
                img_height, img_width = img.shape[:2]
                accuracy = evaluate_box_accuracy(all_pages_data[page_idx], img_width, img_height)
                accuracy_results.append(accuracy)
    
    # JSON 저장 (모든 페이지 + 정확도 평가)
    output_data = {
        "pages": all_pages_data,
        "accuracy_evaluation": accuracy_results
    }
    json_path = out_base / f"{file_stem}_ocr.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(output_data, jf, ensure_ascii=False, indent=2)
    
    # 시각화 생성 - 모두 저장된 중간 이미지 사용
    vis_paths = []
    
    if converted_images and len(converted_images) == len(all_pages_data):
        print(f"  시각화 생성 중...")
        
        for page_idx, (img_path, page_data) in enumerate(zip(converted_images, all_pages_data)):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            vis_img = img.copy()
            
            for entry in page_data['entries']:
                box = entry['box']
                text = entry['text']
                score = entry['score']
                xmin = entry['rect'][0]
                ymin = entry['rect'][1]
                
                # 시각화: box 그리기 (두꺼운 선으로 명확하게)
                pts = np.array(box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                
                # 한국어 폰트를 사용하여 텍스트 그리기
                label_text = f"{text[:30]} ({score:.2f})"
                font_size = 15 if is_image_file(file_path) else 12
                vis_img = draw_text_on_image(vis_img, label_text, (int(xmin), int(max(0, ymin - 25))), 
                                            font_size=font_size, color=(0, 0, 255))
            
            # 파일명 생성 (PNG 무손실 포맷)
            if is_image_file(file_path):
                vis_path = out_base / f"{file_stem}_vis.png"
            else:
                vis_path = out_base / f"{file_stem}_page_{page_idx}_vis.png"
            
            # PNG로 저장 (무손실)
            cv2.imwrite(str(vis_path), vis_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            vis_paths.append(str(vis_path))
        
        print(f"  시각화 생성 완료: {len(vis_paths)}개 페이지")
    
    total_boxes = sum(page['num_boxes'] for page in all_pages_data)
    
    # 정확도 평가 요약 계산
    if accuracy_results:
        avg_in_bounds = sum(a['in_bounds_ratio'] for a in accuracy_results) / len(accuracy_results)
        avg_overlap = sum(a['overlap_ratio'] for a in accuracy_results) / len(accuracy_results)
        accuracy_summary = {
            "avg_in_bounds_ratio": round(avg_in_bounds * 100, 2),
            "avg_overlap_ratio": round(avg_overlap * 100, 2),
            "total_boxes_evaluated": sum(a['total_boxes'] for a in accuracy_results)
        }
    else:
        accuracy_summary = None
    
    result_data = {
        "name": file_name,
        "type": "pdf" if is_pdf_file(file_path) else "image",
        "pages": len(all_pages_data),
        "json": str(json_path),
        "visualizations": vis_paths,
        "total_boxes": total_boxes,
        "accuracy_summary": accuracy_summary
    }
    
    # PDF인 경우 변환된 이미지 경로 추가
    if converted_images:
        result_data["converted_images"] = converted_images
    
    return result_data

def main():
    args = parse_args()
    src = Path(args.input)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # 모델 초기화 (기본값: GPU 사용)
    device = "cpu" if args.use_cpu else "gpu"
    
    # PaddleOCR 초기화 (PP-OCRv5) - 한국어 지원, box 좌표 정확도 최적화
    print(f"PP-OCRv5 모델 초기화 중... (device: {device}, lang: korean)")
    print(f"  감지 파라미터: thresh={args.det_db_thresh}, box_thresh={args.det_db_box_thresh}, unclip={args.det_db_unclip_ratio}")
    print(f"  이미지 처리: det_limit_side_len={args.max_side_len} (이미지 리사이징 최소화로 좌표 정확도 향상)")
    print(f"  모델: PP-OCRv5_server (한국어 포함 다국어 고정밀도 모델)")
    ocr = PaddleOCR(
        lang='korean',  # 한국어 모델 사용
        device=device,
        ocr_version='PP-OCRv5',  # PP-OCRv5 명시적 지정
        text_detection_model_name='PP-OCRv5_server_det',  # 서버용 고정밀도 감지 모델 (한국어 포함 다국어 지원)
        text_recognition_model_name='PP-OCRv5_server_rec',  # 서버용 고정밀도 인식 모델
        det_db_thresh=args.det_db_thresh,
        det_db_box_thresh=args.det_db_box_thresh,
        det_db_unclip_ratio=args.det_db_unclip_ratio,
        det_limit_side_len=args.max_side_len,  # 감지 시 최대 변 길이 (큰 값으로 리사이징 최소화)
        det_limit_type='max',  # 최대 변 제한 방식
        use_angle_cls=False,  # 각도 분류 비활성화로 좌표 정확도 향상
    )

    # 파일 목록 생성
    file_paths = []
    if src.is_dir():
        for f in sorted(src.iterdir()):
            if is_image_file(f) or is_pdf_file(f):
                file_paths.append(f)
    elif src.is_file():
        if is_image_file(src) or is_pdf_file(src):
            file_paths = [src]
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {src}")
    else:
        raise FileNotFoundError(f"파일/폴더를 찾을 수 없음: {src}")

    print(f"처리할 파일 수: {len(file_paths)}")
    
    summary = {}
    for file_path in file_paths:
        print(f"\n처리 중: {file_path.name}")
        
        try:
            result_info = process_file(file_path, ocr, out_dir)
            summary[result_info['name']] = result_info
            
            # 정확도 정보 출력
            acc_summary = result_info.get('accuracy_summary')
            if acc_summary:
                print(f"  완료: {result_info['pages']} 페이지, {result_info['total_boxes']} 박스")
                print(f"  정확도: 경계내 {acc_summary['avg_in_bounds_ratio']}%, 겹침 {acc_summary['avg_overlap_ratio']}%")
            else:
                print(f"  완료: {result_info['pages']} 페이지, {result_info['total_boxes']} 박스")
        except Exception as e:
            print(f"  오류: {e}")
            summary[file_path.name] = {
                "name": file_path.name,
                "error": str(e)
            }

    # 요약 저장
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"\n완료. 출력: {out_dir}")
    print(f"요약 파일: {summary_path}")
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"완료. 출력: {out_dir}")

if __name__ == "__main__":
    main()