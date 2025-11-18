import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

def draw_boxes_from_json(image_path: str, json_path: str, output_path: str):
    """
    JSON 결과 파일에서 rec_boxes 정보를 읽어 원본 이미지에 박스를 그립니다.
    
    Args:
        image_path: 원본 이미지 경로
        json_path: PaddleOCR 결과 JSON 파일 경로
        output_path: 저장할 시각화 이미지 경로
    """
    # JSON 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rec_boxes = data.get('rec_boxes', [])
    rec_texts = data.get('rec_texts', [])
    rec_scores = data.get('rec_scores', [])
    
    # 이미지 열기
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 폰트 설정 (기본 폰트 사용)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # 박스 그리기
    for idx, box in enumerate(rec_boxes):
        xmin, ymin, xmax, ymax = box
        
        # 사각형 그리기 (초록색, 두께 2)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=2)
        
        # 점수 표시 (소수점 3자리)
        if idx < len(rec_scores):
            score = rec_scores[idx]
            label = f"{score:.3f}"
            
            # 텍스트 크기 계산
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 타이틀 박스 위치 (박스 상단)
            title_x = xmin
            title_y = max(0, ymin - text_height - 8)  # 상단 여백 포함
            title_box = [
                title_x,
                title_y,
                title_x + text_width + 8,
                title_y + text_height + 4
            ]
            
            # 타이틀 박스 배경 (흰색)
            draw.rectangle(title_box, fill=(255, 255, 255), outline=(0, 255, 0), width=1)
            
            # 점수 텍스트 (빨간색)
            draw.text((title_x + 4, title_y + 2), label, fill=(255, 0, 0), font=font)
    
    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"✓ 시각화 이미지 저장: {output_path}")

if __name__ == "__main__":
    image_path = "images/3bnyf.png"

    # PaddleOCR 인스턴스 초기화
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # 샘플 이미지에 대해 OCR 추론 실행
    result = ocr.predict(
        input=image_path)

    # 결과 시각화 및 JSON 결과 저장
    for res in result:
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")
    
    # JSON 결과를 사용해 커스텀 시각화 생성
    json_path = "output/3bnyf_res.json"
    vis_output = "output/3bnyf_vis.jpg"
    
    if Path(json_path).exists():
        draw_boxes_from_json(image_path, json_path, vis_output)
    else:
        print(f"⚠️ JSON 파일을 찾을 수 없습니다: {json_path}")

