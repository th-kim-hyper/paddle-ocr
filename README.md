# paddle-ocr

PaddleOCR v5 기반 다국어 OCR 프로젝트 - PDF 문서를 페이지별 이미지로 변환하고 OCR 인식 후 시각화

## 개요
- Framework: PaddleOCR 3.x
- 모델: PP-OCRv5 (80+ 다국어 지원)
- **지원 언어**: 한국어, 중국어, 영어, 일본어, 프랑스어, 독일어, 아랍어, 러시아어, 스페인어, 포르투갈어 등 80+ 언어
- PDF 처리: pdf2image 라이브러리 사용
- 지원 형식: 이미지 (JPG, PNG, BMP, TIFF) 및 PDF
- 주요 기능:
  - 다국어 텍스트 인식 (언어별 최적화된 모델)
  - PDF를 고해상도 이미지로 페이지별 변환 (DPI 300)
  - PaddleOCR을 사용한 텍스트 검출 및 인식
  - 언어별 최적화된 폰트로 시각화
  - JSON 형식의 구조화된 결과 출력

## 필수 요구사항

### 시스템 요구사항
PDF 처리를 위해 **poppler-utils**가 설치되어 있어야 합니다:

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Arch Linux:**
```bash
sudo pacman -S poppler
```

### Python 패키지
```bash
pip install paddleocr>=2.7.0
pip install pdf2image>=1.16.0
pip install opencv-python>=4.8.0
pip install pillow>=10.0.0
pip install numpy>=1.24.0
```

또는 프로젝트 의존성 전체 설치:
```bash
pip install -e .
```

## 빠른 시작

1) (권장) 가상환경 생성

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) 의존성 설치

```bash
pip install -e .
```

3) OCR 실행

**단일 이미지 처리:**
```bash
python pp-ocr-v5.py --input ./images/sample.png --out ./output
```

**단일 PDF 처리 (페이지별 이미지 변환 후 OCR):**
```bash
python pp-ocr-v5.py --input ./images/document.pdf --out ./output
```

**폴더 내 모든 파일 처리:**
```bash
python pp-ocr-v5.py --input ./images --out ./output
```

## 사용법

### 기본 명령어

```bash
# 한국어 문서 처리 (기본값)
python pp-ocr-v5.py --input document.pdf --out ./output --lang korean

# 영어 문서 처리
python pp-ocr-v5.py --input document.pdf --out ./output --lang en

# 일본어 문서 처리
python pp-ocr-v5.py --input document.pdf --out ./output --lang japan

# 중국어 문서 처리
python pp-ocr-v5.py --input document.pdf --out ./output --lang ch

# CPU 사용
python pp-ocr-v5.py --input document.pdf --out ./output --lang korean --use_cpu
```

### 지원 언어 목록

PaddleOCR은 80개 이상의 언어를 지원합니다. 주요 언어:

- **한국어**: `--lang korean`
- **영어**: `--lang en`
- **중국어 (간체)**: `--lang ch`
- **중국어 (번체)**: `--lang chinese_cht`
- **일본어**: `--lang japan`
- **프랑스어**: `--lang french`
- **독일어**: `--lang german`
- **스페인어**: `--lang spanish`
- **포르투갈어**: `--lang portuguese`
- **아랍어**: `--lang arabic`
- **러시아어**: `--lang russian`
- **이탈리아어**: `--lang italian`
- **힌디어**: `--lang hindi`
- **태국어**: `--lang thai`
- **베트남어**: `--lang vietnamese`

전체 지원 언어 목록: [PaddleOCR 다국어 지원](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md)

### 명령행 옵션

**공통 옵션:**
- `--input`: 처리할 이미지/PDF 파일 또는 폴더 경로 (필수)
- `--out`: 결과 출력 디렉토리 (기본값: `./output`)
- `--lang`: 언어 선택 (기본값: `korean`)
- `--use_cpu`: CPU 사용 (기본값: GPU 사용)
- `--page_num`: PDF의 경우 처리할 페이지 수 (현재 미구현, 모든 페이지 처리)

**OCR 파라미터 옵션:**
- `--det_db_thresh`: 텍스트 감지 픽셀 임계값 (기본값: 0.3, 낮을수록 더 많이 감지)
- `--det_db_box_thresh`: 박스 신뢰도 임계값 (기본값: 0.6, 높을수록 정확)
- `--det_db_unclip_ratio`: 박스 확장 비율 (기본값: 1.5, 낮을수록 텍스트에 밀착)

## 출력 형식

### 디렉토리 구조
```
output/
├── document/
│   ├── converted_images/              # PDF 변환 중간 이미지 (PDF만 해당)
│   │   ├── document_page_0_converted.jpg
│   │   ├── document_page_1_converted.jpg
│   │   └── ...
│   ├── document_ocr.json              # OCR 결과 (모든 페이지)
│   ├── document_page_0_vis.jpg        # 페이지 0 시각화 (OCR 결과 표시)
│   ├── document_page_1_vis.jpg        # 페이지 1 시각화 (OCR 결과 표시)
│   └── ...
└── summary.json                        # 전체 처리 요약
```

**파일 설명:**
- **converted_images/**: PDF를 이미지로 변환한 원본 파일 (DPI 300, JPEG 형식)
  - PDF 문서만 생성됨
  - OCR 처리 전 중간 단계 이미지
  - 재처리 시 활용 가능
- **xxx_ocr.json**: 구조화된 OCR 결과 (박스 좌표, 텍스트, 신뢰도)
- **xxx_page_X_vis.jpg**: OCR 결과를 시각화한 이미지 (박스와 텍스트 표시)

### JSON 출력 형식

**document_ocr.json:**
```json
[
  {
    "page": 0,
    "num_boxes": 25,
    "entries": [
      {
        "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "rect": [xmin, ymin, xmax, ymax],
        "text": "인식된 텍스트",
        "score": 0.95
      },
      ...
    ]
  },
  ...
]
```

**summary.json:**
```json
{
  "document.pdf": {
    "name": "document.pdf",
    "type": "pdf",
    "pages": 5,
    "json": "/path/to/output/document/document_ocr.json",
    "visualizations": [
      "/path/to/output/document/document_page_0_vis.jpg",
      ...
    ],
    "total_boxes": 125
  }
}
```

## PDF 처리 방식

1. **PDF → 이미지 변환**: pdf2image를 사용하여 각 페이지를 DPI 300의 고해상도 이미지로 변환
   - 변환된 이미지는 `converted_images/` 폴더에 저장됨
   - 파일명: `{문서명}_page_{번호}_converted.jpg`
   - 재처리 또는 추가 분석에 활용 가능
2. **OCR 처리**: 변환된 이미지를 PaddleOCR로 처리하여 텍스트 검출 및 인식
3. **시각화**: 검출된 박스와 텍스트를 원본 이미지에 오버레이하여 시각화
4. **결과 저장**: JSON 형식으로 구조화된 데이터 저장 및 시각화 이미지 생성

### 중간 이미지 활용 예시

```bash
# PDF를 먼저 처리
python pp-ocr-v5.py --input document.pdf --lang korean --out ./output

# converted_images 폴더의 이미지를 확인
ls -lh output/document/converted_images/

# 특정 페이지만 다시 처리하고 싶을 때 변환된 이미지 사용
python pp-ocr-v5.py --input output/document/converted_images/document_page_0_converted.jpg \
    --lang en --out ./output_page0_en
```

## 사용 예시 및 결과

### 한국어 문서 처리 예시

```bash
$ python pp-ocr-v5.py --input korean_doc.pdf --out ./output --lang korean --use_cpu

PP-OCRv5 모델 초기화 중... (device: cpu, lang: 한국어)
✓ 폰트: NanumGothic.ttf
처리할 파일 수: 1

처리 중: korean_doc.pdf
  PDF 페이지를 이미지로 변환 중...
  ✓ 3 페이지 변환 완료
  OCR 처리 중...
  결과 처리 및 시각화 생성 중...
    페이지 0: 42 박스 검출, 시각화 저장됨
    페이지 1: 38 박스 검출, 시각화 저장됨
    페이지 2: 35 박스 검출, 시각화 저장됨
  ✓ 완료: 3 페이지, 115 박스
```

### 다국어 문서 처리

```bash
# 일본어 문서
python pp-ocr-v5.py --input japanese_doc.pdf --lang japan --out ./output

# 프랑스어 문서
python pp-ocr-v5.py --input french_doc.pdf --lang french --out ./output

# 아랍어 문서 (오른쪽에서 왼쪽)
python pp-ocr-v5.py --input arabic_doc.pdf --lang arabic --out ./output

# 혼합 언어 문서 (중국어+영어)
python pp-ocr-v5.py --input mixed_doc.pdf --lang ch --out ./output
```

### 성공 예시

```bash
$ python pp-ocr-v5.py --input ./images/a.pdf --out ./output --use_cpu

PP-OCRv5 모델 초기화 중... (device: cpu)
처리할 파일 수: 1

처리 중: a.pdf
  PDF 페이지를 이미지로 변환 중...
  ✓ 2 페이지 변환 완료
  OCR 처리 중...
  결과 처리 및 시각화 생성 중...
    페이지 0: 35 박스 검출, 시각화 저장됨
    페이지 1: 30 박스 검출, 시각화 저장됨
  ✓ 완료: 2 페이지, 65 박스

============================================================
처리 완료!
출력 디렉토리: output
요약 파일: output/summary.json
============================================================
```

### 출력 파일 확인

```bash
$ ls -lh output/a/
-rw-rw-r-- 1 user user  28K a_ocr.json
-rw-rw-r-- 1 user user 942K a_page_0_vis.jpg
-rw-rw-r-- 1 user user 325K a_page_1_vis.jpg
```

## PaddleOCR v3 API 변경사항

이 프로젝트는 PaddleOCR v3 API를 사용합니다. 주요 변경사항:

- `use_angle_cls` → `use_textline_orientation`
- `use_gpu` → `device` ('cpu' 또는 'gpu')
- `ocr.ocr()` → `ocr.predict()`
- 결과 구조: 딕셔너리 형태 (`dt_polys`, `rec_texts`, `rec_scores`)

## 문제 해결

### PDF 변환 오류
```
경고: pdf2image가 설치되지 않았습니다. PDF 처리가 불가능합니다.
```
해결: `pip install pdf2image` 및 poppler-utils 설치

### 텍스트 검출 안 됨
- `--det_db_thresh` 값을 낮춰보세요 (예: 0.2)
- `--det_db_unclip_ratio` 값을 높여보세요 (예: 1.8)

### 언어별 인식률 향상
- 올바른 언어 코드 사용: `--lang korean`, `--lang en` 등
- 해당 언어의 시스템 폰트 설치 (MULTILINGUAL.md 참조)
- 언어별 최적화된 파라미터 사용 (MULTILINGUAL.md 참조)

### 폰트 문제
```
⚠️ 한국어 시스템 폰트를 찾을 수 없습니다.
```
해결:
```bash
# Ubuntu/Debian
sudo apt-get install fonts-nanum

# macOS
brew install font-nanum
```

## 추가 문서

- **MULTILINGUAL.md**: 다국어 지원 상세 가이드 (80+ 언어 목록, 언어별 설정)
- **INSTALL.md**: 설치 가이드
- **CHANGES.md**: 변경사항 및 업데이트 내역

## 주요 기능

✅ **80+ 언어 지원**: 한국어, 영어, 중국어, 일본어, 프랑스어, 아랍어 등  
✅ **PDF 페이지별 처리**: 고해상도 이미지 변환 (DPI 300)  
✅ **자동 폰트 선택**: 언어별 최적화된 폰트 자동 감지  
✅ **구조화된 출력**: JSON + 시각화 이미지  
✅ **배치 처리**: 폴더 내 여러 파일 동시 처리  

## 라이선스 및 참고

프로젝트는 PaddleOCR 코드 베이스(https://github.com/PaddlePaddle/PaddleOCR)를 참고하여 작성되었습니다.

PDF 파일 처리 시 스크립트는 **정확한 좌표 매칭과 여백 처리**를 보장합니다:

**v3 개선사항 (2024-10-27 - 여백 처리):**
1. **CropBox 여백 고려**: PDF의 cropbox를 기준으로 렌더링
   - PDF 뷰어에서 실제로 보이는 영역만 렌더링
   - cropbox 오프셋이 자동으로 적용되어 좌표 정확도 보장
   - 디버깅 출력: offset 정보 표시 (예: offset=(50.0, 50.0))

2. **자동 여백 감지**: 
   - MediaBox와 CropBox가 다른 경우 자동 감지
   - page.bound()로 실제 콘텐츠 영역 계산
   - 렌더링 시 clip=page.cropbox로 정확한 영역만 처리

3. **상세 디버깅 정보**:
   ```
   페이지 0: 이미지=1335x2076, scale=3.0, 원본=495x742, offset=(50.0, 50.0)
   ✓ 모든 박스 정상 (3개)
   ```

**v2 개선사항 (2024-10-27):**
1. **마진 제거**: `clip=None` 파라미터로 PDF 렌더링 시 자동 여백 제거
   - PyMuPDF의 기본 마진이 좌표 오프셋을 발생시키는 문제 해결
   - 전체 페이지 렌더링으로 좌표 정확도 극대화

2. **좌표 검증**: 모든 박스가 이미지 범위 내에 있는지 자동 검증
   - 범위를 벗어난 좌표 자동 클리핑 및 경고 출력
   - 디버깅 정보: 페이지별 이미지 크기, 스케일, 원본 크기 출력

3. **페이지 정보 추적**: 원본 PDF 페이지 크기 정보 보존
   - 추후 좌표 변환이 필요한 경우를 대비한 메타데이터 저장
   - 스케일 팩터(1x, 2x, 3x)와 원본 크기 정보 모두 기록

**v1 개선사항 (Context7 기반):**
1. **좌표 일치 보장**: PDF를 이미지로 변환한 후, 그 이미지를 OCR에 직접 전달
   - OCR 좌표와 시각화 이미지의 좌표가 정확히 일치
   - 이전 방식: PDF 경로 직접 전달 → 내부 변환 시 해상도 불일치 가능
   
2. **고해상도 렌더링**: 
   - 기본 3배 해상도 (Matrix 3x3)로 텍스트 경계 정확도 향상
   - 페이지 크기에 따라 자동 조정 (3000px 초과 시 2배, 4500px 초과 시 1배)
   
3. **페이지별 시각화**: 
   - 각 PDF 페이지마다 개별 시각화 이미지 생성
   - 파일명: `{filename}_page_{N}_vis.jpg`
   - 박스와 인식된 텍스트가 정확한 위치에 표시

**기술적 세부사항:**
```python
# PDF → 이미지 변환 (v3: cropbox 기준 렌더링, 여백 자동 처리)
page_info = {
    'cropbox': page.cropbox,  # 실제 보이는 영역
    'mediabox': page.mediabox,  # 물리적 페이지 크기
    'offset': (cropbox.x0, cropbox.y0)  # 여백 오프셋
}

# cropbox 기준 렌더링으로 여백 제거
pm = page.get_pixmap(matrix=mat, alpha=False, clip=page.cropbox)

# 변환된 이미지를 OCR에 직접 전달 (좌표 정확도 보장)
for img in pdf_images:
    result = ocr.predict(img)  # numpy array 직접 사용
    
# 좌표 검증 및 자동 클리핑
for box in boxes:
    if is_out_of_bounds(box, img_size):
        box = clip_to_image(box, img_size)  # 안전 장치
```

**여백이 있는 PDF 지원:**
- CropBox가 MediaBox와 다른 PDF (여백이 있는 문서)
- 스캔된 PDF의 자동 트리밍 영역
- PDF 편집기에서 여백을 설정한 문서
### 출력 형식

스크립트는 각 파일에 대해 다음을 생성합니다:

1. **JSON 파일** (`{filename}_ocr.json`):
   - 페이지별 OCR 결과
   - 각 텍스트 박스의 좌표, 인식된 텍스트, 신뢰도 점수

2. **시각화 이미지**:
   - 이미지 입력: `{filename}_vis.jpg`
   - PDF 입력: 페이지별 `{filename}_page_{N}_vis.jpg`
   - 감지된 텍스트 박스와 인식 결과가 표시됨

3. **요약 파일** (`summary.json`):
   - 모든 처리된 파일의 통계 정보

### JSON 출력 예시

```json
[
  {
    "page": 0,
    "num_boxes": 5,
    "entries": [
      {
        "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "rect": [xmin, ymin, xmax, ymax],
        "text": "인식된 텍스트",
        "score": 0.95
      }
    ]
  }
]
```

출력 예시:
- `output/` 디렉터리에 OCR 결과 이미지와 JSON이 저장됩니다.

## 로컬 이미지 사용법
`pp-ocr-v5.py` 내부의 `ocr.predict(input=...)`에 이미지 파일 경로(절대/상대)나 이미지 URL을 넣어 실행하면 됩니다. 예:

```python
# 예시 (스크립트 내에서 사용):
result = ocr.predict(input="./images/sample.png")
```

## 주의 및 팁
- PaddlePaddle 설치는 OS, Python 버전, CUDA 버전에 민감합니다. GPU 사용 시 공식 설치 문서를 먼저 확인하세요.
- `requirements.txt`는 이 저장소에서 최소한으로 필요한 패키지를 명시합니다. (CPU 환경에서 동작하도록 작성되어 있음)

## 참고
- PaddleOCR 레포: https://github.com/PaddlePaddle/PaddleOCR

## 스모크 테스트(실행 캡처)
다음은 이 저장소 루트에서 진행한 간단한 스모크 테스트와 실제 캡처된 출력 요약입니다. 환경은 프로젝트 가상환경 `.venv`를 사용했습니다.

1) `main.py` 실행 (paddle 설치 확인)

출력(요약):

```
Hello from paddle-ocr!
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 GPU.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

2) `paddleocr` 설치

설치 후 정상적으로 `paddleocr`와 `paddlex`가 가상환경에 설치되었습니다.

3) `pp-ocr-v5.py` 실행

명령:

```bash
python pp-ocr-v5.py --input ./images/sample.png --out ./output
```

정상적으로 OCR이 실행되고 결과가 `output/` 디렉토리에 저장됩니다.

