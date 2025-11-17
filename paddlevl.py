from paddleocr import PaddleOCRVL
image_path = "images/618186.png"

try:
    pipeline = PaddleOCRVL()
except Exception as e:
    # Provide a clear, actionable message when dependencies are missing
    msg = str(e)
    hint = (
        "A dependency error occurred while creating the PaddleOCR-VL pipeline.\n"
        "Install the required extras into your virtualenv and retry:\n"
        "  /home/hyper/project/paddle-ocr/.venv/bin/pip install 'paddlex[ocr]==3.3.9'\n"
        "If your `paddlex` version differs, replace 3.3.9 with your installed version.\n"
    )
    raise RuntimeError(hint + "\nOriginal error:\n" + msg) from e

output = pipeline.predict(image_path)
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")