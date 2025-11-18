from paddleocr import PaddleOCRVL
image_path = "images/a.pdf"
pipeline = PaddleOCRVL()

output = pipeline.predict(image_path)
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")