from pathlib import Path
import json
import fitz
import pytesseract

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DOCUMENTS_DIR = Path("documents")


def render_page(page, dpi=200):
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def ocr_page(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except ValueError:
            confidence = -1
        x = data["left"][i]
        y = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]
        words.append({
            "text": text,
            "confidence": confidence,
            "bbox": [x, y, x + width, y + height],
        })
    return words


def ocr_pdf(pdf_path):
    document = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(document, start=1):
        print(f"  OCR page {page_number}/{len(document)}")
        image = render_page(page)
        words = ocr_page(image)
        pages.append({"page_number": page_number, "words": words, "image_width": image.width, "image_height": image.height})
    document.close()
    return pages


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    output_dir = Path("data/ocr")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(pdf_files)} PDFs")
    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path.name}")
        try:
            pages = ocr_pdf(pdf_path)
            output_file = output_dir / (pdf_path.stem + ".json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"document": pdf_path.name, "pages": pages}, f, indent=2, ensure_ascii=False)
            total_words = sum(len(page["words"]) for page in pages)
            print(f"Extracted {total_words} words")
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"ERROR processing {pdf_path.name}: {e}")


if __name__ == "__main__":
    main()