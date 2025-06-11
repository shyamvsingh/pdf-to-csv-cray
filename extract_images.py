# extract_images.py
import fitz  # PyMuPDF
import os

# 1) Update this to match your PDF filename in the project root:
PDF_PATH = "your_test_pdf.pdf"

# 2) This script will dump images into ./images/
OUT_DIR = "images"
os.makedirs(OUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)
for page_num, page in enumerate(doc, start=1):
    for img_index, img in enumerate(page.get_images(full=True), start=1):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        out_path = os.path.join(
            OUT_DIR, f"Q{page_num}_img{img_index}.png"
        )
        pix.save(out_path)
        pix = None
        print(f"Saved image: {out_path}")
print("✅ Done extracting images.")
