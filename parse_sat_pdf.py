import os
import io
import base64
import json
import json5
import time
import uuid
import re
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image
import openai
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")

CSV_PATH = "parsed_questions.csv"
IMAGE_DIR = "images"

os.makedirs(IMAGE_DIR, exist_ok=True)

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)



def clean_json_reply(reply: str) -> str:
    """Extract JSON content from an OpenAI reply."""
    if not reply:
        return ""
    cleaned = reply.strip()

    # Prefer JSON inside fenced blocks
    fence_match = re.search(r"```(?:json)?(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        # Fall back to the first JSON object found
        obj_match = re.search(r"{.*}", cleaned, re.DOTALL)
        if obj_match:
            cleaned = obj_match.group(0)

    return cleaned.strip()


def safe_json_loads(clean_reply: str) -> Optional[Dict[str, Any]]:
    """Attempt to load JSON using multiple strategies."""
    try:
        return json.loads(clean_reply)
    except json.JSONDecodeError as e:
        if "Invalid \\escape" in str(e):
            clean_reply = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', clean_reply)
            try:
                return json.loads(clean_reply)
            except json.JSONDecodeError:
                pass
    try:
        return json5.loads(clean_reply)
    except Exception:
        return None


def parse_pdf_page(doc: fitz.Document, page_num: int) -> bytes:
    """Render a page from an open PDF document to image bytes."""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=300)
    return pix.tobytes("png")



def extract_mathpix_data(image_bytes: bytes, retries: int = 3) -> Dict[str, Any]:
    """Call Mathpix API and return its JSON response."""
    if not (MATHPIX_APP_ID and MATHPIX_APP_KEY):
        raise RuntimeError("Mathpix credentials not found in environment")
    url = "https://api.mathpix.com/v3/text"
    headers = {"app_id": MATHPIX_APP_ID, "app_key": MATHPIX_APP_KEY}
    data = {
        "formats": ["text", "latex_styled"],
        "data_options": {"include_asciimath": False, "include_latex": True},
        "include_smiles": False,
        "enable_tables": True,
        "include_tables": True,
        "include_image_data": True,
    }
    files = {"file": ("page.png", image_bytes, "image/png")}

    for attempt in range(retries):
        try:
            resp = requests.post(
                url,
                headers=headers,
                data={"options_json": json.dumps(data)},
                files=files,
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logging.error(f"Mathpix attempt {attempt + 1} failed: {e}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Mathpix request failed after {retries} attempts")


def save_image(b64_data: str, question_id: str, suffix: str) -> str:
    """Decode base64 image data and save to disk."""
    if not b64_data:
        return ""
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes))
    filename = f"q{question_id}_{suffix}.png"
    path = os.path.join(IMAGE_DIR, filename)
    img.save(path)
    return path


def extract_pdf_content(pdf_path: str) -> (str, Dict[str, str]):
    """Extract OCR text and images from an entire PDF."""
    doc = fitz.open(pdf_path)
    try:
        full_text = ""
        image_map: Dict[str, str] = {}
        for page_num in range(len(doc)):
            page_bytes = parse_pdf_page(doc, page_num)
            mathpix_data = extract_mathpix_data(page_bytes)
            page_text = mathpix_data.get("latex_styled") or mathpix_data.get("text", "")
            full_text += f"\n=== PAGE {page_num + 1} ===\n" + page_text

            images = mathpix_data.get("images", []) or []
            for idx, img in enumerate(images, start=1):
                img_id = str(uuid.uuid4())[:8]
                path = save_image(img.get("data", ""), img_id, f"p{page_num+1}_{idx}")
                image_map[f"page{page_num+1}_image{idx}"] = path
        return full_text, image_map
    finally:
        doc.close()

def structure_question_with_openai(text: str, image_map: Dict[str, str], retries: int = 3) -> List[Dict[str, Any]]:
    """Use OpenAI to structure raw text into question objects."""
    prompt = (
        "You are parsing SAT practice questions from OCR text. "
        "Count each time the phrase 'Question ID' appears followed by an alphanumeric identifier. "
        "Everything between one 'Question ID' line and the next belongs to the first identifier. "
        "Return exactly that many questions as a JSON object with a 'questions' array. "
        "Each question must include the fields: question_id, question_text, choice_A, choice_B, "
        "choice_C, choice_D, correct_answer, domain, skill, difficulty, image_path. "
        "Use the identifier following 'Question ID' for question_id. Do not fabricate information. "
        "If the text references diagrams, graphs, tables, or other images, set image_path to 'needs image'. "
        "If any answer choice is an image, its value should be 'needs image'. "
        "If domain, skill, or difficulty are missing, use 'Not specified'."
    )
    content = f"OCR TEXT:\n{text}\nIMAGE MAP:{json.dumps(image_map)}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]

    for attempt in range(retries):
        reply = ""
        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2048,
                temperature=0,
            )
            reply = resp.choices[0].message.content
            clean_reply = clean_json_reply(reply)
            data = safe_json_loads(clean_reply)
            if data is None:
                raise ValueError("JSON parsing failed")
            if (
                isinstance(data, dict)
                and isinstance(data.get("questions"), list)
            ):
                return data["questions"]
        except Exception as e:
            logging.error(f"OpenAI attempt {attempt + 1} failed: {e}")
            if reply:
                logging.error(f"OpenAI raw reply: {reply}")
            if 'clean_reply' in locals():
                logging.error(f"OpenAI cleaned reply: {clean_reply}")
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAI request failed or returned invalid JSON")


def append_to_csv(
    questions: List[Dict[str, Any]], csv_path: Optional[str] = None
) -> None:
    """Append question data to a CSV file."""
    if csv_path is None:
        csv_path = CSV_PATH

    df = pd.DataFrame(questions)

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df_prev = pd.read_csv(csv_path)
            df = pd.concat([df_prev, df], ignore_index=True)
        except Exception as e:
            logging.error(f"Failed reading existing CSV: {e}")

    df.to_csv(csv_path, index=False)


def process_pdf(pdf_path: str, csv_path: str = CSV_PATH) -> None:
    text, image_map = extract_pdf_content(pdf_path)
    logging.info(f"Combined OCR text length: {len(text)}")

    questions = structure_question_with_openai(text, image_map)
    for q in questions:
        if q.get("image_path") in image_map:
            q["image_path"] = image_map[q["image_path"]]

    append_to_csv(questions, csv_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse SAT PDF to CSV")
    parser.add_argument("pdf", help="Path to SAT PDF")
    parser.add_argument("--csv", default=CSV_PATH, help="Output CSV file")
    args = parser.parse_args()

    process_pdf(args.pdf, args.csv)
    print(f"Results saved to {args.csv}")
