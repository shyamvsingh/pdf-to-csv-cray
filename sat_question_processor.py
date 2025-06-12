import streamlit as st
import os
import json
import time
import re
import requests
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd
import io

# API configuration
API_URL = "https://api.anthropic.com/v1/messages"

# Mathpix OCR endpoint
MATHPIX_URL = "https://api.mathpix.com/v3/text"


def mathpix_ocr(img_bytes):
    """Run Mathpix OCR on the given image bytes and return LaTeX or text."""
    app_id = os.environ.get("MATHPIX_APP_ID")
    app_key = os.environ.get("MATHPIX_APP_KEY")
    if not (app_id and app_key):
        return None

    headers = {
        "app_id": app_id,
        "app_key": app_key,
    }
    data = {
        "formats": ["text", "latex_styled"],
        "data_options": {"include_asciimath": False, "include_latex": True},
    }

    files = {"file": ("image.png", img_bytes)}
    try:
        resp = requests.post(
            MATHPIX_URL,
            headers=headers,
            data={"options_json": json.dumps(data)},
            files=files,
            timeout=15,
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("latex_styled") or result.get("text")
    except Exception:
        pass
    return None


def looks_like_math(text):
    """Heuristic check whether OCR text likely contains math."""
    if not text:
        return False
    math_chars = "=+-*/^_()[]{}\\|<>"  # common math symbols
    score = sum(1 for c in text if c in math_chars)
    return score > len(text) * 0.2

def call_claude_api(prompt, api_key):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-opus-4-20250514",
        "max_tokens": 8192,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['content'][0]['text']
    else:
        st.error(f"API call failed with status code: {response.status_code}")
        st.error(f"Response: {response.text}")
        return None


def extract_images_with_ocr(pdf_bytes, page_range, out_dir="extracted_images"):
    """Extract images from the specified pages, run OCR and return results."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in page_range:
        if page_num < len(doc):
            page = doc[page_num]
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # convert CMYK/GRAYSCALE/etc to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                pix = None
                img_obj = Image.open(io.BytesIO(img_bytes))
                img_path = os.path.join(
                    out_dir, f"page{page_num + 1}_img{img_index}.png"
                )
                img_obj.save(img_path)

                text = pytesseract.image_to_string(img_obj).strip()
                if looks_like_math(text):
                    latex = mathpix_ocr(img_bytes) or text
                else:
                    latex = text

                if latex:
                    results.append({"image": img_path, "text": latex})
    doc.close()
    return results

def extract_text_from_pdf(pdf_file, start_page, end_page, image_out_dir="extracted_images"):
    """Extract text and OCR content from the given page range."""
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page_num in range(start_page, min(end_page, len(reader.pages))):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            text += page_text

    image_results = extract_images_with_ocr(pdf_bytes, range(start_page, end_page), image_out_dir)
    if image_results:
        ocr_text = "\n".join(f"[IMAGE: {res['image']}] {res['text']}" for res in image_results)
        text += f"\n{ocr_text}"

    pdf_file.seek(0)
    return text, image_results

def process_pdf_chunk(chunk_text, api_key):
    prompt = f"""
    Analyze the following text extracted from an SAT question paper and format it into a structured JSON output. Extract the following details for each question:
    - Question ID (for example - 6ed4df)
    - Question text including the complete passage, table (in markdown), etc.
    - Options (A, B, C, D, and sometimes E)
    - Correct answer (the letter of the correct option)
    - Rationale (a rationale of why the correct answer is right and other options are incorrect which is provided)
    - Test (if available, otherwise use "Not specified")
    - Domain (if available, otherwise use "Not specified")
    - Skill (if available, otherwise use "Not specified")
    - Difficulty (if available, otherwise use "Not specified")

    Format the extracted information into a JSON structure as follows:

    {{
      "questions": [
        {{
          "question_id": "6ed4qc",
          "question_text": "The human brain is primed to recognize faces—so much so that, due to a perceptual tendency called pareidolia, ______ will even find faces in clouds, wooden doors, pieces of fruit, and other faceless inanimate objects. Researcher Susan Magsamen has focused her work on better understanding this everyday phenomenon.Which choice completes the text so that it conforms to the conventions of Standard English?",
          "options": [
            {{"label": "A", "text": "she"}},
            {{"label": "B", "text": "they"}},
            {{"label": "C", "text": "it"}},
            {{"label": "D", "text": "those"}}
          ],
          "correct_answer": "C",
          "rationale": "Choice C is the best answer. "It" is a singular pronoun used to stand in for objects. Since the antecedent in this case is the singular noun phrase "the human brain," "it" is a perfect pronoun to use here. Choice A is incorrect. Although "she" is a singular pronoun, it is reserved for people and animals, not objects like "the human brain." Choice B is incorrect. "They" is a plural pronoun, but we need a singular pronoun to represent the antecedent "the human brain." Choice D is incorrect. "Those" is a plural pronoun, but we need a singular pronoun to represent the antecedent "the human brain."",
          "test": "Reading and Writing",
          "domain": "Standard English Conventions",
          "skill": "Form, Structure and Tense",
          "difficulty": "Medium"
        }}
      ]
    }}

    Here's the text to process:

    {chunk_text}

    Important instructions:
    - Your response should contain ONLY the JSON output, nothing else.
    - Process ALL questions in the input text.
    - Ensure the JSON is properly formatted and can be parsed by a JSON parser.
    - Use double quotes for all strings in the JSON.
    - Escape any special characters in the text that might break the JSON structure.
    - If a question doesn't have a clear skill, domain, or difficulty, use "Not specified" as the value.
    - Do not include any commentary or explanations outside the JSON structure.
    """

    response = call_claude_api(prompt, api_key)
    
    if response:
        json_match = re.search(r'(\{|\[).*(\}|\])', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group().strip()
                parsed_json = json.loads(json_str)
                if "questions" in parsed_json and isinstance(parsed_json["questions"], list):
                    return parsed_json
                else:
                    st.error(f"Invalid JSON structure. Expected 'questions' key with list value.")
                    return None
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in response: {e}")
                st.error(f"JSON string: {json_str}")
                return None
        else:
            st.error(f"No valid JSON found in the response.")
            st.error(f"Response: {response}")
            return None
    else:
        st.error(f"Failed to process input text.")
        return None

def main():
    st.title("SAT Question Processor")

    api_key = st.text_input("Enter your Claude API key:", type="password")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and api_key:
        if st.button("Process PDF"):
            reader = PyPDF2.PdfReader(uploaded_file)
            total_pages = len(reader.pages)

            progress_bar = st.progress(0)
            status_text = st.empty()

            all_questions = []
            image_ocr_results = []

            for start_page in range(0, total_pages, 8):
                end_page = min(start_page + 8, total_pages)
                chunk_text, images_data = extract_text_from_pdf(
                    uploaded_file, start_page, end_page
                )
                
                processed_data = process_pdf_chunk(chunk_text, api_key)
                image_ocr_results.extend(images_data)
                time.sleep(10)
                
                if processed_data and 'questions' in processed_data:
                    all_questions.extend(processed_data['questions'])
                    status_text.text(f"Processed pages {start_page+1}-{end_page}")
                else:
                    status_text.text(f"No valid data found for pages {start_page+1}-{end_page}")
                
                progress = (end_page / total_pages)
                progress_bar.progress(progress)
                
                time.sleep(2)

            if all_questions:
                df = pd.DataFrame(all_questions)
                df['options'] = df['options'].apply(lambda x: '; '.join([f"{opt['label']}: {opt['text']}" for opt in x]))
                
                csv = df.to_csv(index=False)
                csv_bytes = csv.encode()
                
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="sat_questions.csv",
                    mime="text/csv"
                )
                if image_ocr_results:
                    img_df = pd.DataFrame(image_ocr_results)
                    img_csv = img_df.to_csv(index=False)
                    st.download_button(
                        label="Download Image OCR",
                        data=img_csv,
                        file_name="image_ocr.csv",
                        mime="text/csv",
                    )
                st.success("Processing complete! You can now download the CSV file.")
            else:
                st.error("No questions were processed successfully.")

if __name__ == "__main__":
    main()
