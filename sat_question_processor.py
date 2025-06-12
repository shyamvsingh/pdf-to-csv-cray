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


def extract_images_with_ocr(pdf_bytes, page_range):
    """Extract images from the specified pages and run OCR on them."""
    ocr_fragments = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in page_range:
        if page_num < len(doc):
            page = doc[page_num]
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # convert CMYK/GRAYSCALE/etc to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                pix = None
                img_obj = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img_obj)
                if text.strip():
                    ocr_fragments.append(text.strip())
    doc.close()
    return "\n".join(ocr_fragments)

def extract_text_from_pdf(pdf_file, start_page, end_page):
    """Extract text and images with placeholders from the given page range."""
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    placeholder_map = {}
    combined_parts = []
    image_counter = 1

    for page_num in range(start_page, min(end_page, len(doc))):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        elements = []
        for block in page_dict.get("blocks", []):
            bbox = block.get("bbox", (0, 0, 0, 0))
            top = bbox[1]
            if block["type"] == 0:  # text block
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                    text += "\n"
                if text.strip():
                    elements.append({"top": top, "type": "text", "content": text.strip()})
            elif block["type"] == 1:  # image block
                xref = block.get("xref")
                if xref is None:
                    continue
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                pix = None
                placeholder = f"[[IMAGE_{image_counter}]]"
                ocr_text = pytesseract.image_to_string(Image.open(io.BytesIO(img_bytes))).strip()
                placeholder_map[placeholder] = ocr_text
                elements.append({"top": top, "type": "image", "content": placeholder})
                image_counter += 1

        elements.sort(key=lambda e: e["top"])
        for elem in elements:
            combined_parts.append(elem["content"])
        combined_parts.append("\n")

    doc.close()
    pdf_file.seek(0)
    combined_text = "\n".join(part for part in combined_parts if part).strip()
    return combined_text, placeholder_map

def process_pdf_chunk(chunk_text, placeholder_map, api_key):
    placeholder_info = json.dumps(placeholder_map, indent=2)

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

    Image OCR mapping:
    {placeholder_info}

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

            for start_page in range(0, total_pages, 8):
                end_page = min(start_page + 8, total_pages)
                chunk_text, placeholder_map = extract_text_from_pdf(uploaded_file, start_page, end_page)

                processed_data = process_pdf_chunk(chunk_text, placeholder_map, api_key)
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
                st.success("Processing complete! You can now download the CSV file.")
            else:
                st.error("No questions were processed successfully.")

if __name__ == "__main__":
    main()
