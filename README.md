# SAT PDF Parser

This project contains a command line tool for converting SAT practice PDFs into a structured CSV suitable for importing into Supabase.

The script uses the Mathpix API to perform OCR on each page and OpenAI models to structure the results into question objects. Any images extracted from Mathpix are saved locally and referenced in the CSV output. Math formulas are preserved in LaTeX so that expressions like systems of equations render correctly on the front end.

## Requirements

- Python 3.8+
- openai
- python-dotenv
- pandas
- requests
- PyMuPDF
- Pillow
- streamlit
## Installation

1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root containing your API keys:
   ```env
   OPENAI_API_KEY=sk-...
   MATHPIX_APP_ID=your_app_id
   MATHPIX_APP_KEY=your_app_key
   ```

## Usage

Run the parser and specify the input PDF and optional output CSV path:

```bash
python parse_sat_pdf.py path/to/questions.pdf --csv parsed_questions.csv
```

Images extracted from the PDF will be stored in the `images/` directory and the questions will be appended to the CSV file.

If the OpenAI response contains stray backslashes that break JSON formatting,
the parser will attempt to escape them and retry parsing automatically.

## Streamlit Interface

Launch an interactive UI:
```bash
streamlit run streamlit_app.py
```
Upload a PDF, click "Convert" and then use the provided download button to save the resulting CSV.


## Output Format

The CSV contains the following columns:
- `question_id`
- `question_text`
- LaTeX may appear inside `question_text` for equations or functions.
- `choice_A`
- `choice_B`
- `choice_C`
- `choice_D`
- `correct_answer`
- `domain`
- `skill`
- `difficulty`
- `image_path`

If a question has no answer choices, the `choice_A` through `choice_D` fields
will contain the text `free response` so you can easily identify these
questions when reviewing the CSV.

## License

This project is licensed under the MIT License.
