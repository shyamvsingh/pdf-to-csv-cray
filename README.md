# SAT PDF Parser

This project contains a command line tool for converting SAT practice PDFs into a structured CSV suitable for importing into Supabase.

The script uses the Mathpix API to perform OCR on each page and OpenAI models to structure the results into question objects. Any images extracted from Mathpix are saved locally and referenced in the CSV output.

## Requirements

- Python 3.8+
- openai
- python-dotenv
- pandas
- requests
- PyMuPDF
- Pillow

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

## Output Format

The CSV contains the following columns:
- `question_id`
- `question_text`
- `choice_A`
- `choice_B`
- `choice_C`
- `choice_D`
- `correct_answer`
- `domain`
- `skill`
- `difficulty`
- `image_path`

## License

This project is licensed under the MIT License.
