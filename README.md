# SAT Question Processor

This Streamlit app processes SAT questions from a PDF file using the Claude API and outputs the results as a CSV file.

## Features

- Upload PDF files containing SAT questions
- Process questions using Claude API
- Download results as a CSV file

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- pandas
- requests

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sat-question-processor.git
   cd sat-question-processor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run sat_question_processor.py
   ```

2. Open the provided URL in your web browser.

3. Enter your Claude API key in the app.

4. Upload a PDF file containing SAT questions.

5. Click "Process PDF" to start processing.

6. Once processing is complete, download the CSV file with the results.

## Hosting the App

To host the Streamlit app, you can use Streamlit Cloud or deploy it on platforms like Heroku or Google Cloud Platform. Here are instructions for using Streamlit Cloud:

1. Push your code to a GitHub repository.

2. Sign up for a free account at [streamlit.io](https://streamlit.io/).

3. Create a new app and connect it to your GitHub repository.

4. Select the main file (sat_question_processor.py) as the entry point.

5. Deploy the app.

Note: Make sure to set up environment variables for any sensitive information like API keys when deploying to a hosting platform.

## For example:

Input: https://drive.google.com/file/d/1gpgk1bFMxlFvjNb9sd6uIEUF89A9LM1h/view?usp=sharing

Output: https://drive.google.com/file/d/1suAM4bmzMP3qoDdOFxdhTR91rvALV4L2/view?usp=sharing

## License

This project is licensed under the MIT License.
