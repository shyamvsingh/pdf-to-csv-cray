import os
import tempfile
import streamlit as st
import parse_sat_pdf

st.title("PDF to CSV Converter")

uploaded_file = st.file_uploader("Drag and drop a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    output_csv = os.path.join(tempfile.gettempdir(), "converted_questions.csv")
    parse_sat_pdf.CSV_PATH = output_csv

    if st.button("Convert"):
        with st.spinner("Processing PDF..."):
            parse_sat_pdf.process_pdf(tmp_path)
        with open(output_csv, "rb") as f:
            st.download_button(
                label="Download CSV",
                data=f,
                file_name="questions.csv",
                mime="text/csv",
            )
        os.remove(output_csv)
        os.remove(tmp_path)
