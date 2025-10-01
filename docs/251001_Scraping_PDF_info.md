# Research: Scraping Data from PDFs

_Date: 2025-10-01_   

## 📝 Task / Focus
Gather info about sacraping CV data from PDF documents

## Why scrape PDFs?

PDFs are widely used to share structured information (CVs, invoices, reports).
The problem: PDFs were designed for presentation, not for machine reading.
Scraping data from PDFs means developing ways to transform a visual document into usable, structured data.

## Categories of PDFs
**A. “Friendly” PDFs (text)**
- Created digitally (e.g., exported from Word).
- Text is embedded.
- Easy to parse — libraries can extract characters directly.

**B. “Unfriendly” PDFs (image)**
- Scanned or flattened into images.
- Contain no text layer.
- Require OCR → error-prone if the scan is poor.

**C. “Mixed” PDFs**
- Contain both text and images (e.g., CV with a profile picture, logos, and a text layer).
- Need a combination of extraction + OCR.

## Tools I found
**A. Text extraction**
- [pdfplumber](https://www.pdfplumber.com/category/guide/): good balance between simplicity and accuracy, can also handle tables.

    "It is a powerful Python library designed for precise extraction of content from PDF files. It extends the capabilities of pdfminer.six and gives you control over the layout and structure of the PDF, making it ideal for extracting"

- [PDFMiner.six](https://pdfminersix.readthedocs.io/en/latest/): lower-level, more control, but more complex.
    
    "Pdfminer.six is a python package for extracting information from PDF documents."
- [PyPDF2](https://pypi.org/project/PyPDF2/): simple PDF reader and text extractor, but weak with formatting.

    "A pure-python PDF library capable of splitting, merging, cropping, and transforming PDF files"

**B. Table-focused**
- [Camelot](https://camelot-py.readthedocs.io/en/master/): clever at detecting structured tables, but fails with irregular layouts.

    "Camelot is a Python library that can help you extract tables from PDFs."
- [Tabula-py](https://tabula-py.readthedocs.io/en/latest/): very practical for turning tabular data into pandas DataFrames.

    "It is a simple Python wrapper of tabula-java, which can read table of PDF. You can read tables from PDF and convert them into pandas’ DataFrame. tabula-py also converts a PDF file into CSV/TSV/JSON file."

-> Chck out [Comparison with other PDF Table Extraction libraries and tools](https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools), written by Camelot developers.


**C. Image/OCR**
- [pytesseract](https://pypi.org/project/pytesseract/) + [pdf2image](https://pypi.org/project/pdf2image/): convert PDF pages → images → text.

    "Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and “read” the text embedded in images."

    "pdf2image is a python (3.7+) module that wraps pdftoppm and pdftocairo to convert PDF to a PIL Image object"

- External APIs: [Google Vision](https://cloud.google.com/vision/docs), [AWS Textract](https://aws.amazon.com/textract/) → higher accuracy, but external dependency & cost
    - Pros: much higher accuracy (esp. with messy scans), support for forms/tables, language models.
	- Cons: not free (after free tier), requires internet connection, adds dependency on cloud services.

## Testing strategy I would use
1. Detect the **PDF type** (text vs scanned).
    - Try reading with pdfplumber.
    - If result is empty → assume scanned, apply OCR.
2. **Normalize text** (remove line breaks, fix encoding issues).
3. Apply **regex/NLP** to extract structured info:
    - CVs: name, email, phone, skills.
    - Invoices: vendor, total, date.

## Takeaways
- There’s no “one-click” solution: scraping PDFs is about combining tool + strategy + cleaning.
- For CVs: pdfplumber + regex + fallback OCR is probably the best starting stack.
- Always expect manual review for edge cases.
- The hardest part isn’t reading the PDF → it’s turning messy extracted text into structured fields.