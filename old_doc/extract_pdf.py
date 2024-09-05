import fitz  # PyMuPDF

# Path to the PDF file
pdf_path = 'doc/pdf/TrouvainHinaut2022_ReservoirPy-RC-Tool-Python_preprint_HAL-V1.pdf'

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Extract text from each page
text = ""
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    text += page.get_text()

with open('doc/md/TH2022_ReservoirPy_RC_Tool.md', 'w', encoding='utf-8') as file:
    file.write(text)

print("Text extracted'")