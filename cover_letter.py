import pdfplumber
from model import maze_ai
pdf_path = "Pranay_Satija.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for  page in pdf.pages:
        text = page.extract_text()
        
prompt=f"""Write a professional cover letter for the job application based on the following resume details: {text}.
the letter is for swiggy sde1 role. keep it concise and to the point.
"""
print(maze_ai(prompt))
