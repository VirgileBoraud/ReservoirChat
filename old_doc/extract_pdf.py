import fitz  # PyMuPDF
import os

# Dossier contenant les fichiers PDF
pdf_folder = 'Documentation_Training-Dataset/2-Little/no_change/mnemosyne-team'
md_folder = 'Documentation_Training-Dataset/2-Little/no_change/mnemosyne-team'

# Vérifier si le dossier md existe, sinon le créer
if not os.path.exists(md_folder):
    os.makedirs(md_folder)

# Lister tous les fichiers PDF dans le dossier
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        # Chemin complet vers le fichier PDF
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # Ouvrir le fichier PDF
        pdf_document = fitz.open(pdf_path)

        # Extraire le texte de chaque page
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        # Générer le nom du fichier .md à partir du nom du fichier PDF
        md_filename = os.path.splitext(pdf_file)[0] + '.md'
        md_path = os.path.join(md_folder, md_filename)

        # Sauvegarder le texte extrait dans un fichier .md
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(text)

        print(f"Fichier {md_filename} créé avec succès.")
