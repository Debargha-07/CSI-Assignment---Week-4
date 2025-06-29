from fpdf import FPDF
import os

# Folder where plots are saved
output_dir = "eda_images" 

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Title page
pdf.add_page()
pdf.set_font("Arial", "B", 20)
pdf.cell(200, 10, "Exploratory Data Analysis Report", ln=True, align="C")
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, "Dataset: Titanic (train.csv)", ln=True, align="C")
pdf.ln(10)

# Loop through the sorted images
for image in sorted(os.listdir(output_dir)):
    if image.endswith(".png"):
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, image.replace("_", " ").replace(".png", "").title(), ln=True)
        pdf.image(os.path.join(output_dir, image), x=10, y=25, w=180)

# Output 
pdf.output("EDA_Report_Titanic.pdf")
