
import os
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image, ImageDraw

def create_test_pdf(filename="data/test_input/test_doc.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # 1. Text Section
    story.append(Paragraph("1. Text Analysis Layer", styles['Heading1']))
    story.append(Paragraph(
        "The retrieval-augmented generation (RAG) system must accurately parse text from PDFs. "
        "This paragraph checks if the text chunking interacts correctly with the parser. "
        "It should be preserved as a continuous text block.", 
        styles['Normal']
    ))
    story.append(Spacer(1, 12))

    # 2. Table Section
    story.append(Paragraph("2. Financial Data Table", styles['Heading1']))
    data = [
        ['Metric', 'Q1 2024', 'Q2 2024', 'Change'],
        ['Revenue', '$1.2M', '$1.4M', '+16%'],
        ['Expenses', '$0.8M', '$0.9M', '+12%'],
        ['Profit', '$0.4M', '$0.5M', '+25%'],
        ['Churn', '2.1%', '1.9%', '-0.2%']
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # 3. Chart (Vision/Image) Section - Generated via PIL
    story.append(Paragraph("3. Performance Chart (Vision Test)", styles['Heading1']))
    
    # Create simple chart using PIL
    img = Image.new('RGB', (400, 200), color='white')
    d = ImageDraw.Draw(img)
    d.rectangle([50, 50, 350, 180], outline="black", width=2)
    # Bars - (x0, y0, x1, y1) where y0 < y1 (top < bottom)
    d.rectangle([60, 140, 80, 180], fill="blue")
    d.rectangle([100, 130, 120, 180], fill="green")
    d.rectangle([140, 100, 160, 180], fill="red")
    d.text((150, 20), "Monthly Growth", fill="black")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        img_path = tmp.name
        
    story.append(ReportLabImage(img_path, width=400, height=200))
    story.append(Paragraph("Figure 1: Monthly User Growth vs Retention", styles['Italic']))
    story.append(Spacer(1, 12))

    # 4. Low-Info Decorative Image (Should be skipped/OCR-only) - PIL
    story.append(Paragraph("4. Low-Info Element", styles['Heading1']))
    
    # Create a tiny solid square
    tiny = Image.new('RGB', (20, 20), color='blue')
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_tiny:
        tiny.save(tmp_tiny.name)
        tiny_path = tmp_tiny.name
        
    story.append(ReportLabImage(tiny_path, width=20, height=20))
    story.append(Paragraph("(Decorative element above)", styles['Normal']))

    doc.build(story)
    print(f"Test PDF created at: {filename}")
    
    # Cleanup
    try:
        os.remove(img_path)
        os.remove(tiny_path)
    except:
        pass

if __name__ == "__main__":
    create_test_pdf()
