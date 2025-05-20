from PyPDF2 import PdfReader
import re

def is_likely_toc_or_index_page(text):
    """Detect table of contents or index pages by checking for dotted line patterns."""
    lines = text.split("\n")
    return sum(1 for line in lines if re.search(r"\.{2,}\s*\d+$", line)) > 3

def clean_and_extract_text(pdf_path, min_chars=300):
    reader = PdfReader(pdf_path)
    cleaned_text = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text:
            continue

        raw_text = raw_text.strip()
        if len(raw_text) < min_chars:
            continue  # Skip very short pages (likely images or junk)

        if is_likely_toc_or_index_page(raw_text):
            continue  # Skip table of contents/index

        # Remove common artifacts like page headers or numbers
        text = re.sub(r"Page:\s*\d+|©\d{4}.*?Drut Technologies.*?(All Rights Reserved)?", "", raw_text, flags=re.IGNORECASE)
        text = re.sub(r"[\n]{2,}", "\n", text)  # Collapse multiple newlines
        text = re.sub(r"\s{2,}", " ", text)     # Collapse multiple spaces
        text = text.strip()

        if len(text) >= min_chars:
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)

# Run it
pdf_path = "/home/drut/chatbot/data/DSP_Installation_Guide.pdf"
cleaned_output = clean_and_extract_text(pdf_path)

# Save to a clean text file
with open("/home/drut/chatbot/finetuningllm/dsp_cleaned_text.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_output)

print("Clean instructional text saved to dsp_cleaned_text.txt")
