import sys
from pathlib import Path

reader_cls = None
try:
    # Try common PDF libraries
    from PyPDF2 import PdfReader as _PdfReader
    reader_cls = _PdfReader
except Exception:
    try:
        from pypdf import PdfReader as _PdfReader
        reader_cls = _PdfReader
    except Exception:
        print("PyPDF2/pypdf not installed. Install with: pip install PyPDF2 pypdf")
        sys.exit(2)

if len(sys.argv) < 2:
    print("Usage: python extract_pdf.py /path/to/file.pdf [out.txt]")
    sys.exit(1)

pdf_path = Path(sys.argv[1])
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tmp/paper_text.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

reader = reader_cls(str(pdf_path))
all_text = []
for p in reader.pages:
    try:
        text = p.extract_text()
    except Exception:
        text = None
    if text:
        all_text.append(text)

text = "\n\n".join(all_text)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Wrote extracted text to {out_path}")
