from pypdf import PdfReader


def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
