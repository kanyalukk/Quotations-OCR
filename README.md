# Fix for TesseractNotFoundError

If you deploy on **Streamlit Community Cloud**, include **packages.txt** with:
```
tesseract-ocr
tesseract-ocr-tha
```
These are apt packages that install the binary under `/usr/bin/tesseract` and Thai language data.

## Local install quick guide

- **macOS (Homebrew):**
  ```bash
  brew install tesseract tesseract-lang
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-tha
  ```
- **Windows:**
  1) Install Tesseract (UB Mannheim builds)
  2) Add its path to `PATH` or set in app sidebar:
     `C:\Program Files\Tesseract-OCR\tesseract.exe`

In the app sidebar you can set **Tesseract path** if auto-detection fails.