# OCR Quotation/Bill ➜ Table (Thai+English)

This Streamlit app extracts key fields from Thai/English quotation/bill images:
**file, Vendor/Supplier, Quotation No., Date, Description, Subtotal, VAT, Grand Total**, and (optionally) exports to **Google Sheets**.

## Quickstart

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

### Tesseract Installation

- **macOS (Homebrew):**
  ```bash
  brew install tesseract tesseract-lang
  brew install tesseract-lang # if needed
  # Thai data usually included; otherwise:
  brew install tesseract-lang  # or place 'tha.traineddata' into tessdata
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-tha
  ```
- **Windows:**
  - Install Tesseract from the official repo, ensure `tesseract.exe` is on PATH.

If EasyOCR (torch) is hard to install, switch the sidebar to **"Tesseract only"**.

### Google Sheets Export (Optional)

1. Create a **Service Account** in Google Cloud, enable Google Sheets API.
2. Download the JSON key and upload it in the app (left sidebar).
3. Share your Sheet with the Service Account email (Editor).
4. Paste the Sheet URL and click **Export**.

### Notes

- The app **preprocesses** each image: grayscale → denoise → adaptive threshold → deskew → upscale → morphology.
- Hybrid OCR: EasyOCR (paragraph mode) + Tesseract (psm 6). We reconcile and then extract fields via Regex/heuristics.
- Regex patterns are tuned for Thai/English keywords; adjust in `extract_*` functions.