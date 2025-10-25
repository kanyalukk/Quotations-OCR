# app.py — Minimal (Tesseract-only) OCR Quotation Extractor
# Designed for Streamlit Cloud: no torch/EasyOCR deps.

import os, re, json, shutil
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import dateparser

try:
    import pytesseract
except Exception:
    pytesseract = None

# ---------- Utils
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}

def to_english_digits(s: str) -> str:
    return s.translate(THAI_DIGITS) if isinstance(s, str) else s

def fix_numberlike_ocr(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub(r'(?<=\d)[oO](?=[\d,\.])', '0', s)
    s = re.sub(r'(?<=[,\.\s])[oO](?=\d)', '0', s)
    s = re.sub(r'(?<=\d)[lI](?=[\d,\.])', '1', s)
    s = re.sub(r'(?<=\d)B(?=[\d,\.])', '8', s)
    return s

def sanitize_text(text: str) -> str:
    if not text: return ""
    text = to_english_digits(text)
    for short, full in TH_MONTHS.items():
        text = re.sub(short, full, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("—", "-").replace("–", "-").replace("：", ":")
    return text

def normalize_number(s: str) -> Optional[float]:
    if not s: return None
    s = to_english_digits(s)
    s = fix_numberlike_ocr(s)
    s = s.replace(" ", "").replace(",", "").replace("฿", "").replace("บาท", "").replace("%","")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None

def parse_date_candidates(text: str) -> Optional[str]:
    text = sanitize_text(text)
    candidates = set()
    for m in re.finditer(r"(วันที่|date)[:\s\-]*([^\n]{0,40})", text, flags=re.IGNORECASE):
        candidates.add(m.group(0))
    candidates.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text))
    candidates.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text))
    th_month_regex = r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    candidates.update(re.findall(rf"\b\d{{1,2}}\s*{th_month_regex}\s*\d{{2,4}}\b", text))
    parsed = []
    for c in list(candidates)[:30]:
        dt = dateparser.parse(c, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year > 2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    if not parsed: return None
    return sorted(parsed, key=lambda x: x.toordinal())[-1].isoformat()

# ---------- Preprocess
def preprocess(img_bgr: np.ndarray):
    out = {}
    out["original"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); out["grayscale"]=gray
    blur = cv2.medianBlur(gray, 3); out["denoise(median3)"]=blur
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    up = cv2.resize(th, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    out["upscale(1.5x)"]=up
    opened = cv2.morphologyEx(up, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations=1)
    out["morph_open"]=opened
    return out

# ---------- Tesseract
def ensure_tesseract(user_path: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    if pytesseract is None:
        return (False, None, "pytesseract not installed")
    candidates = []
    if user_path: candidates.append(user_path)
    candidates += [
        "/usr/bin/tesseract","/usr/local/bin/tesseract","/opt/homebrew/bin/tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]
    for c in [p for p in candidates if p]:
        if os.path.exists(c):
            try:
                pytesseract.pytesseract.tesseract_cmd = c
                _ = pytesseract.get_tesseract_version()
                return True, c, None
            except Exception as e:
                last_err = str(e)
    exe = shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd = exe
            _ = pytesseract.get_tesseract_version()
            return True, exe, None
        except Exception as e:
            return False, exe, f"Found at {exe} but error: {e}"
    return False, None, "tesseract not found"

def ocr_tesseract(img) -> str:
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6 -l tha+eng")

# ---------- Regex + Rule-based NER
def normalize_qt_code(raw: str) -> str:
    if not raw: return raw
    s = raw.strip().translate(str.maketrans({"$":"S","§":"S"}))
    s = re.sub(r"^[Kk]5", "KS", s); s = re.sub(r"^5", "S", s)
    s = fix_numberlike_ocr(s)
    return s.replace(" ","").upper()

def ner_extract_quotation_no(text: str) -> Optional[str]:
    t = sanitize_text(text); lines = t.splitlines()
    for i, l in enumerate(lines):
        if re.search(r"quotation\s*no\.?", l, re.IGNORECASE):
            after = re.split(r"quotation\s*no\.?\s*[:#]?\s*", l, flags=re.IGNORECASE)[-1]
            m_same = re.search(r"([A-Z\$\§]{0,5}[A-Z0-9\$\§/_\-.]{3,})", after, flags=re.IGNORECASE)
            if m_same and not re.fullmatch(r"date", m_same.group(1), flags=re.IGNORECASE):
                return normalize_qt_code(m_same.group(1))
            look = " ".join(lines[i+1:i+3])
            date_m = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", look)
            tokens = re.findall(r"[A-Z\$\§]?[A-Z0-9\$\§/_\-.]{3,}", look, flags=re.IGNORECASE)
            if tokens:
                if date_m:
                    cut = look[:date_m.start()]
                    cand = re.findall(r"[A-Z\$\§]?[A-Z0-9\$\§/_\-.]{3,}", cut, flags=re.IGNORECASE)
                    if cand: return normalize_qt_code(cand[-1])
                return normalize_qt_code(tokens[0])
    m = re.search(r"\b[A-Z\$\§]{1,4}[-/_.]?\d{2,4}[-/_.]?\d{1,6}\b", t, flags=re.IGNORECASE)
    if m: return normalize_qt_code(m.group(0))
    return None

def last_amount_in_line(line: str) -> Optional[float]:
    no_pct = re.sub(r"\d+(\.\d+)?\s*%","",line)
    no_pct = fix_numberlike_ocr(no_pct)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", no_pct)
    if not nums: return None
    for n in reversed(nums):
        v = normalize_number(n)
        if v is not None and ("," in n or "." in n or v >= 100):
            return v
    return normalize_number(nums[-1])

def amount_by_label(text: str, include: List[str], exclude: List[str]=[]) -> Optional[float]:
    t = sanitize_text(text)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    idxs = [i for i,l in enumerate(lines) if any(k.lower() in l.lower() for k in include) and not any(b.lower() in l.lower() for b in exclude)]
    if not idxs: return None
    i = idxs[-1]
    v = last_amount_in_line(lines[i])
    if v is None and i+1 < len(lines):
        v = last_amount_in_line(lines[i+1])
    return v

def extract_fields(full_text: str) -> Dict[str, Optional[str]]:
    txt = sanitize_text(full_text or "")
    vendor = None
    for l in txt.splitlines()[:25]:
        if re.search(r"(บริษัท|หจก\.|co\.,?\s*ltd\.?|company\s*limited)", l, re.IGNORECASE):
            vendor = l.strip(); break
    qt_no  = ner_extract_quotation_no(txt)
    date_iso = parse_date_candidates(txt)
    subtotal = amount_by_label(txt, include=["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","total"], exclude=["grand","vat"])
    vat      = amount_by_label(txt, include=["vat","ภาษีมูลค่าเพิ่ม"])
    grand    = amount_by_label(txt, include=["grand total","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"])
    if vat is not None and vat < 50 and re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", txt, re.IGNORECASE):
        if grand is not None and subtotal is not None: vat = round(grand - subtotal, 2)
        elif subtotal is not None: vat = round(subtotal * 0.07, 2)
    if grand is None and subtotal is not None and vat is not None: grand = round(subtotal + vat, 2)
    if subtotal is None and grand is not None and vat is not None: subtotal = round(grand - vat, 2)
    if vat is None and grand is not None and subtotal is not None: vat = round(grand - subtotal, 2)
    return {
        "Vendor/Supplier": vendor, "Quotation No.": qt_no, "Date": date_iso,
        "Description": None, "Subtotal": subtotal, "VAT": vat, "Grand Total": grand, "Raw Text": txt,
    }

# ---------- UI
st.set_page_config(page_title="OCR Quotation/Bill (Tesseract-only)", layout="wide")
st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง (Tesseract-only)")

with st.sidebar:
    st.header("⚙️ Settings")
    user_tess_path = st.text_input("Tesseract path (optional)", value="")
    show_steps = st.checkbox("Show preprocessing steps", value=True)
    st.markdown("---")
    st.caption("Diagnostics")
    st.code(str(os.environ.get("PYTHON_VERSION","")))
    st.code(str(os.environ.get("PATH",""))[:120] + "...")

ok, path, msg = ensure_tesseract(user_tess_path.strip() or None)
st.sidebar.write("Tesseract:", "✅ " + str(path) if ok else "❌ " + str(msg))

uploads = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"], accept_multiple_files=True)

rows = []
if uploads:
    for up in uploads:
        st.markdown("---")
        col1, col2 = st.columns([1,1])
        with col1:
            st.write("**File:**", up.name)
            image = Image.open(up).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            steps = preprocess(img_bgr)
            if show_steps:
                tabs = st.tabs(list(steps.keys()))
                for tab, key in zip(tabs, steps.keys()):
                    with tab:
                        st.image(steps[key], caption=key, use_container_width=True, clamp=len(steps[key].shape)==2)
        with col2:
            if not ok:
                st.error("Tesseract not found. Please set path in the sidebar or install via packages.txt")
                raw = ""
            else:
                try:
                    raw = ocr_tesseract(steps["morph_open"])
                except Exception as e:
                    st.exception(e); raw = ""
            st.text_area("Raw Text", value=raw, height=260)
            fields = extract_fields(raw)
            row = {
                "file": up.name,
                "Vendor / Supplier": fields["Vendor/Supplier"],
                "Quotation No.": fields["Quotation No."],
                "Date": fields["Date"],
                "Description": fields["Description"],
                "Subtotal": fields["Subtotal"],
                "VAT": fields["VAT"],
                "Grand Total": fields["Grand Total"],
            }
            rows.append(row)
            st.dataframe(pd.DataFrame([row]))

if rows:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df = pd.DataFrame(rows, columns=["file","Vendor / Supplier","Quotation No.","Date","Description","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="ocr_quotation_results.csv", mime="text/csv")