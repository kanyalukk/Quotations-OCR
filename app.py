# app.py (patched)
# Streamlit OCR (Thai+English) for Quotation/Bill -> Extract Key Fields + Google Sheets Export
# Adds robust Tesseract detection, user-set path, and graceful fallback if missing.

import io
import os
import re
import json
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --- Optional OCR engines ---
EASYOCR_AVAILABLE = True
try:
    import easyocr  # requires torch
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
except Exception as e:
    pytesseract = None

import cv2
import dateparser
import shutil

# -------------------- Utilities --------------------

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}

def to_english_digits(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.translate(THAI_DIGITS)

def normalize_number(s: str) -> Optional[float]:
    if not s:
        return None
    s = to_english_digits(s)
    s = s.replace(" ", "")
    s = s.replace(",", "")
    s = s.replace("฿", "").replace("บาท", "").replace("฿.", "")
    try:
        return float(re.findall(r"-?\d+(?:\.\d+)?", s)[0])
    except Exception:
        return None

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = to_english_digits(text)
    for short, full in TH_MONTHS.items():
        text = re.sub(short, full, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("—", "-").replace("–", "-").replace("：", ":")
    return text

def parse_date_candidates(text: str) -> Optional[str]:
    text = sanitize_text(text)
    candidates = set()

    for m in re.finditer(r"(วันที่|date)[:\s\-]*([^\n]{0,30})", text, flags=re.IGNORECASE):
        nearby = m.group(0)
        candidates.add(nearby)
    candidates.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text))
    candidates.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text))

    th_month_regex = r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    candidates.update(re.findall(rf"\b\d{{1,2}}\s*{th_month_regex}\s*\d{{2,4}}\b", text))

    parsed = []
    for c in list(candidates)[:15]:
        dt = dateparser.parse(c, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            y = dt.year
            if y > 2400:
                y -= 543
                dt = dt.replace(year=y)
            parsed.append(dt.date())

    if not parsed: return None
    best = sorted(parsed, key=lambda x: x.toordinal())[-1]
    return best.isoformat()

def deskew(binary_img: np.ndarray) -> Tuple[np.ndarray, float]:
    coords = np.column_stack(np.where(binary_img > 0))
    if coords.size == 0:
        return binary_img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = binary_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(binary_img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    out["original"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out["grayscale"] = gray
    blur = cv2.medianBlur(gray, 3)
    out["denoise(median3)"] = blur
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    out["adaptive_threshold"] = th
    de, ang = deskew(th)
    out["deskewed"] = de
    up = cv2.resize(de, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    out["upscale(1.5x)"] = up
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(up, cv2.MORPH_OPEN, kernel, iterations=1)
    out["morph_open"] = opened
    return out

def safe_get_tesseract_path(user_path: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns (ok, exe_path, error_msg). Tries user path, common locations, and PATH.
    """
    if pytesseract is None:
        return (False, None, "pytesseract module not installed")
    import shutil
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates += [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    ]
    last_err = None
    for c in candidates:
        if os.path.exists(c):
            try:
                pytesseract.pytesseract.tesseract_cmd = c
                _ = pytesseract.get_tesseract_version()
                return (True, c, None)
            except Exception as e:
                last_err = str(e)
    exe = shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd = exe
            _ = pytesseract.get_tesseract_version()
            return (True, exe, None)
        except Exception as e:
            return (False, exe, f"Found at {exe} but not runnable: {e}")
    return (False, None, last_err or "tesseract executable not found on this system")

def ocr_easyocr(img_rgb: np.ndarray) -> str:
    if not EASYOCR_AVAILABLE:
        return ""
    reader = easyocr.Reader(["th", "en"], gpu=False)
    results = reader.readtext(img_rgb, detail=0, paragraph=True)
    return "\n".join(results)

def ocr_tesseract(img_rgb_or_binary: np.ndarray) -> str:
    config = "--oem 3 --psm 6 -l tha+eng"
    return pytesseract.image_to_string(img_rgb_or_binary, config=config)

AMOUNT_PAT = r"(?P<amt>-?\d[\d,\s]*\.?\d{0,2})"
def find_amount_near(text: str, keys: List[str]) -> Optional[float]:
    text = sanitize_text(text).lower()
    for key in keys:
        pat = rf"{key}[^0-9\-]*{AMOUNT_PAT}|{AMOUNT_PAT}[^0-9\-]*{key}"
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            amt = m.group("amt") if "amt" in m.groupdict() and m.group("amt") else (m.group(1) if m.group(1) else None)
            if amt: return normalize_number(amt)
    nums = re.findall(AMOUNT_PAT, text)
    if nums: return normalize_number(nums[-1])
    return None

def extract_vendor(text: str) -> Optional[str]:
    text = sanitize_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    company_keywords = [
        r"(บริษัท.+จำกัด|บริษัท.+จำกัด\(มหาชน\)|ห้างหุ้นส่วนจำกัด|หจก\.)",
        r"(co\.,?\s*ltd\.?|company\s*limited|limited\s*company)",
        r"(vendor|supplier|ผู้ขาย|ผู้จำหน่าย)[:\s]*([^\n]+)"
    ]
    for l in lines[:30]:
        for pat in company_keywords:
            m = re.search(pat, l, flags=re.IGNORECASE)
            if m:
                if m.lastindex and m.group(m.lastindex):
                    cand = m.group(m.lastindex)
                else:
                    cand = l
                cand = re.sub(r"^(vendor|supplier|ผู้ขาย|ผู้จำหน่าย)[:\s\-]*", "", cand, flags=re.IGNORECASE)
                cand = re.sub(r"\s{2,}", " ", cand)
                cand = re.sub(r"(ที่อยู่|address)[:\s].*$", "", cand, flags=re.IGNORECASE)
                return cand.strip()
    for l in lines[:10]:
        if re.search(r"(บริษัท|หจก\.|co\.,?\s*ltd\.?)", l, flags=re.IGNORECASE):
            return l
    return None

def extract_qt_no(text: str) -> Optional[str]:
    text = sanitize_text(text)
    pat = r"(quotation\s*no\.?|quotation\s*#|qt\s*no\.?|เลขที่ใบเสนอราคา|เลขที่[:\s]|ref\s*no\.?)\s*[:#]?\s*([A-Za-z0-9\/\-\._]{3,})"
    m = re.search(pat, text, flags=re.IGNORECASE)
    if m: return m.group(2)
    m = re.search(r"\b[A-Z]{1,4}[-/_.]?\d{2,4}[-/_.]?\d{1,6}\b", text)
    if m: return m.group(0)
    return None

def extract_description(text: str) -> Optional[str]:
    text = sanitize_text(text)
    m = re.search(r"(description|รายละเอียด|รายการ)\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE|re.DOTALL)
    if m:
        tail = m.group(2).strip()
        tail = re.split(r"(subtotal|รวมก่อนภาษี|vat|ภาษีมูลค่าเพิ่ม|grand\s*total|ยอดรวมสุทธิ)", tail, flags=re.IGNORECASE)[0]
        return tail.strip().splitlines()[0][:180]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    body = []
    for l in lines:
        if len(l) > 20 and not re.search(r"(tel|fax|email|เลขประจำตัวผู้เสียภาษี|tax\s*id|address|ที่อยู่)", l, flags=re.IGNORECASE):
            body.append(l)
        if len(body) >= 2: break
    return " | ".join(body)[:180] if body else None

def extract_fields(full_text: str) -> Dict[str, Optional[str]]:
    txt = sanitize_text(full_text)
    vendor = extract_vendor(txt)
    qt_no  = extract_qt_no(txt)
    date_iso = parse_date_candidates(txt)
    subtotal = find_amount_near(txt, ["subtotal", "รวมก่อนภาษี", "ยอดก่อนภาษี"])
    vat      = find_amount_near(txt, ["vat", "ภาษีมูลค่าเพิ่ม", "vat 7", "ภาษี 7"])
    grand    = find_amount_near(txt, ["grand total", "ยอดรวมสุทธิ", "net total", "รวมทั้งสิ้น", "ยอดชำระสุทธิ"])
    if grand is not None and subtotal is not None and vat is None:
        vat = round(grand - subtotal, 2)
    if vat is not None and subtotal is None and grand is not None:
        subtotal = round(grand - vat, 2)
    desc = extract_description(txt)
    return {
        "Vendor/Supplier": vendor,
        "Quotation No.": qt_no,
        "Date": date_iso,
        "Description": desc,
        "Subtotal": subtotal,
        "VAT": vat,
        "Grand Total": grand,
        "Raw Text": txt,
    }

def export_to_google_sheets(df: pd.DataFrame, sheet_url: str, service_json: dict, worksheet_name: str = "OCR_QT"):
    try:
        import gspread
        gc = gspread.service_account_from_dict(service_json)
        sh = gc.open_by_url(sheet_url)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        header = list(df.columns)
        existing = ws.get_all_values()
        if not existing:
            ws.append_row(header)
        for _, row in df.iterrows():
            ws.append_row([str(x) if x is not None else "" for x in row.tolist()])
        return True, "Exported to Google Sheets successfully."
    except Exception as e:
        return False, f"Export failed: {e}"

st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")
st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง")
st.caption("อัปโหลด JPG/PNG (ไทย+อังกฤษ) → แสดง Pre-processing → OCR (EasyOCR/Tesseract) → Regex/NER แยกข้อมูลสำคัญ → ส่งออก Google ชีท (ตัวเลือก)")

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    engine = st.selectbox("OCR Engine", ["Hybrid (EasyOCR ➜ fallback Tesseract)","Tesseract only","EasyOCR only"], index=0)
    user_tess_path = st.text_input("Tesseract path (ถ้าระบบหาไม่เจอ)", value="")
    show_steps = st.checkbox("แสดงภาพ Pre-processing ทุกขั้นตอน", value=True)
    worksheet_name = st.text_input("Worksheet name (Google Sheets)", value="OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("วางลิงก์ Google ชีท ที่มีสิทธิ์แก้ไขได้", value="")
    service_json_file = st.file_uploader("อัปโหลดไฟล์ Service Account JSON", type=["json"], accept_multiple_files=False)

# Probe Tesseract availability once (and show status)
TESSERACT_OK = False
tess_path_info = ""
if pytesseract is not None:
    ok, exe, err = safe_get_tesseract_path(user_tess_path.strip() or None)
    TESSERACT_OK = ok
    tess_path_info = exe if exe else (err or "not found")
else:
    tess_path_info = "pytesseract module not installed"

st.sidebar.markdown("**Tesseract status:** " + ("✅ " + tess_path_info if TESSERACT_OK else "❌ " + tess_path_info))

st.subheader("อัปโหลดรูปใบเสนอราคา/บิล (JPG/PNG)")
uploads = st.file_uploader("Drag & drop หรือ Browse files", type=["jpg","jpeg","png"], accept_multiple_files=True)

results = []
if uploads:
    for up in uploads:
        st.markdown("---")
        colL, colR = st.columns([1,1])
        with colL:
            st.write(f"**ไฟล์:** {up.name}")
            image = Image.open(up).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            steps = preprocess(img_bgr)
            if show_steps:
                tabs = st.tabs(list(steps.keys()))
                for tab, key in zip(tabs, steps.keys()):
                    with tab:
                        if len(steps[key].shape) == 2:
                            st.image(steps[key], caption=key, use_container_width=True, clamp=True)
                        else:
                            st.image(steps[key], caption=key, use_container_width=True)

        with colR:
            st.write("**OCR Output (Raw Text)**")
            text_easy = ""
            text_tess = ""

            if engine in ["Hybrid (EasyOCR ➜ fallback Tesseract)","EasyOCR only"] and EASYOCR_AVAILABLE:
                try:
                    text_easy = ocr_easyocr(steps["original"])
                except Exception as e:
                    st.warning(f"EasyOCR error: {e}")

            if engine in ["Hybrid (EasyOCR ➜ fallback Tesseract)","Tesseract only"] and TESSERACT_OK:
                try:
                    text_tess = ocr_tesseract(steps["morph_open"])
                except Exception as e:
                    st.warning(f"Tesseract error: {e}")
            elif engine in ["Hybrid (EasyOCR ➜ fallback Tesseract)","Tesseract only"] and not TESSERACT_OK:
                st.info("Tesseract not found. กรณีนี้ให้ใช้ EasyOCR เท่านั้น หรือ ติดตั้ง Tesseract ตาม README.")

            raw_combined = ""
            if engine == "EasyOCR only":
                raw_combined = text_easy
            elif engine == "Tesseract only":
                raw_combined = text_tess
            else:
                raw_combined = text_easy if len(text_easy) >= len(text_tess) else text_tess
                if not raw_combined and text_easy:
                    raw_combined = text_easy

            st.text_area("Raw Text", value=raw_combined, height=250)

            fields = extract_fields(raw_combined or "")
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
            results.append(row)
            st.write("**สรุปฟิลด์ที่สกัดได้**")
            st.dataframe(pd.DataFrame([row]))

if results:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df = pd.DataFrame(results, columns=["file","Vendor / Supplier","Quotation No.","Date","Description","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ ดาวน์โหลด CSV", data=csv_bytes, file_name="ocr_quotation_results.csv", mime="text/csv")
    if sheet_url and service_json_file is not None:
        try:
            service_dict = json.load(service_json_file)
        except Exception as e:
            st.error(f"ไม่สามารถอ่านไฟล์ JSON: {e}")
            service_dict = None
        if service_dict is not None:
            if st.button("🚀 โอนข้อมูลขึ้น Google ชีท"):
                ok, msg = export_to_google_sheets(df, sheet_url, service_dict, worksheet_name=worksheet_name)
                (st.success if ok else st.error)(msg)

st.markdown("---")
with st.expander("ℹ️ Tips / Notes"):
    st.markdown("""
- ถ้า EasyOCR ติดตั้งไม่ได้ ให้เลือก **Tesseract only** (ต้องติดตั้ง tesseract บนระบบก่อน)
- ถ้าเจอ `TesseractNotFoundError` ให้พิมพ์ path ในช่อง **Tesseract path** ด้านซ้าย เช่น:
  - macOS (brew): `/opt/homebrew/bin/tesseract` หรือ `/usr/local/bin/tesseract`
  - Ubuntu/Cloud: `/usr/bin/tesseract`
  - Windows: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
- รูปเอกสารเอียง/มืด: ระบบมี **de-skew + adaptive threshold + morphology** ให้แล้ว
- ปรับ Regex เพิ่มคำสำคัญได้ที่ฟังก์ชัน `extract_*`
""")