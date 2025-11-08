# app.py — OCR Quotation/Bill → Table (Hybrid EasyOCR + Tesseract)
# - แสดงขั้นตอน Image Pre-processing
# - Normalize ข้อความ/ตัวเลข แก้ o/O->0, I/l->1, B->8
# - Regex/NER จับ Vendor, Quotation No. (รองรับเคส Quotation No:  Date: <next line>),
#   Date, Subtotal, VAT, Grand Total (ข้ามเลขที่เป็น % และ reconcile ค่า)
# - ส่งออก Google Sheets ด้วย gspread (วางลิงก์ + อัปโหลด Service Account JSON)

import os, re, json, shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import dateparser

# ---------- OCR Engines (optional EasyOCR) ----------
EASYOCR_AVAILABLE = True
try:
    import easyocr  # ต้องใช้ torch
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
except Exception:
    pytesseract = None

# ---------- Text utils ----------
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}

def to_english_digits(s: str) -> str:
    return s.translate(THAI_DIGITS) if isinstance(s, str) else s

def fix_numberlike_ocr(s: str) -> str:
    """แก้ตัวที่มักอ่านผิดเฉพาะบริบทตัวเลข"""
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
    cands = set()
    for m in re.finditer(r"(วันที่|date)[:\s\-]*([^\n]{0,40})", text, flags=re.IGNORECASE):
        cands.add(m.group(0))
    cands.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text))
    cands.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text))
    th_month_regex = r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    cands.update(re.findall(rf"\b\d{{1,2}}\s*{th_month_regex}\s*\d{{2,4}}\b", text))
    parsed = []
    for c in list(cands)[:30]:
        dt = dateparser.parse(c, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year > 2400:  # ปี พ.ศ.
                dt = dt.replace(year=dt.year - 543)
            parsed.append(dt.date())
    if not parsed: return None
    return sorted(parsed)[-1].isoformat()

# ---------- Image preprocessing ----------
def deskew(binary_img: np.ndarray) -> Tuple[np.ndarray, float]:
    coords = np.column_stack(np.where(binary_img > 0))
    if coords.size == 0: return binary_img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = binary_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    out["original"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out["grayscale"] = gray
    blur = cv2.medianBlur(gray, 3)
    out["denoise(median3)"] = blur
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    out["adaptive_threshold"] = th
    de, ang = deskew(th)
    out["deskewed"] = de
    up = cv2.resize(de, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    out["upscale(1.5x)"] = up
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(up, cv2.MORPH_OPEN, kernel, iterations=1)
    out["morph_open"] = opened
    return out

# ---------- Tesseract helpers ----------
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
    config = "--oem 3 --psm 6 -l tha+eng"
    return pytesseract.image_to_string(img, config=config)

def ocr_easyocr(img_rgb) -> str:
    if not EASYOCR_AVAILABLE: return ""
    reader = easyocr.Reader(["th","en"], gpu=False)
    res = reader.readtext(img_rgb, detail=0, paragraph=True)
    return "\n".join(res)

def hybrid_text(img_rgb, img_bin, mode: str) -> str:
    """รวมผล EasyOCR+Tesseract ตามโหมด"""
    t_easy, t_tess = "", ""
    if mode in ["Hybrid (EasyOCR + Tesseract)", "EasyOCR only"] and EASYOCR_AVAILABLE:
        try: t_easy = ocr_easyocr(img_rgb)
        except Exception: t_easy = ""
    if mode in ["Hybrid (EasyOCR + Tesseract)", "Tesseract only"] and pytesseract is not None:
        try: t_tess = ocr_tesseract(img_bin)
        except Exception: t_tess = ""
    if mode == "EasyOCR only": return t_easy
    if mode == "Tesseract only": return t_tess
    # hybrid: รวมบรรทัดที่ยาวกว่า/ไม่ซ้ำ
    lines = set()
    for L in (t_easy.splitlines() + t_tess.splitlines()):
        L = L.strip()
        if L: lines.add(L)
    return "\n".join(sorted(lines, key=len))

# ---------- Regex + Rule-based NER ----------
def normalize_qt_code(raw: str) -> str:
    if not raw: return raw
    s = raw.strip().translate(str.maketrans({"$":"S","§":"S"}))
    s = re.sub(r"^[Kk]5", "KS", s)
    s = re.sub(r"^5", "S", s)
    s = fix_numberlike_ocr(s)
    return s.replace(" ","").upper()

def ner_extract_quotation_no(text: str) -> Optional[str]:
    """
    ครอบคลุมกรณี:
    - 'Quotation No: KS2209191'
    - 'Quotation No: Date:' (ค่าอยู่บรรทัดถัดไป) เช่น:
        Quotation No:  Date:
        KS2209191     19/09/2022
    - 'Quotation No. KS2209191 Date ...'
    """
    t = sanitize_text(text)
    lines = [l for l in t.splitlines()]
    for i, l in enumerate(lines):
        if re.search(r"quotation\s*no\.?", l, flags=re.IGNORECASE):
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
    # ตัดเปอร์เซ็นต์ เช่น "7%" ออกก่อน
    no_pct = re.sub(r"\d+(\.\d+)?\s*%","",line)
    no_pct = fix_numberlike_ocr(no_pct)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", no_pct)
    if not nums: return None
    # เลือก "ตัวเงิน" ที่น่าเชื่อถือ: มี comma/จุด หรือ >= 100
    for n in reversed(nums):
        v = normalize_number(n)
        if v is not None and ("," in n or "." in n or v >= 100):
            return v
    return normalize_number(nums[-1])

def amount_by_label(text: str, include: List[str], exclude: List[str]=[]) -> Optional[float]:
    """
    หาเลขในบรรทัดที่มีคีย์เวิร์ด (หรือบรรทัดถัดไป) — เลือก match สุดท้ายในเอกสาร (โซนท้าย)
    """
    t = sanitize_text(text)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    idxs = []
    for i, l in enumerate(lines):
        L = l.lower()
        if any(k.lower() in L for k in include) and not any(b.lower() in L for b in exclude):
            idxs.append(i)
    if not idxs: return None
    i = idxs[-1]
    v = last_amount_in_line(lines[i])
    if v is None and i+1 < len(lines):
        v = last_amount_in_line(lines[i+1])
    return v

def extract_vendor(text: str) -> Optional[str]:
    t = sanitize_text(text)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    company_keywords = [
        r"(บริษัท.+จำกัด|บริษัท.+จำกัด\(มหาชน\)|ห้างหุ้นส่วนจำกัด|หจก\.)",
        r"(co\.,?\s*ltd\.?|company\s*limited|limited\s*company)",
        r"(vendor|supplier|ผู้ขาย|ผู้จำหน่าย)[:\s]*([^\n]+)"
    ]
    for l in lines[:30]:
        for pat in company_keywords:
            m = re.search(pat, l, flags=re.IGNORECASE)
            if m:
                cand = m.group(m.lastindex) if (m.lastindex and m.group(m.lastindex)) else l
                cand = re.sub(r"^(vendor|supplier|ผู้ขาย|ผู้จำหน่าย)[:\s\-]*", "", cand, flags=re.IGNORECASE)
                cand = re.sub(r"\s{2,}", " ", cand)
                cand = re.sub(r"(ที่อยู่|address)[:\s].*$", "", cand, flags=re.IGNORECASE)
                return cand.strip()
    for l in lines[:10]:
        if re.search(r"(บริษัท|หจก\.|co\.,?\s*ltd\.?)", l, flags=re.IGNORECASE):
            return l
    return None

def extract_fields(full_text: str) -> Dict[str, Optional[str]]:
    txt = sanitize_text(full_text or "")
    vendor = extract_vendor(txt)
    qt_no  = ner_extract_quotation_no(txt)
    date_iso = parse_date_candidates(txt)

    subtotal = amount_by_label(txt, include=["subtotal", "รวมก่อนภาษี", "ยอดก่อนภาษี", "total"], exclude=["grand", "vat"])
    vat      = amount_by_label(txt, include=["vat", "ภาษีมูลค่าเพิ่ม"])
    grand    = amount_by_label(txt, include=["grand total", "ยอดรวมสุทธิ", "รวมทั้งสิ้น", "ยอดชำระสุทธิ"])

    # สมมติถ้าเห็น "VAT 7%" แต่ไม่มีจำนวน → คำนวณจาก grand/subtotal
    if vat is not None and vat < 50 and re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", txt, flags=re.IGNORECASE):
        if grand is not None and subtotal is not None:
            vat = round(grand - subtotal, 2)
        elif subtotal is not None:
            vat = round(subtotal * 0.07, 2)
    # เติมค่าที่หาย
    if grand is None and subtotal is not None and vat is not None:
        grand = round(subtotal + vat, 2)
    if subtotal is None and grand is not None and vat is not None:
        subtotal = round(grand - vat, 2)
    if vat is None and grand is not None and subtotal is not None:
        vat = round(grand - subtotal, 2)

    return {
        "Vendor / Supplier": vendor,
        "Quotation No.": qt_no,
        "Date": date_iso,
        "Subtotal": subtotal,
        "VAT": vat,
        "Grand Total": grand,
    }

# ---------- Google Sheets export ----------
def export_to_google_sheets(df: pd.DataFrame, sheet_url: str, service_json: dict, worksheet_name: str = "OCR_QT"):
    try:
        import gspread
        gc = gspread.service_account_from_dict(service_json)
        sh = gc.open_by_url(sheet_url)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        if not ws.get_all_values():
            ws.append_row(list(df.columns))
        for _, row in df.iterrows():
            ws.append_row([str(x) if x is not None else "" for x in row.tolist()])
        return True, "Exported to Google Sheets successfully."
    except Exception as e:
        return False, f"Export failed: {e}"

# ---------- UI ----------
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")
st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง")

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    engine = st.selectbox("OCR Engine", ["Hybrid (EasyOCR + Tesseract)","Tesseract only","EasyOCR only"], index=0)
    user_tess_path = st.text_input("Tesseract path (ถ้าระบบหาไม่เจอ)", value="")
    show_steps = st.checkbox("แสดงภาพ Pre-processing ทุกขั้นตอน", value=True)
    worksheet_name = st.text_input("Worksheet name (Google Sheets)", value="OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("วางลิงก์ Google ชีท (ต้องแชร์สิทธิ์แก้ไขให้ Service Account)")
    service_json_file = st.file_uploader("อัปโหลดไฟล์ Service Account JSON", type=["json"], accept_multiple_files=False)

TESS_OK, TESS_PATH, TESS_MSG = ensure_tesseract(user_tess_path.strip() or None)
st.sidebar.markdown("**Tesseract:** " + ("✅ " + str(TESS_PATH) if TESS_OK else "❌ " + str(TESS_MSG)))

st.subheader("อัปโหลดรูปใบเสนอราคา/บิล (JPG/PNG)")
uploads = st.file_uploader("Drag & drop หรือ Browse files", type=["jpg","jpeg","png"], accept_multiple_files=True)

rows = []
if uploads:
    for up in uploads:
        st.markdown("---")
        c1, c2 = st.columns([1,1])
        with c1:
            st.write(f"**ไฟล์:** {up.name}")
            image = Image.open(up).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            steps = preprocess(img_bgr)
            if show_steps:
                tabs = st.tabs(list(steps.keys()))
                for tab, key in zip(tabs, steps.keys()):
                    with tab:
                        # ใช้ use_column_width เพื่อรองรับหลายเวอร์ชัน
                        st.image(steps[key], caption=key, use_column_width=True)
        with c2:
            raw = ""
            if engine == "EasyOCR only" and not EASYOCR_AVAILABLE:
                st.warning("EasyOCR ไม่พร้อมใช้งานในสภาพแวดล้อมนี้ → จะใช้ Tesseract ชั่วคราว")
            if engine in ["Hybrid (EasyOCR + Tesseract)","Tesseract only"] and not TESS_OK:
                st.info("Tesseract ไม่พร้อมใช้งาน → ใช้ EasyOCR ชั่วคราว หรือกรอก path ในแถบซ้าย")

            raw = hybrid_text(steps["original"], steps["morph_open"], engine)
            st.text_area("OCR Output (Raw Text)", value=raw, height=260)

            fields = extract_fields(raw or "")
            row = {
                "file": up.name,
                "Vendor / Supplier": fields["Vendor / Supplier"],
                "Quotation No.": fields["Quotation No."],
                "Date": fields["Date"],
                "Subtotal": fields["Subtotal"],
                "VAT": fields["VAT"],
                "Grand Total": fields["Grand Total"],
            }
            rows.append(row)
            st.write("**สรุปฟิลด์ที่สกัดได้**")
            st.dataframe(pd.DataFrame([row]))

if rows:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df = pd.DataFrame(rows, columns=["file","Vendor / Supplier","Quotation No.","Date","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลด CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ocr_quotation_results.csv", mime="text/csv")
    if sheet_url and service_json_file is not None:
        try:
            service_dict = json.load(service_json_file)
        except Exception as e:
            st.error(f"อ่าน Service JSON ไม่ได้: {e}")
            service_dict = None
        if service_dict is not None and st.button("🚀 โอนข้อมูลขึ้น Google ชีท"):
            ok, msg = export_to_google_sheets(df, sheet_url, service_dict, worksheet_name=worksheet_name)
            (st.success if ok else st.error)(msg)
