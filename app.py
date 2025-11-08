# ต้องอยู่บรรทัดแรกเพื่อกัน SessionInfo error
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import os, re, json, shutil, io
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import dateparser
import fitz  # PyMuPDF

# OCR (Tesseract)
try:
    import pytesseract
except Exception:
    pytesseract = None

# ──────────────────────────────────────────────────────────────
# Basic text helpers
# ──────────────────────────────────────────────────────────────
TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}
def to_en_digits(s: str) -> str:
    return s.translate(TH_DIGITS) if isinstance(s, str) else s

def fix_numberlike_ocr(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub(r'(?<=\d)[oO](?=[\d,\.])', '0', s)
    s = re.sub(r'(?<=[,\.\s])[oO](?=\d)', '0', s)
    s = re.sub(r'(?<=\d)[lI](?=[\d,\.])', '1', s)
    s = re.sub(r'(?<=\d)B(?=[\d,\.])', '8', s)
    return s

def sanitize_text(t: str) -> str:
    if not t: return ""
    t = to_en_digits(t)
    for k, v in TH_MONTHS.items():
        t = re.sub(k, v, t)
    t = re.sub(r"[ \t]+", " ", t).replace("—","-").replace("–","-").replace("：",":")
    return t

def normalize_number(s: str) -> Optional[float]:
    if not s: return None
    s = fix_numberlike_ocr(to_en_digits(s))
    s = s.replace(" ", "").replace(",", "").replace("฿", "").replace("บาท", "").replace("%","")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None

def parse_date_candidates(text: str) -> Optional[str]:
    t = sanitize_text(text)
    cands = set()
    cands.update(re.findall(r"(?:วันที่|date)[:\-\s]*([^\n]{1,40})", t, flags=re.I))
    cands.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t))
    cands.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", t))
    th = r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    cands.update(re.findall(rf"\b\d{{1,2}}\s*{th}\s*\d{{2,4}}\b", t))
    parsed = []
    for c in list(cands)[:30]:
        dt = dateparser.parse(c, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year > 2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    return sorted(parsed)[-1].isoformat() if parsed else None

# ──────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────
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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); out["grayscale"] = gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray); out["clahe"] = clahe
    blur = cv2.medianBlur(clahe, 3); out["denoise(median3)"] = blur
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9); out["adaptive_threshold"] = th
    de, _ = deskew(th); out["deskewed"] = de
    up = cv2.resize(de, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC); out["upscale(1.7x)"] = up
    opened = cv2.morphologyEx(up, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations=1); out["morph_open"] = opened
    return out

# ──────────────────────────────────────────────────────────────
# Tesseract helpers
# ──────────────────────────────────────────────────────────────
def ensure_tesseract(user_path: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    if pytesseract is None:
        return (False, None, "pytesseract not installed")
    cand = []
    if user_path: cand.append(user_path)
    cand += [
        "/usr/bin/tesseract","/usr/local/bin/tesseract","/opt/homebrew/bin/tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]
    for p in cand:
        if os.path.exists(p):
            try:
                pytesseract.pytesseract.tesseract_cmd = p
                _ = pytesseract.get_tesseract_version()
                return True, p, None
            except Exception:
                pass
    exe = shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd = exe
            _ = pytesseract.get_tesseract_version()
            return True, exe, None
        except Exception as e:
            return False, exe, f"Found {exe} but error: {e}"
    return False, None, "tesseract not found"

def tesseract_text(img, psm=6, lang="tha+eng", whitelist=None) -> str:
    cfg = f"--oem 3 --psm {psm} -l {lang}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, config=cfg)

def tesseract_data(img, psm=6, lang="tha+eng") -> pd.DataFrame:
    cfg = f"--oem 3 --psm {psm} -l {lang}"
    df = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    df["text_norm"] = df["text"].str.lower().str.replace(r"[^a-z0-9ก-๙]+","", regex=True)
    return df

# ──────────────────────────────────────────────────────────────
# ROI finder with image_to_data (ไม่ต้องพึ่ง EasyOCR)
# ──────────────────────────────────────────────────────────────
def right_roi(img_rgb: np.ndarray, box: Tuple[int,int,int,int], w_ratio=0.45, pad=10):
    h, w = img_rgb.shape[:2]
    x, y, b_w, b_h = box
    x1 = min(w-1, x + b_w + pad)
    y1 = max(0, y - 4)
    x2 = min(w, int(x1 + w*w_ratio))
    y2 = min(h, y + b_h + 4)
    if x2 <= x1 or y2 <= y1:
        return img_rgb[0:1,0:1]
    return img_rgb[y1:y2, x1:x2]

def locate_first(df: pd.DataFrame, keywords: List[str]) -> Optional[pd.Series]:
    keys = [re.sub(r"[^a-z0-9ก-๙]+","", k.lower()) for k in keywords]
    cand = df[df["text_norm"].isin(keys)]
    if not cand.empty:
        return cand.sort_values(["page_num","top","left"]).iloc[0]
    # fuzzy contains (สั้น ๆ)
    for k in keys:
        sub = df[df["text_norm"].str.contains(k, na=False)]
        if not sub.empty:
            return sub.sort_values(["page_num","top","left"]).iloc[0]
    return None

def extract_with_guidance(img_rgb: np.ndarray, steps: Dict[str,np.ndarray]) -> Dict[str, Optional[str]]:
    # ใช้ tesseract data หา label → อ่าน ROI ขวา ด้วย whitelist
    df = tesseract_data(steps["original"], psm=6, lang="tha+eng")

    # Quotation No.
    qrow = locate_first(df, ["quotation","quotationno","quotationno."])
    qt = None
    dt = None
    if qrow is not None:
        roi = right_roi(steps["original"], (qrow["left"], qrow["top"], qrow["width"], qrow["height"]), w_ratio=0.55)
        qt_txt = tesseract_text(roi, psm=6, lang="eng",
                                whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.,")
        qt_txt = re.split(r"\bdate\b", qt_txt, flags=re.I)[0]
        qt_txt = re.sub(r"[^A-Za-z0-9/_\-.]+","", qt_txt).strip()
        if qt_txt: qt = qt_txt
        # date ใน ROI เดียวกัน
        dt = parse_date_candidates(tesseract_text(roi, psm=6, lang="tha+eng"))

    # Date (สำรอง)
    if dt is None:
        drow = locate_first(df, ["date","วันที่"])
        if drow is not None:
            roi = right_roi(steps["original"], (drow["left"], drow["top"], drow["width"], drow["height"]), w_ratio=0.35)
            dt = parse_date_candidates(tesseract_text(roi, psm=7, lang="tha+eng"))

    # เงิน: Subtotal / VAT / Grand Total
    subtotal = vat = grand = None
    # Grand Total
    grow = locate_first(df, ["grand","grandtotal","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"])
    if grow is not None:
        roi = right_roi(steps["original"], (grow["left"], grow["top"], grow["width"], grow["height"]), w_ratio=0.38)
        g_txt = tesseract_text(roi, psm=7, lang="eng", whitelist="0123456789.,")
        grand = normalize_number(g_txt)

    # VAT
    vrow = locate_first(df, ["vat","ภาษีมูลค่าเพิ่ม"])
    if vrow is not None:
        roi = right_roi(steps["original"], (vrow["left"], vrow["top"], vrow["width"], vrow["height"]), w_ratio=0.32)
        v_txt = tesseract_text(roi, psm=7, lang="eng", whitelist="0123456789.,")
        vat = normalize_number(v_txt)

    # Subtotal: มักเป็น "Total" บรรทัดเหนือ VAT
    srow = locate_first(df, ["total","รวมก่อนภาษี","ยอดก่อนภาษี","subtotal"])
    if srow is not None:
        roi = right_roi(steps["original"], (srow["left"], srow["top"], srow["width"], srow["height"]), w_ratio=0.32)
        s_txt = tesseract_text(roi, psm=7, lang="eng", whitelist="0123456789.,")
        subtotal = normalize_number(s_txt)

    return {
        "Quotation No.": qt,
        "Date": dt,
        "Subtotal": subtotal,
        "VAT": vat,
        "Grand Total": grand
    }

# ──────────────────────────────────────────────────────────────
# Fallback (regex จาก Raw)
# ──────────────────────────────────────────────────────────────
def fallback_from_raw(raw: str) -> Dict[str, Optional[str]]:
    t = sanitize_text(raw or "")
    # Quotation
    qt = None
    m = re.search(r"(?:quotation\s*no\.?[:\s]*)([A-Z0-9/_\-.]{5,})", t, flags=re.I)
    if m: qt = m.group(1).strip()
    # Date
    dt = parse_date_candidates(t)
    # Amounts
    def amount_by_label(include, exclude=[]):
        lines = [L.strip() for L in t.splitlines() if L.strip()]
        cand = [i for i,L in enumerate(lines) if any(k.lower() in L.lower() for k in include) and not any(e.lower() in L.lower() for e in exclude)]
        if not cand: return None
        i = cand[-1]
        def pick(L):
            no_pct = re.sub(r"\d+(\.\d+)?\s*%","", L)
            no_pct = fix_numberlike_ocr(no_pct)
            nums = re.findall(r"-?\d[\d,]*\.?\d*", no_pct)
            if not nums: return None
            for n in reversed(nums):
                v = normalize_number(n)
                if v is not None and ("," in n or "." in n or v >= 100):
                    return v
            return normalize_number(nums[-1])
        val = pick(lines[i]) or (pick(lines[i+1]) if i+1 < len(lines) else None)
        return val
    subtotal = amount_by_label(["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","total"], exclude=["grand","vat"])
    vat      = amount_by_label(["vat","ภาษีมูลค่าเพิ่ม"])
    grand    = amount_by_label(["grand total","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"])
    # reconcile
    if vat is not None and vat < 50 and re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", t, flags=re.I):
        if grand is not None and subtotal is not None: vat = round(grand - subtotal, 2)
        elif subtotal is not None: vat = round(subtotal * 0.07, 2)
    if grand is None and subtotal is not None and vat is not None: grand = round(subtotal + vat, 2)
    if subtotal is None and grand is not None and vat is not None: subtotal = round(grand - vat, 2)
    if vat is None and grand is not None and subtotal is not None: vat = round(grand - subtotal, 2)
    return {"Quotation No.": qt, "Date": dt, "Subtotal": subtotal, "VAT": vat, "Grand Total": grand}

def extract_vendor(raw: str) -> Optional[str]:
    t = sanitize_text(raw)
    for L in t.splitlines()[:25]:
        if re.search(r"(บริษัท.+จำกัด|บริษัท.+\(มหาชน\).+จำกัด|ห้างหุ้นส่วนจำกัด|หจก\.)", L):
            return L.strip()
        if re.search(r"co\.,?\s*ltd\.?|company\s*limited", L, flags=re.I):
            return L.strip()
    return None

# ──────────────────────────────────────────────────────────────
# Google Sheets
# ──────────────────────────────────────────────────────────────
def export_to_google_sheets(df: pd.DataFrame, sheet_url: str, service_json: dict, worksheet_name: str="OCR_QT"):
    try:
        import gspread
        gc = gspread.service_account_from_dict(service_json)
        sh = gc.open_by_url(sheet_url)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        if not ws.get_all_values(): ws.append_row(list(df.columns))
        for _, row in df.iterrows():
            ws.append_row([("" if v is None else str(v)) for v in row.tolist()])
        return True, "Exported to Google Sheets successfully."
    except Exception as e:
        return False, f"Export failed: {e}"

# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    tess_path = st.text_input("Tesseract path (ถ้าระบบหาไม่เจอ)", "")
    show_steps = st.checkbox("แสดงภาพ Pre-processing", True)
    worksheet_name = st.text_input("Worksheet name (Google Sheets)", "OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("ลิงก์ Google ชีท (แชร์สิทธิ์แก้ไขให้ Service Account)")
    service_json_file = st.file_uploader("อัปโหลด Service Account JSON", type=["json"], accept_multiple_files=False)

TESS_OK, TESS_PATH, TESS_MSG = ensure_tesseract(tess_path.strip() or None)
st.sidebar.markdown("**Tesseract:** " + ("✅ " + str(TESS_PATH) if TESS_OK else "❌ " + str(TESS_MSG)))

st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง (Guided ROI + Regex/NER)")
uploads = st.file_uploader("อัปโหลด JPG/PNG/PDF", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)

rows = []
if uploads:
    for up in uploads:
        st.markdown("---")
        st.write(f"**ไฟล์:** {up.name}")

        # Load as images
        if up.type == "application/pdf" or up.name.lower().endswith(".pdf"):
            with fitz.open(stream=up.read(), filetype="pdf") as doc:
                pages = []
                for p in doc:
                    pix = p.get_pixmap(dpi=300, alpha=False)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                    pages.append(img[:,:,::-1])  # BGR
        else:
            im = Image.open(up).convert("RGB")
            pages = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)]

        for pi, img_bgr in enumerate(pages, start=1):
            steps = preprocess(img_bgr)
            col1, col2 = st.columns([1,1])
            with col1:
                if show_steps:
                    tabs = st.tabs(list(steps.keys()))
                    for tb, k in zip(tabs, steps.keys()):
                        with tb: st.image(steps[k], caption=f"{k} (page {pi})", use_column_width=True)

            # Raw text (เพื่อ debug และ fallback)
            raw = ""
            if TESS_OK:
                try:
                    # ลองหลาย psm แล้วเอายาวสุด
                    cand = []
                    for psm in (6,4,11,12,3):
                        cand.append(tesseract_text(steps["morph_open"], psm=psm, lang="tha+eng"))
                    raw = max(cand, key=len)
                except Exception as e:
                    raw = f"[Tesseract error] {e}"
            st.text_area(f"OCR Output (Raw Text) — page {pi}", value=raw, height=260)

            guided = extract_with_guidance(steps["original"], steps) if TESS_OK else {}
            fallback = fallback_from_raw(raw)

            vendor = extract_vendor(raw)
            result = {
                "file": f"{up.name}#p{pi}",
                "Vendor / Supplier": vendor,
                "Quotation No.": guided.get("Quotation No.") or fallback.get("Quotation No."),
                "Date": guided.get("Date") or fallback.get("Date"),
                "Subtotal": guided.get("Subtotal") or fallback.get("Subtotal"),
                "VAT": guided.get("VAT") or fallback.get("VAT"),
                "Grand Total": guided.get("Grand Total") or fallback.get("Grand Total"),
            }

            with col2:
                st.write("**สรุปฟิลด์ที่สกัดได้ (มาตรฐาน)**")
                st.dataframe(pd.DataFrame([result]))

            rows.append(result)

if rows:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df = pd.DataFrame(rows, columns=["file","Vendor / Supplier","Quotation No.","Date","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลด CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ocr_quotation_results.csv", mime="text/csv")
    if sheet_url and service_json_file is not None:
        try:
            service_dict = json.load(service_json_file)
            ok, msg = export_to_google_sheets(df, sheet_url, service_dict, worksheet_name=worksheet_name)
            (st.success if ok else st.error)(msg)
        except Exception as e:
            st.error(f"อ่าน Service JSON ไม่ได้: {e}")
