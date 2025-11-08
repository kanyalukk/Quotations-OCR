# ต้องอยู่บรรทัดแรกกันปัญหา SessionInfo
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")

import os, re, json, shutil
from typing import Dict, List, Tuple, Optional
import numpy as np, pandas as pd
from PIL import Image
import cv2, dateparser, fitz  # PyMuPDF (อ่าน PDF เป็นภาพ)

# ===== OCR engines =====
EASYOCR_AVAILABLE = True
try:
    import easyocr
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
except Exception:
    pytesseract = None

# ===== Utils =====
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}
def to_english_digits(s): return s.translate(THAI_DIGITS) if isinstance(s,str) else s
def fix_numberlike_ocr(s):
    if not isinstance(s,str): return s
    s = re.sub(r'(?<=\d)[oO](?=[\d,\.])','0',s)
    s = re.sub(r'(?<=[,\.\s])[oO](?=\d)','0',s)
    s = re.sub(r'(?<=\d)[lI](?=[\d,\.])','1',s)
    s = re.sub(r'(?<=\d)B(?=[\d,\.])','8',s)
    return s
def sanitize_text(text:str)->str:
    if not text: return ""
    text = to_english_digits(text)
    for k,v in TH_MONTHS.items(): text = re.sub(k,v,text)
    text = re.sub(r"[ \t]+"," ",text).replace("—","-").replace("–","-").replace("：",":")
    return text
def normalize_number(s:str)->Optional[float]:
    if not s: return None
    s = fix_numberlike_ocr(to_english_digits(s))
    s = s.replace(" ","").replace(",","").replace("฿","").replace("บาท","").replace("%","")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None
def parse_date_candidates(text:str)->Optional[str]:
    t = sanitize_text(text)
    c=set()
    for m in re.finditer(r"(วันที่|date)[:\s\-]*([^\n]{0,40})",t,re.I): c.add(m.group(0))
    c.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",t))
    c.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",t))
    th=r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    c.update(re.findall(rf"\b\d{{1,2}}\s*{th}\s*\d{{2,4}}\b",t))
    parsed=[]
    for s in list(c)[:30]:
        dt = dateparser.parse(s,languages=["th","en"],settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year>2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    return (sorted(parsed)[-1].isoformat() if parsed else None)

# ===== Preprocess =====
def deskew(binary_img: np.ndarray)->Tuple[np.ndarray,float]:
    coords=np.column_stack(np.where(binary_img>0))
    if coords.size==0: return binary_img,0.0
    angle=cv2.minAreaRect(coords)[-1]
    angle=-(90+angle) if angle<-45 else -angle
    (h,w)=binary_img.shape[:2]
    M=cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
    rotated=cv2.warpAffine(binary_img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle
def preprocess(img_bgr: np.ndarray):
    out={}
    out["original"]=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY); out["grayscale"]=gray
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray); out["clahe"]=clahe
    blur=cv2.medianBlur(clahe,3); out["denoise(median3)"]=blur
    th=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9); out["adaptive_threshold"]=th
    de,_=deskew(th); out["deskewed"]=de
    up=cv2.resize(de,None,fx=1.7,fy=1.7,interpolation=cv2.INTER_CUBIC); out["upscale(1.7x)"]=up
    opened=cv2.morphologyEx(up,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),iterations=1); out["morph_open"]=opened
    return out

# ===== Tesseract helpers =====
def ensure_tesseract(user_path: Optional[str]):
    if pytesseract is None: return (False,None,"pytesseract not installed")
    cand=[]
    if user_path: cand.append(user_path)
    cand += ["/usr/bin/tesseract","/usr/local/bin/tesseract","/opt/homebrew/bin/tesseract",
             r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for p in cand:
        if os.path.exists(p):
            try:
                pytesseract.pytesseract.tesseract_cmd=p
                pytesseract.get_tesseract_version()
                return True,p,None
            except Exception: pass
    exe=shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd=exe
            pytesseract.get_tesseract_version()
            return True,exe,None
        except Exception as e:
            return False,exe,str(e)
    return False,None,"tesseract not found"

def tesseract_text(img, cfg="--oem 3 --psm 6 -l tha+eng"):
    return pytesseract.image_to_string(img, config=cfg)

# ===== EasyOCR helpers =====
@st.cache_resource(show_spinner=False)
def get_reader():
    if not EASYOCR_AVAILABLE: return None
    return easyocr.Reader(["th","en"], gpu=False)

def easyocr_boxes(img_rgb):
    """คืนค่า list ของ dict: {box, text, conf}"""
    rd = get_reader()
    if rd is None: return []
    results = rd.readtext(img_rgb, detail=1, paragraph=False, min_size=5)
    out=[]
    for (box, txt, conf) in results:
        out.append({"box": np.array(box).astype(int), "text": txt, "conf": float(conf), "lower": txt.lower().strip()})
    return out

def crop_right_of_box(img_rgb, box, x_pad=10, w_ratio=0.45):
    h, w = img_rgb.shape[:2]
    x1 = int(max(box[:,0].max()+x_pad, 0))
    y1 = int(max(box[:,1].min()-5, 0))
    y2 = int(min(box[:,1].max()+5, h))
    x2 = int(min(w, x1 + int(w*w_ratio)))
    if x2<=x1 or y2<=y1: return img_rgb[0:1,0:1]  # empty tiny
    return img_rgb[y1:y2, x1:x2]

# ===== Guided OCR for key fields =====
LABELS_SUBTOTAL = ["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี"]
LABELS_VAT      = ["vat","ภาษีมูลค่าเพิ่ม","vat7","vat 7"]
LABELS_GRAND    = ["grand total","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"]
def normalize_label(s): return re.sub(r"[^a-z0-9ก-๙]+","", s.lower())

def guided_amount(img_rgb) -> Dict[str, Optional[float]]:
    out={"Subtotal":None,"VAT":None,"Grand Total":None}
    boxes = easyocr_boxes(img_rgb)
    if not boxes: return out
    # map labels to ROIs
    for b in boxes:
        lb = normalize_label(b["lower"])
        def any_in(keys): return any(k.replace(" ","") in lb for k in keys)
        if any_in(LABELS_GRAND):
            roi = crop_right_of_box(img_rgb, b["box"], w_ratio=0.5)
            txt = tesseract_text(roi, cfg="--oem 3 --psm 6 -l eng --dpi 300 -c tessedit_char_whitelist=0123456789.,")
            v = normalize_number(txt)
            if v: out["Grand Total"]=v
        elif any_in(LABELS_VAT):
            roi = crop_right_of_box(img_rgb, b["box"], w_ratio=0.35)
            txt = tesseract_text(roi, cfg="--oem 3 --psm 6 -l eng --dpi 300 -c tessedit_char_whitelist=0123456789.,")
            v = normalize_number(txt)
            if v: out["VAT"]=v
        elif any_in(LABELS_SUBTOTAL) or "total"==lb:  # 'total' (ไม่ใช่ grand)
            roi = crop_right_of_box(img_rgb, b["box"], w_ratio=0.35)
            txt = tesseract_text(roi, cfg="--oem 3 --psm 6 -l eng --dpi 300 -c tessedit_char_whitelist=0123456789.,")
            v = normalize_number(txt)
            if v: out["Subtotal"]=v
    return out

def guided_qt_date(img_rgb) -> Tuple[Optional[str], Optional[str]]:
    qt, date = None, None
    boxes = easyocr_boxes(img_rgb)
    if not boxes: return qt, date
    # ค้นกล่อง 'Quotation' หรือ 'Quotation No'
    for i, b in enumerate(boxes):
        if "quotation" in b["lower"]:
            # หา No / Date รอบ ๆ
            roi_right = crop_right_of_box(img_rgb, b["box"], w_ratio=0.6)
            # ตัวเลข/ตัวอักษรสำหรับรหัสเอกสาร
            qt_try = tesseract_text(roi_right, cfg="--oem 3 --psm 6 -l eng --dpi 300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.,")
            qt_try = re.split(r"\s+date\b", qt_try, flags=re.I)[0]
            qt_try = re.sub(r"[^A-Za-z0-9/_\-.]+","", qt_try).strip()
            if qt_try and len(qt_try)>=6: qt = qt_try
            # หา date ด้วย OCR อีกครั้ง (ตัวเลข/ / -)
            date_try = tesseract_text(roi_right, cfg="--oem 3 --psm 6 -l tha+eng --dpi 300")
            d = parse_date_candidates(date_try)
            if d: date = d
            break
    return qt, date

# ===== Regex/NER จาก Raw text (fallback/เสริม) =====
def normalize_qt_code(raw:str)->str:
    if not raw: return raw
    s=raw.strip().translate(str.maketrans({"$":"S","§":"S"}))
    s=re.sub(r"^[Kk]5","KS",s); s=re.sub(r"^5","S",s)
    s=fix_numberlike_ocr(s)
    return s.replace(" ","").upper()

def ner_extract_quotation_no(text:str)->Optional[str]:
    t=sanitize_text(text); lines=t.splitlines()
    for i,l in enumerate(lines):
        if re.search(r"quotation\s*no\.?", l, re.I):
            after=re.split(r"quotation\s*no\.?\s*[:#]?\s*", l, flags=re.I)[-1]
            m=re.search(r"([A-Z\$\§]{0,5}[A-Z0-9\$\§/_\-.]{3,})", after, re.I)
            if m and not re.fullmatch(r"date", m.group(1), re.I): return normalize_qt_code(m.group(1))
            look=" ".join(lines[i+1:i+3])
            date_m=re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", look)
            tokens=re.findall(r"[A-Z\$\§]?[A-Z0-9\$\§/_\-.]{3,}", look, re.I)
            if tokens:
                if date_m:
                    cut=look[:date_m.start()]
                    cand=re.findall(r"[A-Z\$\§]?[A-Z0-9\$\§/_\-.]{3,}", cut, re.I)
                    if cand: return normalize_qt_code(cand[-1])
                return normalize_qt_code(tokens[0])
    m=re.search(r"\b[A-Z\$\§]{1,4}[-/_.]?\d{2,4}[-/_.]?\d{1,6}\b", t, re.I)
    return normalize_qt_code(m.group(0)) if m else None

def last_amount_in_line(line:str)->Optional[float]:
    no_pct=re.sub(r"\d+(\.\d+)?\s*%","",line)
    no_pct=fix_numberlike_ocr(no_pct)
    nums=re.findall(r"-?\d[\d,]*\.?\d*", no_pct)
    if not nums: return None
    for n in reversed(nums):
        v=normalize_number(n)
        if v is not None and ("," in n or "." in n or v>=100): return v
    return normalize_number(nums[-1])

def amount_by_label(text:str, include:List[str], exclude:List[str]=[])->Optional[float]:
    t=sanitize_text(text); lines=[l.strip() for l in t.splitlines() if l.strip()]
    idxs=[i for i,l in enumerate(lines) if any(k.lower() in l.lower() for k in include) and not any(b.lower() in l.lower() for b in exclude)]
    if not idxs: return None
    i=idxs[-1]; v=last_amount_in_line(lines[i])
    if v is None and i+1<len(lines): v=last_amount_in_line(lines[i+1])
    return v

def extract_vendor(text:str)->Optional[str]:
    t=sanitize_text(text); lines=[l.strip() for l in t.splitlines() if l.strip()]
    pats=[r"(บริษัท.+จำกัด|บริษัท.+จำกัด\(มหาชน\)|ห้างหุ้นส่วนจำกัด|หจก\.)",
          r"(co\.,?\s*ltd\.?|company\s*limited|limited\s*company)",
          r"(vendor|supplier|ผู้ขาย|ผู้จำหน่าย)[:\s]*([^\n]+)"]
    for l in lines[:30]:
        for pat in pats:
            m=re.search(pat,l,re.I)
            if m:
                cand=m.group(m.lastindex) if (m.lastindex and m.group(m.lastindex)) else l
                cand=re.sub(r"^(vendor|supplier|ผู้ขาย|ผู้จำหน่าย)[:\s\-]*","",cand,flags=re.I)
                cand=re.sub(r"\s{2,}"," ",cand)
                cand=re.sub(r"(ที่อยู่|address)[:\s].*$","",cand,flags=re.I)
                return cand.strip()
    for l in lines[:10]:
        if re.search(r"(บริษัท|หจก\.|co\.,?\s*ltd\.?)", l, re.I): return l
    return None

# ===== PDF =====
def pdf_pages_to_rgb_images(file_bytes:bytes, dpi:int=300)->List[np.ndarray]:
    images=[]
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pix=p.get_pixmap(dpi=dpi, alpha=False)
            img=np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,pix.width,3)
            images.append(img[:,:,::-1])
    return images

# ===== Main extraction =====
def extract_fields_full(img_rgb, img_bin, raw_text:str)->Dict[str,Optional[str]]:
    # 1) guided by EasyOCR boxes
    qt, dt = guided_qt_date(img_rgb) if EASYOCR_AVAILABLE else (None,None)
    amts = guided_amount(img_rgb) if EASYOCR_AVAILABLE else {"Subtotal":None,"VAT":None,"Grand Total":None}

    # 2) regex/NER fallback from raw text
    txt = sanitize_text(raw_text or "")
    vendor = extract_vendor(txt)
    if not qt:  qt = ner_extract_quotation_no(txt)
    if not dt:  dt = parse_date_candidates(txt)

    subtotal = amts["Subtotal"]
    vat      = amts["VAT"]
    grand    = amts["Grand Total"]
    if subtotal is None:
        subtotal = amount_by_label(txt, ["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","total"], exclude=["grand","vat"])
    if vat is None:
        vat = amount_by_label(txt, ["vat","ภาษีมูลค่าเพิ่ม"])
    if grand is None:
        grand = amount_by_label(txt, ["grand total","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"])

    # reconcile
    if vat is not None and vat < 50 and re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", txt, re.I):
        if grand is not None and subtotal is not None: vat = round(grand - subtotal, 2)
        elif subtotal is not None: vat = round(subtotal * 0.07, 2)
    if grand is None and subtotal is not None and vat is not None: grand = round(subtotal + vat, 2)
    if subtotal is None and grand is not None and vat is not None: subtotal = round(grand - vat, 2)
    if vat is None and grand is not None and subtotal is not None: vat = round(grand - subtotal, 2)

    return {"Vendor / Supplier":vendor,"Quotation No.":qt,"Date":dt,
            "Subtotal":subtotal,"VAT":vat,"Grand Total":grand}

# ===== Google Sheets =====
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
            ws.append_row([("" if x is None else str(x)) for x in row.tolist()])
        return True, "Exported to Google Sheets successfully."
    except Exception as e:
        return False, f"Export failed: {e}"

# ===== UI =====
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    engine = st.selectbox("OCR Engine", ["Hybrid (EasyOCR + Tesseract)","Tesseract only","EasyOCR only"], 0)
    user_tess_path = st.text_input("Tesseract path (ถ้าระบบหาไม่เจอ)", "")
    show_steps = st.checkbox("แสดงภาพ Pre-processing ทุกขั้นตอน", True)
    worksheet_name = st.text_input("Worksheet name (Google Sheets)", "OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("ลิงก์ Google ชีท (แชร์สิทธิ์แก้ไขให้ Service Account)")
    service_json_file = st.file_uploader("อัปโหลด Service Account JSON", type=["json"], accept_multiple_files=False)

TESS_OK, TESS_PATH, TESS_MSG = ensure_tesseract(user_tess_path.strip() or None)
st.sidebar.markdown("**Tesseract:** " + ("✅ " + str(TESS_PATH) if TESS_OK else "❌ " + str(TESS_MSG)))

st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง")
st.caption("Hybrid OCR + Guided ROI (EasyOCR → crop → Tesseract) + Regex/NER + Google Sheets export")

uploads = st.file_uploader("อัปโหลด JPG/PNG/PDF ได้หลายไฟล์", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)

rows=[]
if uploads:
    for up in uploads:
        st.markdown("---")
        st.write(f"**ไฟล์:** {up.name}")
        # โหลดเป็นภาพ
        if up.type=="application/pdf" or up.name.lower().endswith(".pdf"):
            images_bgr = pdf_pages_to_rgb_images(up.read())
        else:
            image = Image.open(up).convert("RGB")
            images_bgr = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)]

        for page_idx, img_bgr in enumerate(images_bgr, start=1):
            steps = preprocess(img_bgr)
            col1,col2 = st.columns([1,1])
            with col1:
                if show_steps:
                    tabs = st.tabs(list(steps.keys()))
                    for tab,k in zip(tabs,steps.keys()):
                        with tab: st.image(steps[k], caption=f"{k} (page {page_idx})", use_column_width=True)
            with col2:
                # Raw OCR (เพื่อ regex/fallback)
                raw = ""
                if engine in ["Hybrid (EasyOCR + Tesseract)","EasyOCR only"] and EASYOCR_AVAILABLE:
                    try: raw += ocr_easyocr(steps["original"]) + "\n"
                    except Exception: pass
                if engine in ["Hybrid (EasyOCR + Tesseract)","Tesseract only"] and pytesseract is not None:
                    try: raw += tesseract_text(steps["morph_open"])
                    except Exception: pass
                st.text_area(f"OCR Output (Raw Text) — page {page_idx}", value=raw, height=240)

                fields = extract_fields_full(steps["original"], steps["morph_open"], raw)
                row = {"file": f"{up.name}#p{page_idx}", **fields}
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
            ok, msg = export_to_google_sheets(df, sheet_url, service_dict, worksheet_name=worksheet_name)
            (st.success if ok else st.error)(msg)
        except Exception as e:
            st.error(f"อ่านไฟล์ Service JSON ไม่ได้: {e}")
