# ต้องอยู่บรรทัดแรกเพื่อกัน SessionInfo error
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")

import os, re, json, shutil
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2, fitz, dateparser
from difflib import SequenceMatcher  # fuzzy แบบไม่ต้องติดตั้งเพิ่ม

# ---------------- OCR: Tesseract ----------------
try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None

# ---------------- Text helpers ----------------
TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")
TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}
def to_en_digits(s: str) -> str:
    return s.translate(TH_DIGITS) if isinstance(s, str) else s

def fix_numberlike(s: str) -> str:
    if not isinstance(s, str): return s
    # อักษรที่มักเพี้ยนเป็นตัวเลข
    s = re.sub(r'(?<=\d)[oO](?=[\d,\.])','0',s)
    s = re.sub(r'(?<=[,\.\s])[oO](?=\d)','0',s)
    s = re.sub(r'(?<=\d)[lI](?=[\d,\.])','1',s)
    s = re.sub(r'(?<=\d)B(?=[\d,\.])','8',s)
    return s

def sanitize_text(t: str) -> str:
    if not t: return ""
    t = to_en_digits(t)
    for k,v in TH_MONTHS.items():
        t = re.sub(k, v, t)
    t = re.sub(r"[ \t]+"," ", t).replace("—","-").replace("–","-").replace("：",":")
    return t

def normalize_number(s: str) -> Optional[float]:
    """รองรับทั้ง 107,000.00 และรูปแบบยุโรป 107.000,00"""
    if not s: return None
    s0 = s
    s = fix_numberlike(to_en_digits(s))
    s = s.strip().replace("฿","").replace("บาท","")
    # European: 1.234,56
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", s):
        s = s.replace(".","").replace(",",".")
    # US/TH: 1,234.56
    s = s.replace(" ", "").replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None

def parse_date_candidates(text: str) -> Optional[str]:
    t = sanitize_text(text)
    c = set()
    c.update(re.findall(r"(?:วันที่|date|issued\s*date|quotation\s*date)[:\-\s]*([^\n]{1,40})", t, flags=re.I))
    c.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t))
    c.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", t))
    th = r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    c.update(re.findall(rf"\b\d{{1,2}}\s*{th}\s*\d{{2,4}}\b", t))
    parsed=[]
    for s in list(c)[:50]:
        dt = dateparser.parse(s, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year>2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    return sorted(parsed)[-1].isoformat() if parsed else None

# ---------------- Preprocessing ----------------
def binarize_for_tesseract(gray):
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    if (th==255).mean() > 0.92:
        _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (th==0).mean() > 0.6:
        th = 255 - th
    return th

def deskew(binary_img: np.ndarray)->Tuple[np.ndarray,float]:
    coords = np.column_stack(np.where(binary_img<128))
    if coords.size==0: return binary_img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h,w) = binary_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rot = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle

def preprocess(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    out={}
    out["original"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); out["grayscale"]=gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray); out["clahe"]=clahe
    th = binarize_for_tesseract(clahe); out["binary"]=th
    rot,_ = deskew(th); out["deskewed"]=rot
    up = cv2.resize(rot, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC); out["upscale(1.8x)"]=up
    open1 = cv2.morphologyEx(up, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations=1); out["morph_open"]=open1
    return out

# ---------------- Tesseract utils ----------------
def ensure_tesseract(user_path: Optional[str]):
    if pytesseract is None: return (False,None,"pytesseract not installed")
    cand=[]
    if user_path: cand.append(user_path)
    cand += ["/usr/bin/tesseract","/usr/local/bin/tesseract","/opt/homebrew/bin/tesseract",
             r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for p in cand:
        if os.path.exists(p):
            try:
                pytesseract.pytesseract.tesseract_cmd = p
                pytesseract.get_tesseract_version()
                return True,p,None
            except Exception: pass
    exe = shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd = exe
            pytesseract.get_tesseract_version()
            return True,exe,None
        except Exception as e:
            return False,exe,str(e)
    return False,None,"tesseract not found"

def ocr_data(img_bin) -> pd.DataFrame:
    df = pytesseract.image_to_data(img_bin, config="--oem 3 --psm 6 -l tha+eng", output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    if "conf" in df.columns:
        df = df[df["conf"].astype(float) > 40].copy()
    df["text"] = df["text"].astype(str)
    df["norm"] = df["text"].str.lower().str.replace(r"[^a-z0-9ก-๙]+","", regex=True)
    return df.reset_index(drop=True)

def ocr_text_best(img_bin) -> str:
    outs=[]
    for psm in (6,4,11,12):
        outs.append(pytesseract.image_to_string(img_bin, config=f"--oem 3 --psm {psm} -l tha+eng"))
    return max(outs, key=len)

# ---------------- Lines & grouping ----------------
def lines_from_df(df: pd.DataFrame) -> pd.DataFrame:
    g=["page_num","block_num","par_num","line_num"]
    agg = df.groupby(g).agg(left=("left","min"),
                            top=("top","min"),
                            right=("left","max"),
                            bottom=("top","max"),
                            height=("height","max")).reset_index()
    texts = df.groupby(g)["text"].apply(lambda s:" ".join([x for x in s if x.strip()])).reset_index(name="text")
    norms = df.groupby(g)["norm"].apply(lambda s:" ".join([x for x in s if x.strip()])).reset_index(name="norm")
    ln = agg.merge(texts,on=g).merge(norms,on=g)
    ln["right"] = ln["right"] + df.groupby(g)["width"].max().values
    return ln

# ===== fuzzy/helpers =====
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9ก-๙]+","", (s or "").lower())

def _ratio(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b: return 0.0
    sm = SequenceMatcher(None, a, b).ratio()
    if len(a) < len(b):
        best = 0
        for i in range(0, len(b)-len(a)+1):
            best = max(best, SequenceMatcher(None, a, b[i:i+len(a)]).ratio())
        sm = max(sm, best)
    return sm

COMMON_FIX = {
    "quotation no": [
        "quotationno","quotatlonno","quotatlon","quotation №",
        "เลขที่ใบเสนอราคา","quotation number","quotation id","quote no","ref no"
    ],
    "date": [
        "วันที่","oate","dare","quotation date","issue date","issued date",
        "ใบเสนอราคาวันที่","เอกสารลงวันที่","doc date"
    ],
    "subtotal": [
        "sub-total","sutotal","subtota|","subtotai","subtotl",
        "รวมก่อนภาษี","ยอดก่อนภาษี","net total","amount before vat"
    ],
    "vat": [
        "vat7%","vat 7 %","va7","ภาษีมูลค่าเพิ่ม","vat:","tax","vat amount"
    ],
    "grand total": [
        "grandtotai","grandtotl","grand tota|","ยอดรวมสุทธิ","รวมทั้งสิ้น",
        "ยอดชำระสุทธิ","total amount","amount due","total due"
    ]
}
def expand_keywords(keys: List[str]) -> List[str]:
    out=set()
    for k in keys:
        out.add(k)
        base=_norm(k)
        for canon, alts in COMMON_FIX.items():
            if _norm(canon)==base:
                out.update(alts)
    return list(out)

def find_line_fuzzy(ln: pd.DataFrame, include: List[str], exclude: List[str]=None,
                    prefer_last: bool = True, cutoff: float = 0.72) -> Optional[pd.Series]:
    if exclude is None: exclude=[]
    inc = expand_keywords(include)
    exc = [ _norm(x) for x in (exclude or []) ]
    cand = ln.copy()
    def score_row(text):
        t = _norm(text)
        if any(x in t for x in exc): return 0.0
        return max(_ratio(t, _norm(k)) for k in inc)
    cand["__score__"] = cand["text"].apply(score_row)
    cand = cand[cand["__score__"]>=cutoff]
    if cand.empty: return None
    cand = cand.sort_values(["page_num","top","left","__score__"])
    return cand.iloc[-1] if prefer_last else cand.iloc[-1]

# --------- tokens/values near anchor (ไม่บังคับอยู่บรรทัดเดียว) ----------
def tokens_right_of_anchor(df_words: pd.DataFrame, anchor_row: pd.Series,
                           max_dx: int = 900, dy_factor: float = 1.4) -> List[str]:
    """เลือก token ที่อยู่ทางขวาของฉลาก (ภายในกรอบกว้าง max_dx และสูง dy_factor*height)"""
    h = int(anchor_row["height"])
    top_min = int(anchor_row["top"] - max(12, h*0.4))
    top_max = int(anchor_row["top"] + h*dy_factor)
    mask = (df_words["page_num"]==anchor_row["page_num"]) & \
           (df_words["left"] > anchor_row["right"]+2) & \
           (df_words["left"] < anchor_row["right"]+max_dx) & \
           (df_words["top"] >= top_min) & (df_words["top"] <= top_max)
    return df_words[mask].sort_values(["top","left"])["text"].tolist()

# ---------------- Vendor cleaner ----------------
def _clean_vendor_line(s: str) -> str:
    s = " ".join(s.split())
    # อังกฤษ: จบที่ Co., Ltd. / Company Limited และตัดอักษรเดี่ยวหน้า (เศษโลโก้)
    m = re.search(r"(?<![A-Za-z]\s)([A-Za-z][A-Za-z '&\.\-]+?(?:Co\.,?\s*Ltd\.|Company\s*Limited))", s, flags=re.I)
    if m:
        v = m.group(1).strip()
        v = re.sub(r"^[A-Za-z]\s+(?=[A-Za-z])", "", v)  # ตัด "M " จากโลโก้ที่มักติดมา
        v = re.sub(r"\s*,\s*", ", ", v)
        v = re.sub(r"\s+Co\.,?\s*Ltd\.?", " Co., Ltd.", v, flags=re.I)
        v = re.sub(r"\s+Company\s+Limited", " Company Limited", v, flags=re.I)
        v = re.sub(r"\s{2,}", " ", v).strip()
        return v
    # ภาษาไทย
    m = re.search(r"(บริษัท.+?(?:จำกัด\(มหาชน\)|จำกัด))", s)
    if m:
        return m.group(1).strip()
    return s.strip()

def extract_vendor(df_words: pd.DataFrame, page_h: int) -> Optional[str]:
    ln = lines_from_df(df_words)
    head = ln[ln["top"] < page_h * 0.30]
    if head.empty:
        head = ln[ln["top"] < page_h * 0.40]
    BAD_VENDOR = r"(customer|address|project|quotation|page[:\s]|date[:\s])"
    head = head[~head["text"].str.contains(BAD_VENDOR, flags=re.I, regex=True, na=False)]
    head_text = " ".join(head.sort_values(["top","left"])["text"].tolist())
    head_text = " ".join(head_text.split())
    m = re.search(r"(?<![A-Za-z]\s)([A-Za-z][A-Za-z '&\.\-]+?(?:Co\.,?\s*Ltd\.|Company\s*Limited))", head_text, flags=re.I)
    if m:
        return _clean_vendor_line(m.group(1))
    m = re.search(r"(บริษัท.+?(?:จำกัด\(มหาชน\)|จำกัด))", head_text)
    if m:
        return _clean_vendor_line(m.group(1))
    if not head.empty:
        pri = pd.Series(0, index=head.index, dtype=float)
        pri += head["text"].str.contains(r"บริษัท|จำกัด", regex=True).astype(int)*2
        pri += head["text"].str.contains(r"co\.,?\s*ltd\.?|company\s*limited", flags=re.I, regex=True).astype(int)*2
        pri += head["text"].str.contains(r"solutions|consultants|askme", flags=re.I, regex=True).astype(int)
        winner = head.loc[pri.idxmax()]["text"]
        return _clean_vendor_line(winner)
    return None

# ---------------- Header fields & Amounts ----------------
def extract_header(df_words: pd.DataFrame)->Tuple[Optional[str], Optional[str]]:
    ln = lines_from_df(df_words)
    # Quotation No
    ql = find_line_fuzzy(ln, ["quotation no","quotation"], cutoff=0.68)
    qt, dt = None, None
    if ql is not None:
        tx = " ".join(tokens_right_of_anchor(df_words, ql))
        m = re.search(r"\b[A-Z]{1,3}\d{6,}\b", tx)
        if m: qt = m.group(0)
        if qt is None:
            qs = re.findall(r"[A-Za-z][A-Za-z0-9/_\-.]{5,}", tx)
            if qs: qt = max(qs, key=len).upper()
    # Date (ใช้กรอบใกล้ขวาของคำว่า Date โดยตรง)
    dl = find_line_fuzzy(ln, ["date","วันที่"], cutoff=0.55)
    if dl is not None:
        tokens = tokens_right_of_anchor(df_words, dl)
        dt = parse_date_candidates(" ".join(tokens))
    if dt is None:
        alltxt = " ".join(ln["text"].tolist())
        dt = parse_date_candidates(alltxt)
    if qt is None:
        alltxt = " ".join(ln["text"].tolist())
        m = re.search(r"\b[A-Z]{1,3}\d{6,}\b", alltxt)
        if m: qt = m.group(0)
    return qt, dt

def rightmost_number_on_line(df_words: pd.DataFrame, line_row: pd.Series) -> Optional[float]:
    mask = (df_words["page_num"]==line_row["page_num"]) & \
           (df_words["block_num"]==line_row["block_num"]) & \
           (df_words["par_num"]==line_row["par_num"]) & \
           (df_words["line_num"]==line_row["line_num"])
    sub = df_words[mask].sort_values("left")
    nums=[]
    for _,r in sub.iterrows():
        if re.fullmatch(r"\d[\d,\.]*", r["text"]):
            nums.append((r["left"], normalize_number(r["text"])))
    if nums: return nums[-1][1]
    return None

def extract_amounts(df_words: pd.DataFrame, page_w:int, page_h:int=None)->Tuple[Optional[float], Optional[float], Optional[float]]:
    ln = lines_from_df(df_words)
    # --- โซนคาดการณ์สรุปยอด: ขวาล่าง & ซ้ายล่าง ---
    ZONES = [
        (0.55, 0.60, 0.98, 0.98),  # ขวาล่าง
        (0.02, 0.60, 0.45, 0.98),  # ซ้ายล่าง (เผื่อบางฟอร์ม)
    ]
    candidates = []

    def amounts_in_lines(lines):
        gl = find_line_fuzzy(lines, ["grand total"], cutoff=0.60)
        vl = find_line_fuzzy(lines, ["vat"], cutoff=0.55)
        sl = find_line_fuzzy(lines, ["subtotal"], exclude=["grand","vat"], cutoff=0.55)
        grand = rightmost_number_on_line(df_words, gl) if gl is not None else None
        vat   = rightmost_number_on_line(df_words, vl) if vl is not None else None
        sub   = rightmost_number_on_line(df_words, sl) if sl is not None else None
        return sub, vat, grand

    for (x1,y1,x2,y2) in ZONES:
        zone = ln.copy()
        zone = zone[(zone["left"]  >= page_w*x1) & (zone["right"] <= page_w*x2)]
        if page_h is not None:
            zone = zone[(zone["top"]   >= page_h*y1) & (zone["bottom"]<= page_h*y2)]
        sub, vat, grand = amounts_in_lines(zone)
        candidates.append((sub,vat,grand))

    # Fallback: ทั้งหน้าฝั่งขวา
    zone_all = ln[ln["right"] > page_w*0.50]
    candidates.append(amounts_in_lines(zone_all))

    # Tail numbers: เก็บค่าตัวเงินท้ายหน้า
    money_rows=[]
    for _,r in ln.iterrows():
        v = rightmost_number_on_line(df_words, r)
        if v is not None: money_rows.append((r["top"], v))
    money_rows = sorted(money_rows, key=lambda x:x[0])[-6:]
    tail = [v for _,v in money_rows]
    if len(tail)>=2:
        t_sorted = sorted(tail)
        candidates.append((t_sorted[-2], None, t_sorted[-1]))  # (sub?, ?, grand)

    # เลือกชุดที่สมการเงินดีที่สุด
    best = (None,None,None); best_err = 1e18
    for (s,v,g) in candidates:
        if s is None and v is not None and g is not None: s = round(g - v, 2)
        if v is None and s is not None and g is not None: v = round(g - s, 2)
        if g is None and s is not None and v is not None: g = round(s + v, 2)
        if s is None and v is None and g is None: continue
        if (s is not None) and (v is not None) and (g is not None):
            err = abs((s+v) - g)
        elif g is not None and (s is not None or v is not None):
            err = 0.09
        else:
            err = 0.5
        if err < best_err:
            best_err, best = err, (s,v,g)

    sub, vat, grand = best
    # ถ้ามีคำว่า 7% ให้ reconcile อีกครั้ง
    all_text = " ".join(ln["text"].tolist())
    if re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", all_text, flags=re.I):
        if grand and sub and (vat is None or vat < 50): vat = round(grand - sub, 2)
    return sub, vat, grand

# ---------------- PDF helper ----------------
def pdf_to_bgr_list(file_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    out=[]
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pix = p.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            out.append(img[:,:,::-1])  # BGR
    return out

# ---------------- Google Sheets ----------------
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

# ---------------- UI ----------------
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    user_tess_path = st.text_input("Tesseract path (ถ้าไม่เจอให้ระบุ)", "")
    show_steps = st.checkbox("แสดงภาพ Pre-processing", True)
    worksheet_name = st.text_input("Worksheet name (Google Sheets)", "OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("ลิงก์ Google ชีท (แชร์สิทธิ์แก้ไขให้ Service Account)")
    service_json_file = st.file_uploader("อัปโหลด Service Account JSON", type=["json"], accept_multiple_files=False)

TESS_OK, TESS_PATH, TESS_MSG = ensure_tesseract(user_tess_path.strip() or None)
st.sidebar.write("**Tesseract:** ", "✅ "+str(TESS_PATH) if TESS_OK else "❌ "+str(TESS_MSG))

st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง")
uploads = st.file_uploader("อัปโหลด JPG/PNG/PDF ได้หลายไฟล์", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)

records=[]
if uploads:
    for up in uploads:
        st.markdown("---")
        st.write(f"**ไฟล์:** {up.name}")

        if up.type=="application/pdf" or up.name.lower().endswith(".pdf"):
            pages = pdf_to_bgr_list(up.read())
        else:
            im = Image.open(up).convert("RGB")
            pages = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)]

        for pidx, img_bgr in enumerate(pages, start=1):
            steps = preprocess(img_bgr)

            c1,c2 = st.columns([1,1])
            with c1:
                if show_steps:
                    keys = list(steps.keys())
                    tabs = st.tabs(keys)
                    for i, name in enumerate(keys):
                        img = steps[name]
                        with tabs[i]:
                            try:
                                if isinstance(img, np.ndarray):
                                    if img.ndim == 2:
                                        if img.dtype != np.uint8:
                                            img = img.astype(np.uint8)
                                        st.image(img, caption=f"{name} (page {pidx})",
                                                 use_column_width=True, clamp=True)
                                    elif img.ndim == 3:
                                        st.image(img, caption=f"{name} (page {pidx})",
                                                 use_column_width=True)
                                    else:
                                        st.write("ไม่สามารถแสดงภาพรูปแบบนี้ได้")
                                elif isinstance(img, Image.Image):
                                    st.image(img, caption=f"{name} (page {pidx})",
                                             use_column_width=True)
                            except Exception as e:
                                st.write(f"⚠️ แสดงรูป {name} ไม่ได้: {e}")

            if not TESS_OK:
                st.error("ไม่พบ Tesseract ในระบบ กรุณาติดตั้ง/ระบุ path ในแถบด้านซ้าย")
                continue

            df_words = ocr_data(steps["upscale(1.8x)"])
            page_h, page_w = steps["original"].shape[:2]

            vendor = extract_vendor(df_words, page_h)
            qt, dt = extract_header(df_words)
            sub, vat, grand = extract_amounts(df_words, page_w, page_h)

            try:
                raw = ocr_text_best(steps["morph_open"])
            except Exception as e:
                raw = f"[Tesseract error] {e}"

            with c2:
                st.text_area(f"OCR Output (Raw Text) — page {pidx}", value=raw, height=220)
                rec = {
                    "file": f"{up.name}#p{pidx}",
                    "Vendor / Supplier": vendor,
                    "Quotation No.": qt,
                    "Date": dt,
                    "Subtotal": sub,
                    "VAT": vat,
                    "Grand Total": grand
                }
                st.write("**สรุปฟิลด์ที่สกัดได้**")
                st.dataframe(pd.DataFrame([rec]), use_container_width=True)
                records.append(rec)

if records:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df = pd.DataFrame(records, columns=["file","Vendor / Supplier","Quotation No.","Date","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลด CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ocr_quotation_results.csv", mime="text/csv")
    if sheet_url and service_json_file is not None:
        try:
            svc = json.load(service_json_file)
            ok, msg = export_to_google_sheets(df, sheet_url, svc, worksheet_name=worksheet_name)
            (st.success if ok else st.error)(msg)
        except Exception as e:
            st.error(f"อ่าน Service JSON ไม่ได้: {e}")
