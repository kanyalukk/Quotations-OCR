# ต้องอยู่บรรทัดแรกเพื่อกัน SessionInfo error
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")

import os, re, json, shutil
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2, fitz, dateparser

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
    if not s: return None
    s = fix_numberlike(to_en_digits(s))
    s = s.replace(" ", "").replace(",", "").replace("฿","").replace("บาท","").replace("%","")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None

def parse_date_candidates(text: str) -> Optional[str]:
    t = sanitize_text(text)
    c = set()
    c.update(re.findall(r"(?:วันที่|date)[:\-\s]*([^\n]{1,40})", t, flags=re.I))
    c.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t))
    c.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", t))
    th = r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    c.update(re.findall(rf"\b\d{{1,2}}\s*{th}\s*\d{{2,4}}\b", t))
    parsed=[]
    for s in list(c)[:40]:
        dt = dateparser.parse(s, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year>2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    return sorted(parsed)[-1].isoformat() if parsed else None

# ---------------- Preprocessing ----------------
def binarize_for_tesseract(gray):
    # adaptive + ensure "black text on white"
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    # ถ้าส่วนขาวมากเกิน 0.8 แสดงว่าขาวทั้งหน้า → ใช้ Otsu
    if (th==255).mean() > 0.92:
        _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ถ้าขาวมากกว่า 0.6 → กลับสี (ต้องการพื้นขาว)
    if (th==0).mean() > 0.6:
        th = 255 - th
    return th

def deskew(binary_img: np.ndarray)->Tuple[np.ndarray,float]:
    coords = np.column_stack(np.where(binary_img<128))  # จุดดำเป็นตัวหนังสือ
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
    # กรอง noise ด้วยความมั่นใจ
    if "conf" in df.columns:
        df = df[df["conf"].astype(float) > 40].copy()
    df["text"] = df["text"].astype(str)
    df["norm"] = df["text"].str.lower().str.replace(r"[^a-z0-9ก-๙]+","", regex=True)
    return df.reset_index(drop=True)

def ocr_text_best(img_bin) -> str:
    # ทดลองหลาย psm เลือกอันที่ยาวสุด
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

def find_line(ln: pd.DataFrame, include: List[str], exclude: List[str]=None, prefer_last=False) -> Optional[pd.Series]:
    if exclude is None: exclude=[]
    inc=[re.sub(r"[^a-z0-9ก-๙]+","",k.lower()) for k in include]
    exc=[re.sub(r"[^a-z0-9ก-๙]+","",k.lower()) for k in exclude]
    def ok(s):
        s = re.sub(r"[^a-z0-9ก-๙]+","", s.lower())
        return any(k in s for k in inc) and not any(x in s for x in exc)
    cand = ln[ln["text"].apply(ok)]
    if cand.empty: return None
    cand = cand.sort_values(["page_num","top","left"])
    return cand.iloc[-1] if prefer_last else cand.iloc[0]

def right_tokens(df_words: pd.DataFrame, line_row: pd.Series, cutoff: Optional[int]=None) -> List[str]:
    if cutoff is None:
        cutoff = line_row["left"] + (line_row["right"]-line_row["left"])//2
    mask = (df_words["page_num"]==line_row["page_num"]) & \
           (df_words["block_num"]==line_row["block_num"]) & \
           (df_words["par_num"]==line_row["par_num"]) & \
           (df_words["line_num"]==line_row["line_num"]) & \
           (df_words["left"]>cutoff+5)
    return df_words[mask].sort_values("left")["text"].tolist()

def rightmost_number_on_line(df_words: pd.DataFrame, line_row: pd.Series) -> Optional[float]:
    mask = (df_words["page_num"]==line_row["page_num"]) & \
           (df_words["block_num"]==line_row["block_num"]) & \
           (df_words["par_num"]==line_row["par_num"]) & \
           (df_words["line_num"]==line_row["line_num"])
    sub = df_words[mask].sort_values("left")
    nums=[]
    for _,r in sub.iterrows():
        if re.fullmatch(r"\d[\d,]*\.?\d*", r["text"]):
            nums.append((r["left"], normalize_number(r["text"])))
    if nums: return nums[-1][1]
    return None

# ---------------- Vendor / Header / Amounts ----------------
BAD_VENDOR = r"(customer|address|project|quotation|page[:\s]|date[:\s])"
def extract_vendor(df_words: pd.DataFrame, page_h:int)->Optional[str]:
    ln = lines_from_df(df_words)
    head = ln[ln["top"] < page_h*0.25]
    if head.empty: head = ln[ln["top"] < page_h*0.35]
    cand = head[~head["text"].str.contains(BAD_VENDOR, flags=re.I, regex=True, na=False)]
    pri = pd.Series(0, index=cand.index, dtype=float)
    pri += cand["text"].str.contains(r"บริษัท|จำกัด", regex=True).astype(int)*2
    pri += cand["text"].str.contains(r"co\.,?\s*ltd\.?|company\s*limited", flags=re.I, regex=True).astype(int)*2
    pri += cand["text"].str.contains(r"solutions|consultants|askme", flags=re.I, regex=True).astype(int)
    if not cand.empty:
        return cand.loc[pri.idxmax()]["text"].strip()
    return None

def extract_header(df_words: pd.DataFrame)->Tuple[Optional[str], Optional[str]]:
    ln = lines_from_df(df_words)
    ql = find_line(ln, ["quotation no","quotation"], prefer_last=False)
    qt, dt = None, None
    if ql is not None:
        tx = " ".join(right_tokens(df_words, ql))
        qs = re.findall(r"[A-Za-z][A-Za-z0-9/_\-.]{5,}", tx)  # เช่น KS2209191
        if qs: qt = max(qs, key=len).upper()
        dt = parse_date_candidates(tx)
    if dt is None:
        dl = find_line(ln, ["date","วันที่"], prefer_last=False)
        if dl is not None:
            dt = parse_date_candidates(" ".join(right_tokens(df_words, dl)))
    # fallback หา code จากทั้งหน้า
    if qt is None:
        alltxt = " ".join(ln["text"].tolist())
        m = re.search(r"\b[A-Z]{1,3}[0-9]{6,}\b", alltxt)
        if m: qt = m.group(0)
    return qt, dt

def extract_amounts(df_words: pd.DataFrame, page_w:int)->Tuple[Optional[float], Optional[float], Optional[float]]:
    ln = lines_from_df(df_words)
    # เฉพาะคอลัมน์ขวา (จำนวนเงิน)
    right_lines = ln[(ln["right"] > page_w*0.55)]
    gl = find_line(right_lines, ["grand total","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"], prefer_last=True)
    vl = find_line(right_lines, ["vat","ภาษีมูลค่าเพิ่ม"], prefer_last=True)
    sl = find_line(right_lines, ["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","total"], exclude=["grand","vat"], prefer_last=True)

    grand = rightmost_number_on_line(df_words, gl) if gl is not None else None
    vat   = rightmost_number_on_line(df_words, vl) if vl is not None else None
    sub   = rightmost_number_on_line(df_words, sl) if sl is not None else None

    # Fallback: เลือก 3 บรรทัดล่างสุดที่เป็นจำนวนเงินในคอลัมน์ขวา
    if grand is None or sub is None:
        money_rows=[]
        for _,r in right_lines.iterrows():
            v = rightmost_number_on_line(df_words, r)
            if v is not None: money_rows.append((r["top"], v, r))
        money_rows = sorted(money_rows, key=lambda x:x[0])  # ตามแนวตั้ง
        if len(money_rows)>=2:
            # สองค่าล่างสุดคือ grand / vat|subtotal
            tail = [v for _,v,_ in money_rows[-3:]]
            tail = sorted(tail)
            # ค่ามากสุด = grand
            if grand is None: grand = tail[-1]
            # ถ้ามี 7% ให้คำนวณ vat/sub
            if sub is None and grand is not None and len(tail)>=2:
                # เดาทาง: ที่รองลงมาอาจเป็น subtotal
                sub = tail[-2]
            if vat is None and grand is not None and sub is not None:
                vat = round(grand - sub, 2)

    # Heuristic 7%
    all_text = " ".join(ln["text"].tolist())
    if vat is not None and vat < 50 and re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", all_text, flags=re.I):
        if grand is not None and sub is not None: vat = round(grand - sub, 2)
        elif sub is not None: vat = round(sub * 0.07, 2)

    # Reconcile
    if grand is None and sub is not None and vat is not None: grand = round(sub + vat, 2)
    if sub is None and grand is not None and vat is not None: sub = round(grand - vat, 2)
    if vat is None and grand is not None and sub is not None: vat = round(grand - sub, 2)
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

st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง (right-column guided)")
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
            col1,col2 = st.columns([1,1])
            with col1:
                if show_steps:
                    tabs = st.tabs(list(steps.keys()))
                    for tb,k in zip(tabs, steps.keys()):
                        with tb: st.image(steps[k], caption=f"{k} (page {pidx})", use_container_width=True)

            df_words = ocr_data(steps["upscale(1.8x)"])
            page_h, page_w = steps["original"].shape[:2]

            vendor = extract_vendor(df_words, page_h)
            qt, dt = extract_header(df_words)
            sub, vat, grand = extract_amounts(df_words, page_w)

            try:
                raw = ocr_text_best(steps["morph_open"])
            except Exception as e:
                raw = f"[Tesseract error] {e}"

            with col2:
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
