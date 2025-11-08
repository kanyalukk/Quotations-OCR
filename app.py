# ต้องอยู่เป็นบรรทัดแรกเพื่อกัน SessionInfo error
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")

import os, re, json, shutil
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import dateparser
import fitz  # PyMuPDF

# --- OCR: pytesseract ---
try:
    import pytesseract
except Exception:
    pytesseract = None

# ----------------------------- helpers: text -----------------------------
TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}
def to_en_digits(s:str)->str: return s.translate(TH_DIGITS) if isinstance(s,str) else s
def fix_numberlike_ocr(s:str)->str:
    if not isinstance(s,str): return s
    s = re.sub(r'(?<=\d)[oO](?=[\d,\.])','0',s)
    s = re.sub(r'(?<=[,\.\s])[oO](?=\d)','0',s)
    s = re.sub(r'(?<=\d)[lI](?=[\d,\.])','1',s)
    s = re.sub(r'(?<=\d)B(?=[\d,\.])','8',s)
    return s
def sanitize_text(t:str)->str:
    if not t: return ""
    t = to_en_digits(t)
    for k,v in TH_MONTHS.items(): t = re.sub(k,v,t)
    t = re.sub(r"[ \t]+"," ",t).replace("—","-").replace("–","-").replace("：",":")
    return t
def normalize_number(s:str)->Optional[float]:
    if not s: return None
    s = fix_numberlike_ocr(to_en_digits(s))
    s = s.replace(" ","").replace(",","").replace("฿","").replace("บาท","").replace("%","")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None
def parse_date_candidates(text:str)->Optional[str]:
    t = sanitize_text(text)
    c=set()
    c.update(re.findall(r"(?:วันที่|date)[:\-\s]*([^\n]{1,40})",t,flags=re.I))
    c.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",t))
    c.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",t))
    th=r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    c.update(re.findall(rf"\b\d{{1,2}}\s*{th}\s*\d{{2,4}}\b",t))
    parsed=[]
    for s in list(c)[:40]:
        dt = dateparser.parse(s,languages=["th","en"],settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year>2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    return (sorted(parsed)[-1].isoformat() if parsed else None)

# --------------------------- preprocessing ---------------------------
def deskew(binary_img: np.ndarray)->Tuple[np.ndarray,float]:
    coords=np.column_stack(np.where(binary_img>0))
    if coords.size==0: return binary_img,0.0
    angle=cv2.minAreaRect(coords)[-1]
    angle=-(90+angle) if angle<-45 else -angle
    (h,w)=binary_img.shape[:2]
    M=cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
    rotated=cv2.warpAffine(binary_img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return rotated,angle
def preprocess(img_bgr: np.ndarray)->Dict[str,np.ndarray]:
    out={}
    out["original"]=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY); out["grayscale"]=gray
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray); out["clahe"]=clahe
    blur=cv2.medianBlur(clahe,3); out["denoise(median3)"]=blur
    th=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9); out["adaptive_threshold"]=th
    de,_=deskew(th); out["deskewed"]=de
    up=cv2.resize(de,None,fx=1.7,fy=1.7,interpolation=cv2.INTER_CUBIC); out["upscale(1.7x)"]=up
    opened=cv2.morphologyEx(up,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),iterations=1); out["morph_open"]=opened
    return out

# ---------------------------- tesseract ----------------------------
def ensure_tesseract(user_path: Optional[str]):
    if pytesseract is None: return (False,None,"pytesseract not installed")
    cand=[]
    if user_path: cand.append(user_path)
    cand+=["/usr/bin/tesseract","/usr/local/bin/tesseract","/opt/homebrew/bin/tesseract",
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

def tesseract_text(img, psm=6, lang="tha+eng", whitelist=None)->str:
    cfg=f"--oem 3 --psm {psm} -l {lang}"
    if whitelist: cfg+=f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, config=cfg)

def tesseract_df(img_rgb)->pd.DataFrame:
    cfg="--oem 3 --psm 6 -l tha+eng"
    df=pytesseract.image_to_data(img_rgb, config=cfg, output_type=pytesseract.Output.DATAFRAME)
    df=df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df["norm"]=df["text"].str.lower().str.replace(r"[^a-z0-9ก-๙]+","",regex=True)
    return df

# ------------------------ line & ROI utilities ------------------------
def lines_from_df(df: pd.DataFrame)->pd.DataFrame:
    # สรุปเป็นบรรทัด (page, block, par, line)
    gcols=["page_num","block_num","par_num","line_num"]
    agg=df.groupby(gcols).agg(
        left=("left","min"), top=("top","min"),
        right=("left","max"), bottom=("top","max"),
        height=("height","max")
    ).reset_index()
    texts=df.groupby(gcols)["text"].apply(lambda s:" ".join([x for x in s if x.strip()])).reset_index(name="text")
    norms=df.groupby(gcols)["norm"].apply(lambda s:"".join([x for x in s if x.strip()])).reset_index(name="norm")
    ln=agg.merge(texts,on=gcols).merge(norms,on=gcols)
    ln["right"]=ln["right"]+df.groupby(gcols)["width"].max().values
    return ln

def find_line(ln: pd.DataFrame, keywords: List[str], prefer_last=False)->Optional[pd.Series]:
    ks=[re.sub(r"[^a-z0-9ก-๙]+","",k.lower()) for k in keywords]
    cand=ln[ln["norm"].apply(lambda s:any(k in s for k in ks))]
    if cand.empty: return None
    cand = cand.sort_values(["page_num","top","left"])
    return cand.iloc[-1] if prefer_last else cand.iloc[0]

def right_text_on_line(df_words: pd.DataFrame, line_row: pd.Series, cutoff_x: int)->str:
    mask = (df_words["page_num"]==line_row["page_num"]) & \
           (df_words["block_num"]==line_row["block_num"]) & \
           (df_words["par_num"]==line_row["par_num"]) & \
           (df_words["line_num"]==line_row["line_num"]) & \
           (df_words["left"]>cutoff_x+5)
    words = df_words[mask].sort_values("left")["text"].tolist()
    return " ".join(words)

def rightmost_number_on_line(df_words: pd.DataFrame, line_row: pd.Series)->Optional[float]:
    mask = (df_words["page_num"]==line_row["page_num"]) & \
           (df_words["block_num"]==line_row["block_num"]) & \
           (df_words["par_num"]==line_row["par_num"]) & \
           (df_words["line_num"]==line_row["line_num"])
    sub=df_words[mask].copy()
    # เอาเลขตัวที่อยู่ขวาสุดในบรรทัด
    sub=sub.sort_values("left")
    nums=[]
    for _,r in sub.iterrows():
        if re.fullmatch(r"\d[\d,]*\.?\d*", r["text"]):
            nums.append((r["left"], normalize_number(r["text"])))
    if nums:
        return nums[-1][1]
    return None

# ----------------------- vendor / fields extraction -----------------------
VENDOR_PAT = r"(บริษัท.+จำกัด|บริษัท.+จำกัด\(มหาชน\)|ห้างหุ้นส่วนจำกัด|หจก\.|co\.,?\s*ltd\.?|solutions\s*&\s*consultants\s*co\.,?\s*ltd\.?|company\s*limited)"

def extract_vendor_by_position(df_words: pd.DataFrame, page_h: int)->Optional[str]:
    ln = lines_from_df(df_words)
    top_band = ln[ln["top"] < page_h * 0.22]  # เฉพาะโซนหัวเอกสาร
    if top_band.empty: top_band = ln[ln["top"] < page_h * 0.35]
    # กรองคำว่า customer/project/address ออก
    def bad(s): return re.search(r"(customer|address|project|page[:\s]|date[:\s]|quotation)", s, flags=re.I)
    cand = top_band[~top_band["text"].apply(lambda s: bool(bad(s)))]
    # เลือกประโยคที่ match รูปแบบชื่อบริษัท + ยาวสุด
    m1=cand[cand["text"].str.contains(VENDOR_PAT, flags=re.I, regex=True)]
    if not m1.empty:
        return m1.iloc[m1["text"].str.len().argmax()]["text"].strip()
    # ถ้าไม่เจอ ให้เอาบรรทัดบนสุดที่ยาวสุด (มักเป็นชื่อบริษัทภาษาไทย)
    return cand.iloc[cand["text"].str.len().argmax()]["text"].strip() if not cand.empty else None

def extract_header_fields(df_words: pd.DataFrame)->Tuple[Optional[str], Optional[str]]:
    ln = lines_from_df(df_words)
    ql = find_line(ln, ["quotation no","quotation"], prefer_last=False)
    qt = None; dt = None
    if ql is not None:
        # cutoff ที่ท้าย token 'quotation' หรือ 'quotation no'
        cutoff = ql["left"] + (ql["right"] - ql["left"])//2
        right_txt = right_text_on_line(df_words, ql, cutoff)
        # โค้ดใบเสนอราคา: เอา alnum/_- ยาวที่สุด
        tokens = re.findall(r"[A-Za-z0-9/_\-.]{5,}", right_txt)
        if tokens:
            qt = max(tokens, key=len).upper()
        # date: หาในบรรทัดเดียวหรือบรรทัดถัดไป
        dt = parse_date_candidates(right_txt)
        if dt is None:
            # มองบรรทัดถัดไปใน block เดียวกัน
            next_line = ln[(ln["page_num"]==ql["page_num"]) & (ln["block_num"]==ql["block_num"]) &
                           (ln["par_num"]==ql["par_num"]) & (ln["line_num"]==ql["line_num"]+1)]
            if not next_line.empty:
                rt = right_text_on_line(df_words, next_line.iloc[0], next_line.iloc[0]["left"])
                dt = parse_date_candidates(rt)
    return qt, dt

def extract_amounts(df_words: pd.DataFrame)->Tuple[Optional[float], Optional[float], Optional[float]]:
    ln = lines_from_df(df_words)
    # ใช้ prefer_last=True เพื่อเลือกแถวด้านล่างของตาราง (ใกล้ผลรวม)
    gl = find_line(ln, ["grand total","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"], prefer_last=True)
    vl = find_line(ln, ["vat","ภาษีมูลค่าเพิ่ม"], prefer_last=True)
    sl = find_line(ln, ["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","total"], prefer_last=True)

    grand = rightmost_number_on_line(df_words, gl) if gl is not None else None
    vat   = rightmost_number_on_line(df_words, vl) if vl is not None else None
    sub   = rightmost_number_on_line(df_words, sl) if sl is not None else None

    # reconcile
    text_all = " ".join(lines_from_df(df_words)["text"].tolist())
    if vat is not None and vat < 50 and re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", text_all, flags=re.I):
        if grand is not None and sub is not None: vat = round(grand - sub, 2)
        elif sub is not None: vat = round(sub * 0.07, 2)
    if grand is None and sub is not None and vat is not None: grand = round(sub + vat, 2)
    if sub is None and grand is not None and vat is not None: sub = round(grand - vat, 2)
    if vat is None and grand is not None and sub is not None: vat = round(grand - sub, 2)

    return sub, vat, grand

# ------------------------------ PDF helper ------------------------------
def pdf_to_bgr_list(file_bytes: bytes, dpi: int = 300)->List[np.ndarray]:
    out=[]
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pix=p.get_pixmap(dpi=dpi, alpha=False)
            img=np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,pix.width,3)
            out.append(img[:,:,::-1])  # BGR
    return out

# ------------------------------ Google Sheets ------------------------------
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

# -------------------------------- UI --------------------------------
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
st.sidebar.write("**Tesseract:** ", "✅ "+str(TESS_PATH) if TESS_OK else "❌ "+str(TESS_MSG))

st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง (Guided by image_to_data)")

uploads = st.file_uploader("อัปโหลด JPG/PNG/PDF (หลายไฟล์ได้)", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)

rows=[]
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
                        with tb: st.image(steps[k], caption=f"{k} (page {pidx})", use_column_width=True)

            # ใช้ image_to_data เพื่อดึงข้อมูลแบบพิกัด
            df_words = tesseract_df(steps["original"])
            page_h = steps["original"].shape[0]

            vendor = extract_vendor_by_position(df_words, page_h)
            qt, dt = extract_header_fields(df_words)
            sub, vat, grand = extract_amounts(df_words)

            # debug raw text (ถ้าต้องการ)
            try:
                raw = tesseract_text(steps["morph_open"], psm=6, lang="tha+eng")
            except Exception as e:
                raw = f"[Tesseract error] {e}"
            with col2:
                st.text_area(f"OCR Output (Raw Text) — page {pidx}", value=raw, height=220)

                row = {
                    "file": f"{up.name}#p{pidx}",
                    "Vendor / Supplier": vendor,
                    "Quotation No.": qt,
                    "Date": dt,
                    "Subtotal": sub,
                    "VAT": vat,
                    "Grand Total": grand
                }
                st.write("**สรุปฟิลด์ที่สกัดได้**")
                st.dataframe(pd.DataFrame([row]))
                rows.append(row)

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
