# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล ⇒ ตาราง", layout="wide")

import os, re, json, shutil
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from PIL import Image
import cv2, fitz, dateparser

# ====== OCR backends ======
try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None

# EasyOCR (optional)
try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

# ====== text helpers ======
TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")
TH_MONTHS = {"ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
             "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
             "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"}

def to_en_digits(s:str)->str: return s.translate(TH_DIGITS) if isinstance(s,str) else s

def sanitize_text(t:str)->str:
    if not t: return ""
    t = to_en_digits(t)
    for k,v in TH_MONTHS.items(): t = t.replace(k, v)
    t = t.replace("—","-").replace("–","-").replace("：",":")
    return re.sub(r"[ \t]+"," ", t)

def normalize_number(s:str)->Optional[float]:
    if not s: return None
    s = to_en_digits(s)
    s = s.replace("฿","").replace("บาท","").strip()
    # รูปแบบยุโรป 1.234,56
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", s):
        s = s.replace(".","").replace(",",".")
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None

def parse_date_candidates(text:str)->Optional[str]:
    t = sanitize_text(text)
    c=set()
    c.update(re.findall(r"(?:วันที่|date|issued\s*date|quotation\s*date)[:\-\s]*([^\n]{1,40})", t, flags=re.I))
    c.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t))
    c.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", t))
    th = r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    c.update(re.findall(rf"\b\d{{1,2}}\s*{th}\s*\d{{2,4}}\b", t))
    parsed=[]
    for s in list(c)[:60]:
        dt = dateparser.parse(s, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year>2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    return (sorted(parsed)[-1].isoformat() if parsed else None)

# ====== preprocessing ======
def to_gray(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def adaptive_bin(gray):
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    # ป้องกันขาวโพลน/ดำเต้ม
    if (th==255).mean()>0.92:
        _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (th==0).mean()>0.6:
        th = 255 - th
    return th

def deskew(binary_img):
    coords = np.column_stack(np.where(binary_img<128))
    if coords.size==0: return binary_img,0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h,w) = binary_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rot = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle

def remove_table_lines(bin_img):
    inv = 255 - bin_img
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(55,1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,55))
    h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    mask = cv2.bitwise_or(h,v)
    clean = cv2.inpaint(inv, mask, 3, cv2.INPAINT_TELEA)
    return 255 - clean

def preprocess(bgr:np.ndarray, remove_lines:bool=True, scale:float=2.0)->Dict[str,np.ndarray]:
    out={}
    out["original"]=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    g = to_gray(bgr); out["gray"]=g
    g = cv2.createCLAHE(2.0,(8,8)).apply(g); out["clahe"]=g
    th = adaptive_bin(g); out["binary"]=th
    rot,_ = deskew(th); out["deskew"]=rot
    if remove_lines:
        rot = remove_table_lines(rot); out["no_lines"]=rot
    up = cv2.resize(rot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC); out["up"]=up
    sharp = cv2.GaussianBlur(up,(0,0),sigmaX=1)
    sharp = cv2.addWeighted(up, 1.5, sharp, -0.5, 0)  # sharpen
    out["sharp"]=sharp
    return out

# ====== tesseract utils ======
def ensure_tesseract(user_path:str=None):
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

def tesseract_text(img_bin)->str:
    outs=[]
    for psm in (6,4,3,11,12):
        outs.append(pytesseract.image_to_string(img_bin, config=f"--oem 3 --psm {psm} -l tha+eng"))
    return max(outs, key=len)

def tesseract_data(img_bin)->pd.DataFrame:
    df = pytesseract.image_to_data(img_bin, config="--oem 3 --psm 6 -l tha+eng", output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    if "conf" in df.columns:
        df = df[df["conf"].astype(float)>40]
    df["text"]=df["text"].astype(str)
    return df.reset_index(drop=True)

def tesseract_numeric(img_bin)->str:
    # whitelist เพื่ออ่านเฉพาะเลข/คอมมา/จุด
    return pytesseract.image_to_string(img_bin, config="--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789., ")

# EasyOCR fallback (บรรทัด)
def easyocr_lines(img_bin)->List[str]:
    if not _HAS_EASYOCR: return []
    reader = easyocr.Reader(['th','en'], gpu=False)
    res = reader.readtext(img_bin)
    lines = [txt for (*_, txt, prob) in res if prob>=0.4]
    return lines

# ====== layout helpers ======
def lines_from_df(df:pd.DataFrame)->pd.DataFrame:
    g=["page_num","block_num","par_num","line_num"]
    agg = df.groupby(g).agg(left=("left","min"), top=("top","min"),
                            right=("left","max"), bottom=("top","max"),
                            height=("height","max")).reset_index()
    txt = df.groupby(g)["text"].apply(lambda s:" ".join([x for x in s if x.strip()])).reset_index(name="text")
    ln = agg.merge(txt,on=g)
    ln["right"] = ln["right"] + df.groupby(g)["width"].max().values
    return ln

def tokens_right(df_words:pd.DataFrame, anchor:pd.Series, max_dx:int=1200, dy_factor:float=1.8)->List[str]:
    h = int(anchor["height"])
    top_min = int(anchor["top"] - max(12, h*0.4))
    top_max = int(anchor["top"] + h*dy_factor)
    mask = (df_words["page_num"]==anchor["page_num"]) & \
           (df_words["left"] > anchor["right"]+2) & (df_words["left"] < anchor["right"]+max_dx) & \
           (df_words["top"] >= top_min) & (df_words["top"] <= top_max)
    return df_words[mask].sort_values(["top","left"])["text"].tolist()

def find_line_fuzzy(ln:pd.DataFrame, keys:List[str], cutoff:float=0.6)->Optional[pd.Series]:
    def norm(s): return re.sub(r"[^a-z0-9ก-๙]+","", s.lower())
    ex = [norm(k) for k in keys]
    cand = []
    for _,r in ln.iterrows():
        t = norm(r["text"])
        sc = max([similarity(t, k) for k in ex] + [0])
        cand.append((sc, r))
    cand = [r for sc,r in cand if sc>=cutoff]
    if not cand: return None
    return sorted(cand, key=lambda x:(x[0], x[1]["top"], x[1]["left"]))[-1][1]

def similarity(a,b):
    # substring-friendly ratio
    if not a or not b: return 0.0
    import difflib
    if len(a) < len(b):
        best = 0
        for i in range(len(b)-len(a)+1):
            best = max(best, difflib.SequenceMatcher(None, a, b[i:i+len(a)]).ratio())
        return best
    return difflib.SequenceMatcher(None, a, b).ratio()

# ====== extraction ======
VENDOR_PAT_EN = r"([A-Za-z][A-Za-z &\.\-]+?(?:Co\.,?\s*Ltd\.|Company\s*Limited|PCL))"
VENDOR_PAT_TH = r"(บริษัท.+?(?:จำกัด\(มหาชน\)|จำกัด))"

def clean_vendor_line(s:str)->str:
    s=" ".join(s.split())
    m=re.search(VENDOR_PAT_EN, s, flags=re.I)
    if m:
        v=m.group(1)
        v=re.sub(r"\s*,\s*",", ",v)
        v=re.sub(r"\s+Co\.,?\s*Ltd\.?"," Co., Ltd.",v,flags=re.I)
        v=re.sub(r"\s+Company\s+Limited"," Company Limited",v,flags=re.I)
        return re.sub(r"\s{2,}"," ",v).strip()
    m=re.search(VENDOR_PAT_TH, s)
    if m: return m.group(1).strip()
    return s.strip()

def extract_vendor(df_words:pd.DataFrame, page_h:int)->Optional[str]:
    ln = lines_from_df(df_words)
    head = ln[ln["top"] < page_h*0.4]
    text_head = " ".join(head.sort_values(["top","left"])["text"].tolist())
    text_all  = " ".join(ln.sort_values(["top","left"])["text"].tolist())
    for tx in (text_head, text_all):
        for pat,flg in ((VENDOR_PAT_TH,0), (VENDOR_PAT_EN,re.I)):
            m=re.search(pat, tx, flags=flg)
            if m: return clean_vendor_line(m.group(1))
    return None

QUO_KEYS = ["quotation no", "quotation", "quo no", "quo. no", "ref no", "เลขที่ใบเสนอราคา", "เลขที่"]
DATE_KEYS = ["date","วันที่","quotation date","issue date","issued date","doc date","ใบเสนอราคาวันที่","เอกสารลงวันที่"]

def extract_header(df_words:pd.DataFrame)->Tuple[Optional[str], Optional[str]]:
    ln = lines_from_df(df_words)
    qt = dt = None

    ql = find_line_fuzzy(ln, QUO_KEYS, cutoff=0.55)
    if ql is not None:
        tx = " ".join(tokens_right(df_words, ql))
        m  = re.search(r"\b[A-Z0-9]{2,}[A-Z0-9\-_/]{3,}\b", tx)
        if m: qt = m.group(0)

    dl = find_line_fuzzy(ln, DATE_KEYS, cutoff=0.55)
    if dl is not None:
        dt = parse_date_candidates(" ".join(tokens_right(df_words, dl)))
    if dt is None:
        dt = parse_date_candidates(" ".join(ln["text"].tolist()))
    if qt is None:
        m=re.search(r"\b[A-Z]{1,6}[A-Z0-9\-_/]{5,}\b", " ".join(ln["text"].tolist()))
        if m: qt=m.group(0)
    return qt, dt

def rightmost_number_on_line(df_words:pd.DataFrame, line_row:pd.Series)->Optional[float]:
    mask=(df_words["page_num"]==line_row["page_num"]) & \
         (df_words["block_num"]==line_row["block_num"]) & \
         (df_words["par_num"]==line_row["par_num"]) & \
         (df_words["line_num"]==line_row["line_num"])
    sub=df_words[mask].sort_values("left")
    nums=[]
    for _,r in sub.iterrows():
        if re.fullmatch(r"\d[\d,\.]*", r["text"]):
            nums.append((r["left"], normalize_number(r["text"])))
    return nums[-1][1] if nums else None

AMT_KEYS = {
    "subtotal": ["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","net total","amount before vat"],
    "vat":      ["vat","vat 7%","ภาษีมูลค่าเพิ่ม","vat amount","tax"],
    "grand":    ["grand total","รวมทั้งสิ้น","ยอดรวมสุทธิ","ราคาขายรวมภาษี","ยอดชำระสุทธิ","total amount","amount due"]
}

def find_amounts_by_zone(df_words:pd.DataFrame, page_w:int, page_h:int)->List[Tuple[Optional[float],Optional[float],Optional[float]]]:
    ln = lines_from_df(df_words)
    zones = [
        (0.55, 0.58, 0.98, 0.98),  # ขวาล่าง
        (0.02, 0.60, 0.48, 0.98),  # ซ้ายล่าง
        (0.50, 0.40, 0.98, 0.98),  # ขวาครึ่งล่าง
    ]
    cand=[]
    def pick(lines):
        g=find_line_fuzzy(lines, AMT_KEYS["grand"], cutoff=0.55)
        v=find_line_fuzzy(lines, AMT_KEYS["vat"],   cutoff=0.50)
        s=find_line_fuzzy(lines, AMT_KEYS["subtotal"], cutoff=0.50)
        G=rightmost_number_on_line(df_words, g) if g is not None else None
        V=rightmost_number_on_line(df_words, v) if v is not None else None
        S=rightmost_number_on_line(df_words, s) if s is not None else None
        return S,V,G
    for x1,y1,x2,y2 in zones:
        z = ln[(ln["left"]>=page_w*x1)&(ln["right"]<=page_w*x2)&(ln["top"]>=page_h*y1)&(ln["bottom"]<=page_h*y2)]
        cand.append(pick(z))
    cand.append(pick(ln[ln["right"]>page_w*0.45]))
    return cand

def find_amounts_by_keyword_roi(img_bin:np.ndarray, df_words:pd.DataFrame)->List[Tuple[Optional[float],Optional[float],Optional[float]]]:
    """ค้นคำสำคัญบนทั้งหน้าแล้วตัด ROI บริเวณนั้น อ่านเลขด้วย whitelist"""
    ln = lines_from_df(df_words)
    outs=[]
    for keys in [AMT_KEYS["grand"], AMT_KEYS["vat"], AMT_KEYS["subtotal"]]:
        l = find_line_fuzzy(ln, keys, cutoff=0.52)
        if l is None: continue
        x1 = max(int(l["left"] - 10), 0)
        y1 = max(int(l["top"]  - 10), 0)
        x2 = min(int(l["right"] + 800), img_bin.shape[1]-1)
        y2 = min(int(l["bottom"]+ 80), img_bin.shape[0]-1)
        roi = img_bin[y1:y2, x1:x2]
        txt = tesseract_numeric(roi)
        outs.append((keys[0], normalize_number(txt)))
    # รวมผลที่ได้ให้ครบชุด
    s=v=g=None
    for k,val in outs:
        if any("grand" in kk or "รวมทั้งสิ้น" in kk or "ราคาขายรวม" in kk for kk in [k]): g = val
        if any("vat" in kk or "ภาษี" in kk for kk in [k]): v = val
        if any("subtotal" in kk or "ก่อนภาษี" in kk for kk in [k]): s = val
    return [(s,v,g)] if (s or v or g) else []

def choose_best_amount(cands:List[Tuple[Optional[float],Optional[float],Optional[float]]], raw_text:str)->Tuple[Optional[float],Optional[float],Optional[float]]:
    if not cands: return (None,None,None)
    best=(None,None,None); best_err=1e18
    for s,v,g in cands:
        if s is None and v is not None and g is not None: s = round(g - v, 2)
        if v is None and s is not None and g is not None: v = round(g - s, 2)
        if g is None and s is not None and v is not None: g = round(s + v, 2)
        if s is None and v is None and g is None: continue
        err = abs((s+v)-g) if (s is not None and v is not None and g is not None) else 0.09
        if err < best_err:
            best_err, best = err, (s,v,g)
    # ถ้าพบ 7% ในข้อความ ช่วยคำนวณ VAT
    if re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", raw_text, flags=re.I):
        s,v,g = best
        if g and s and (v is None or v < 50): v = round(g - s, 2)
        best = (s,v,g)
    return best

# ====== pdf helper ======
def pdf_to_bgr_list(file_bytes:bytes, dpi:int=300)->List[np.ndarray]:
    out=[]
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pix = p.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            out.append(img[:,:,::-1])
    return out

# ====== Google Sheets ======
def export_to_google_sheets(df:pd.DataFrame, sheet_url:str, service_json:dict, worksheet_name:str="OCR_QT"):
    try:
        import gspread
        gc = gspread.service_account_from_dict(service_json)
        sh = gc.open_by_url(sheet_url)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        if not ws.get_all_values(): ws.append_row(list(df.columns))
        for _,row in df.iterrows(): ws.append_row([("" if v is None else str(v)) for v in row.tolist()])
        return True,"Exported to Google Sheets successfully."
    except Exception as e:
        return False, f"Export failed: {e}"

# ====== UI ======
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    tess_path = st.text_input("Tesseract path (optional)", "")
    scale = st.slider("Upscale ×", 1.4, 2.6, 2.0, 0.1)
    rm_lines = st.checkbox("ลบเส้นตารางก่อน OCR", True)
    use_easy = st.checkbox("เปิดใช้ EasyOCR fallback (ช้าลงเล็กน้อย)", False and _HAS_EASYOCR)
    worksheet_name = st.text_input("Worksheet (Google Sheets)", "OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("ลิงก์ Google ชีท (แชร์สิทธิ์แก้ไขให้ Service Account)")
    svc_json = st.file_uploader("อัปโหลด Service Account JSON", type=["json"])

ok, loc, msg = ensure_tesseract(tess_path or None)
st.sidebar.write("**Tesseract:** ", ("✅ "+loc) if ok else ("❌ "+str(msg)))

st.title("🧾 OCR ใบเสนอราคา/บิล ⇒ ตาราง (Tesseract + ROI หลายโซน)")
files = st.file_uploader("อัปโหลด JPG/PNG/PDF (หลายไฟล์ได้)", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)

records=[]
if files:
    for f in files:
        st.markdown("---")
        st.write(f"**ไฟล์:** {f.name}")
        if f.type=="application/pdf" or f.name.lower().endswith(".pdf"):
            pages = pdf_to_bgr_list(f.read())
        else:
            im = Image.open(f).convert("RGB")
            pages = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)]

        for pidx, bgr in enumerate(pages, 1):
            steps = preprocess(bgr, remove_lines=rm_lines, scale=scale)

            # แสดงภาพขั้นตอน
            tabs = st.tabs(["original","gray","binary","deskew","no_lines" if rm_lines else "—","up","sharp"])
            show_keys = ["original","gray","binary","deskew"] + (["no_lines"] if rm_lines else []) + ["up","sharp"]
            for i,k in enumerate(show_keys):
                with tabs[i]:
                    img = steps[k]
                    if img.ndim==2:
                        st.image(img, use_column_width=True, clamp=True, caption=f"{k} (page {pidx})")
                    else:
                        st.image(img, use_column_width=True, caption=f"{k} (page {pidx})")

            if not ok:
                st.error("ไม่พบ Tesseract ในระบบ"); continue

            page_h, page_w = steps["original"].shape[:2]
            df_words = tesseract_data(steps["sharp"])
            raw_tess = tesseract_text(steps["sharp"])

            # EasyOCR (เสริม)
            raw_easy = "\n".join(easyocr_lines(steps["sharp"])) if use_easy and _HAS_EASYOCR else ""
            raw_all = (raw_tess + "\n" + raw_easy).strip()

            st.text_area(f"OCR Output (Raw Text) — page {pidx}", value=raw_all, height=200)

            vendor = extract_vendor(df_words, page_h)
            qt, dt = extract_header(df_words)

            cand = []
            cand += find_amounts_by_zone(df_words, page_w, page_h)
            cand += find_amounts_by_keyword_roi(steps["sharp"], df_words)
            sub, vat, grand = choose_best_amount(cand, raw_all)

            rec = {
                "file": f"{f.name}#p{pidx}",
                "Vendor / Supplier": vendor,
                "Quotation No.": qt,
                "Date": dt,
                "Subtotal": sub,
                "VAT": vat,
                "Grand Total": grand
            }
            st.dataframe(pd.DataFrame([rec]), use_container_width=True)
            records.append(rec)

if records:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df = pd.DataFrame(records, columns=["file","Vendor / Supplier","Quotation No.","Date","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลด CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="ocr_quotation_results.csv", mime="text/csv")
    if sheet_url and svc_json is not None:
        try:
            svc = json.load(svc_json)
            ok2,msg2 = export_to_google_sheets(df, sheet_url, svc, worksheet_name=worksheet_name)
            (st.success if ok2 else st.error)(msg2)
        except Exception as e:
            st.error(f"อ่าน Service JSON ไม่ได้: {e}")
