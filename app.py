# ----- app.py (ready to run) -----
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → ตาราง", layout="wide")

import os, re, json, shutil
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2, fitz, dateparser
from difflib import SequenceMatcher

# ---------------- OCR backend ----------------
try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None

# ---------------- text helpers ----------------
TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")
TH_MONTHS = {"ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
             "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
             "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"}

def to_en_digits(s:str)->str: return s.translate(TH_DIGITS) if isinstance(s,str) else s
def fix_numberlike(s:str)->str:
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
    return re.sub(r"[ \t]+"," ", t).replace("—","-").replace("–","-").replace("：",":")
def normalize_number(s:str)->Optional[float]:
    if not s: return None
    s = fix_numberlike(to_en_digits(s)).strip().replace("฿","").replace("บาท","")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", s): s = s.replace(".","").replace(",",".")
    s = s.replace(" ","").replace(",","")
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
    for s in list(c)[:50]:
        dt = dateparser.parse(s, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year>2400: dt = dt.replace(year=dt.year-543)
            parsed.append(dt.date())
    return (sorted(parsed)[-1].isoformat() if parsed else None)

# ---------------- preprocessing ----------------
def binarize(gray):
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    if (th==255).mean() > 0.92: _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (th==0).mean() > 0.6: th = 255 - th
    return th

def deskew(binary_img):
    coords = np.column_stack(np.where(binary_img<128))
    if coords.size==0: return binary_img,0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h,w) = binary_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    rot = cv2.warpAffine(binary_img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle

def remove_table_lines(bin_img):
    inv = 255 - bin_img
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,50))
    h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    mask = cv2.bitwise_or(h,v)
    clean = cv2.inpaint(inv, mask, 3, cv2.INPAINT_TELEA)
    return 255 - clean

def preprocess(bgr:np.ndarray)->Dict[str,np.ndarray]:
    out={}
    out["original"]=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); out["grayscale"]=gray
    clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray); out["clahe"]=clahe
    th = binarize(clahe); out["binary"]=th
    rot,_ = deskew(th); out["deskewed"]=rot
    no_lines = remove_table_lines(rot); out["no_lines"]=no_lines
    up = cv2.resize(no_lines, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC); out["upscale(1.8x)"]=up
    open1 = cv2.morphologyEx(up, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations=1); out["morph_open"]=open1
    return out

# ---------------- tesseract utils ----------------
def ensure_tesseract(path:Optional[str]):
    if pytesseract is None: return (False,None,"pytesseract not installed")
    c=[]
    if path: c.append(path)
    c += ["/usr/bin/tesseract","/usr/local/bin/tesseract","/opt/homebrew/bin/tesseract",
          r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for p in c:
        if os.path.exists(p):
            try:
                pytesseract.pytesseract.tesseract_cmd=p
                pytesseract.get_tesseract_version(); return True,p,None
            except Exception: pass
    exe = shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd=exe
            pytesseract.get_tesseract_version(); return True,exe,None
        except Exception as e: return False,exe,str(e)
    return False,None,"tesseract not found"

def ocr_data(img_bin)->pd.DataFrame:
    df = pytesseract.image_to_data(img_bin, config="--oem 3 --psm 6 -l tha+eng", output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    if "conf" in df.columns: df = df[df["conf"].astype(float)>40]
    df["text"]=df["text"].astype(str)
    df["norm"]=df["text"].str.lower().str.replace(r"[^a-z0-9ก-๙]+","", regex=True)
    return df.reset_index(drop=True)

def ocr_text_best(img_bin)->str:
    outs=[]
    for psm in (6,4,11,12): outs.append(pytesseract.image_to_string(img_bin, config=f"--oem 3 --psm {psm} -l tha+eng"))
    return max(outs, key=len)

# ---------------- layout utils ----------------
def lines_from_df(df:pd.DataFrame)->pd.DataFrame:
    g=["page_num","block_num","par_num","line_num"]
    agg=df.groupby(g).agg(left=("left","min"),top=("top","min"),
                          right=("left","max"),bottom=("top","max"),
                          height=("height","max")).reset_index()
    texts=df.groupby(g)["text"].apply(lambda s:" ".join([x for x in s if x.strip()])).reset_index(name="text")
    norms=df.groupby(g)["norm"].apply(lambda s:" ".join([x for x in s if x.strip()])).reset_index(name="norm")
    ln=agg.merge(texts,on=g).merge(norms,on=g); ln["right"]=ln["right"]+df.groupby(g)["width"].max().values
    return ln

def _norm(s:str)->str: return re.sub(r"[^a-z0-9ก-๙]+","", (s or "").lower())
def _ratio(a:str,b:str)->float:
    a,b=_norm(a),_norm(b)
    if not a or not b: return 0.0
    sm=SequenceMatcher(None,a,b).ratio()
    if len(a)<len(b):
        best=0
        for i in range(0,len(b)-len(a)+1): best=max(best,SequenceMatcher(None,a,b[i:i+len(a)]).ratio())
        sm=max(sm,best)
    return sm

COMMON_FIX={
    "quotation no":[
        "quotationno","quotatlonno","quotatlon","quotation №","เลขที่ใบเสนอราคา",
        "quotation number","quotation id","quote no","ref no","quo no","quo. no"
    ],
    "date":[
        "วันที่","oate","dare","quotation date","issue date","issued date","ใบเสนอราคาวันที่","เอกสารลงวันที่","doc date"
    ],
    "subtotal":["sub-total","sutotal","subtota|","subtotai","subtotl","รวมก่อนภาษี","ยอดก่อนภาษี","net total","amount before vat"],
    "vat":["vat7%","vat 7 %","va7","ภาษีมูลค่าเพิ่ม","vat:","tax","vat amount"],
    "grand total":["grandtotai","grandtotl","grand tota|","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ","total amount","amount due","total due"]
}
def expand_keywords(keys:List[str])->List[str]:
    out=set()
    for k in keys:
        out.add(k); base=_norm(k)
        for canon,alts in COMMON_FIX.items():
            if _norm(canon)==base: out.update(alts)
    return list(out)
def find_line_fuzzy(ln:pd.DataFrame, include:List[str], exclude:List[str]=None, prefer_last=True, cutoff:float=0.72)->Optional[pd.Series]:
    if exclude is None: exclude=[]
    inc=expand_keywords(include); exc=[_norm(x) for x in exclude]
    cand=ln.copy()
    def score(text):
        t=_norm(text)
        if any(x in t for x in exc): return 0.0
        return max(_ratio(t,_norm(k)) for k in inc)
    cand["__score__"]=cand["text"].apply(score); cand=cand[cand["__score__"]>=cutoff]
    if cand.empty: return None
    cand=cand.sort_values(["page_num","top","left","__score__"])
    return cand.iloc[-1] if prefer_last else cand.iloc[-1]

def tokens_right(df_words:pd.DataFrame, anchor:pd.Series, max_dx:int=1200, dy_factor:float=1.8)->List[str]:
    h=int(anchor["height"])
    top_min=int(anchor["top"]-max(12,h*0.4)); top_max=int(anchor["top"]+h*dy_factor)
    mask=(df_words["page_num"]==anchor["page_num"]) & \
         (df_words["left"]>anchor["right"]+2) & (df_words["left"]<anchor["right"]+max_dx) & \
         (df_words["top"]>=top_min) & (df_words["top"]<=top_max)
    return df_words[mask].sort_values(["top","left"])["text"].tolist()

# ---------------- extraction ----------------
def _clean_vendor_line(s:str)->str:
    s=" ".join(s.split())
    m=re.search(r"(?<![A-Za-z]\s)([A-Za-z][A-Za-z '&\.\-]+?(?:Co\.,?\s*Ltd\.|Company\s*Limited|PCL))", s, flags=re.I)
    if m:
        v=m.group(1).strip()
        v=re.sub(r"^[A-Za-z]\s+(?=[A-Za-z])","",v)
        v=re.sub(r"\s*,\s*",", ",v)
        v=re.sub(r"\s+Co\.,?\s*Ltd\.?"," Co., Ltd.",v,flags=re.I)
        v=re.sub(r"\s+Company\s+Limited"," Company Limited",v,flags=re.I)
        v=re.sub(r"\s{2,}"," ",v).strip()
        return v
    m=re.search(r"(บริษัท.+?(?:จำกัด\(มหาชน\)|จำกัด))", s)
    if m: return m.group(1).strip()
    return s.strip()

def extract_vendor(df_words:pd.DataFrame, page_h:int)->Optional[str]:
    ln = lines_from_df(df_words)
    head = ln[ln["top"] < page_h*0.35]
    if head.empty: head = ln[ln["top"] < page_h*0.50]
    BAD=r"(customer|address|project|quotation|page[:\s]|date[:\s])"
    head=head[~head["text"].str.contains(BAD,flags=re.I,regex=True,na=False)]
    text_all=" ".join(ln.sort_values(["top","left"])["text"].tolist())
    text_head=" ".join(head.sort_values(["top","left"])["text"].tolist())
    for tx in (text_head, text_all):
        m=re.search(r"(บริษัท.+?(?:จำกัด\(มหาชน\)|จำกัด))", tx)
        if m: return _clean_vendor_line(m.group(1))
        m=re.search(r"([A-Za-z][A-Za-z '&\.\-]+?(?:Co\.,?\s*Ltd\.|Company\s*Limited|PCL))", tx, flags=re.I)
        if m: return _clean_vendor_line(m.group(1))
    return None

def extract_header(df_words:pd.DataFrame)->Tuple[Optional[str],Optional[str]]:
    ln=lines_from_df(df_words)
    qt=dt=None
    ql=find_line_fuzzy(ln,["quotation no","quo no","quotation"], cutoff=0.62)
    if ql is not None:
        tx=" ".join(tokens_right(df_words, ql))
        m=re.search(r"\b[A-Z]{1,6}[-]?[A-Z0-9]{2,}[-]?[A-Z0-9\-]{2,}\b", tx)
        if m: qt=m.group(0)
        if qt is None:
            qs=re.findall(r"[A-Za-z][A-Za-z0-9/_\-.]{5,}", tx)
            if qs: qt=max(qs,key=len).upper()
    dl=find_line_fuzzy(ln,["date","วันที่"], cutoff=0.55)
    if dl is not None: dt=parse_date_candidates(" ".join(tokens_right(df_words, dl)))
    if dt is None: dt=parse_date_candidates(" ".join(ln["text"].tolist()))
    if qt is None:
        m=re.search(r"\b[A-Z]{1,6}[A-Z0-9\-]{5,}\b"," ".join(ln["text"].tolist()))
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
        if re.fullmatch(r"\d[\d,\.]*", r["text"]): nums.append((r["left"], normalize_number(r["text"])))
    return nums[-1][1] if nums else None

def extract_amounts(df_words:pd.DataFrame, page_w:int, page_h:int)->Tuple[Optional[float],Optional[float],Optional[float]]:
    ln=lines_from_df(df_words)
    ZONES=[(0.55,0.58,0.98,0.98),(0.02,0.60,0.48,0.98)]  # ขวาล่าง + ซ้ายล่าง
    cand=[]
    def in_lines(lines):
        gl=find_line_fuzzy(lines,["grand total"], cutoff=0.58)
        vl=find_line_fuzzy(lines,["vat"], cutoff=0.52)
        sl=find_line_fuzzy(lines,["subtotal"], exclude=["grand","vat"], cutoff=0.52)
        g=rightmost_number_on_line(df_words,gl) if gl is not None else None
        v=rightmost_number_on_line(df_words,vl) if vl is not None else None
        s=rightmost_number_on_line(df_words,sl) if sl is not None else None
        return s,v,g
    for x1,y1,x2,y2 in ZONES:
        zone=ln[(ln["left"]>=page_w*x1)&(ln["right"]<=page_w*x2)&(ln["top"]>=page_h*y1)&(ln["bottom"]<=page_h*y2)]
        cand.append(in_lines(zone))
    zone_all=ln[ln["right"]>page_w*0.50]; cand.append(in_lines(zone_all))
    money=[]
    for _,r in ln.iterrows():
        v=rightmost_number_on_line(df_words,r)
        if v is not None: money.append((r["top"],v))
    money=sorted(money,key=lambda x:x[0])[-6:]; tail=[v for _,v in money]
    if len(tail)>=2:
        t=sorted(tail); cand.append((t[-2],None,t[-1]))
    best=(None,None,None); err_best=1e18
    for s,v,g in cand:
        if s is None and v is not None and g is not None: s=round(g-v,2)
        if v is None and s is not None and g is not None: v=round(g-s,2)
        if g is None and s is not None and v is not None: g=round(s+v,2)
        if s is None and v is None and g is None: continue
        err=abs((s+v)-g) if (s is not None and v is not None and g is not None) else 0.09
        if err<err_best: err_best=err; best=(s,v,g)
    s,v,g=best
    alltxt=" ".join(ln["text"].tolist())
    if re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", alltxt, flags=re.I):
        if g and s and (v is None or v<50): v=round(g-s,2)
    return s,v,g

# ---------------- PDF helper ----------------
def pdf_to_bgr_list(file_bytes:bytes, dpi:int=300)->List[np.ndarray]:
    out=[]
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pix=p.get_pixmap(dpi=dpi, alpha=False)
            img=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
            out.append(img[:,:,::-1])
    return out

# ---------------- Google Sheets ----------------
def export_to_google_sheets(df:pd.DataFrame, sheet_url:str, service_json:dict, worksheet_name:str="OCR_QT"):
    try:
        import gspread
        gc=gspread.service_account_from_dict(service_json)
        sh=gc.open_by_url(sheet_url)
        try: ws=sh.worksheet(worksheet_name)
        except Exception: ws=sh.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        if not ws.get_all_values(): ws.append_row(list(df.columns))
        for _,row in df.iterrows(): ws.append_row([("" if v is None else str(v)) for v in row.tolist()])
        return True,"Exported to Google Sheets successfully."
    except Exception as e:
        return False,f"Export failed: {e}"

# ---------------- UI ----------------
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    tess_path = st.text_input("Tesseract path (ถ้าไม่เจอให้ระบุ)", "")
    show_steps = st.checkbox("แสดงภาพ Pre-processing", True)
    worksheet = st.text_input("Worksheet (Google Sheets)", "OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("ลิงก์ Google ชีท")
    svc_file = st.file_uploader("Service Account JSON", type=["json"])

TESS_OK, TESS_LOC, TESS_MSG = ensure_tesseract(tess_path.strip() or None)
st.sidebar.write("**Tesseract:** ", "✅ "+str(TESS_LOC) if TESS_OK else "❌ "+str(TESS_MSG))

st.title("🧾 OCR ใบเสนอราคา/บิล → สรุปเป็นตาราง (ลบเส้นตาราง + โซนหลายแบบ)")
uploads = st.file_uploader("อัปโหลด JPG/PNG/PDF (หลายไฟล์ได้)", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)

records=[]
if uploads:
    for up in uploads:
        st.markdown("---"); st.write(f"**ไฟล์:** {up.name}")
        if up.type=="application/pdf" or up.name.lower().endswith(".pdf"):
            pages = pdf_to_bgr_list(up.read())
        else:
            im = Image.open(up).convert("RGB"); pages=[cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)]
        for pidx,bgr in enumerate(pages, start=1):
            steps = preprocess(bgr)
            page_h, page_w = steps["original"].shape[:2]

            if show_steps:
                cols = st.columns(5)
                view = ["original","grayscale","no_lines","upscale(1.8x)","morph_open"]
                for i,k in enumerate(view):
                    with cols[i%5]:
                        img=steps[k]
                        if img.ndim==2: st.image(img, caption=f"{k} (p{pidx})", use_column_width=True, clamp=True)
                        else: st.image(img, caption=f"{k} (p{pidx})", use_column_width=True)

            if not TESS_OK:
                st.error("ไม่พบ Tesseract"); continue

            df_words = ocr_data(steps["morph_open"])
            vendor = extract_vendor(df_words, page_h)
            qt, dt = extract_header(df_words)
            sub, vat, grand = extract_amounts(df_words, page_w, page_h)

            try: raw = ocr_text_best(steps["morph_open"])
            except Exception as e: raw = f"[Tesseract error] {e}"

            st.text_area(f"OCR Output (Raw Text) — page {pidx}", value=raw, height=220)
            rec = {"file":f"{up.name}#p{pidx}","Vendor / Supplier":vendor,"Quotation No.":qt,
                   "Date":dt,"Subtotal":sub,"VAT":vat,"Grand Total":grand}
            st.dataframe(pd.DataFrame([rec]), use_container_width=True)
            records.append(rec)

if records:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df = pd.DataFrame(records, columns=["file","Vendor / Supplier","Quotation No.","Date","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลด CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ocr_quotation_results.csv", mime="text/csv")
    if sheet_url and svc_file is not None:
        try:
            svc = json.load(svc_file)
            ok,msg = export_to_google_sheets(df, sheet_url, svc, worksheet_name=worksheet)
            (st.success if ok else st.error)(msg)
        except Exception as e:
            st.error(f"อ่าน Service JSON ไม่ได้: {e}")
