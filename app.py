# ต้องอยู่บรรทัดแรกเพื่อกัน SessionInfo error
import streamlit as st
st.set_page_config(page_title="OCR ใบเสนอราคา/บิล → สรุปตาราง", layout="wide")

import os, re, json, shutil, math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2, fitz, dateparser
from difflib import SequenceMatcher

# ============== OCR libs ==============
try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None

try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

# ============== Text helpers ==============
TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")
TH_MONTHS = {
    "ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน",
    "พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม",
    "ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"
}
ALLOW_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ก-๙/@#%()[]{}:;.,-+_ "
ALLOW_REGEX = r"[A-Za-z0-9ก-๙/@#%()\[\]\{\}:;.,\-\+_ ]"

def to_en_digits(s: str) -> str:
    return s.translate(TH_DIGITS) if isinstance(s,str) else s

def fix_numberlike(s: str) -> str:
    if not isinstance(s,str): return s
    # ตัวที่สลับบ่อยในเอกสารสแกน
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
    t = t.replace("—","-").replace("–","-").replace("：",":")
    # keep only allowed (ลด noise อักษรแตก)
    t = "".join(ch for ch in t if re.match(ALLOW_REGEX, ch))
    t = re.sub(r"[ \t]+"," ", t)
    return t

def normalize_number(s: str) -> Optional[float]:
    if not s: return None
    s = fix_numberlike(to_en_digits(s)).strip()
    s = s.replace("฿","").replace("บาท","").replace(" ", "")
    # EU: 1.234,56
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", s):
        s = s.replace(".","").replace(",",".")
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None

def parse_date_candidates(text: str) -> Optional[str]:
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

# ============== Preprocess ==============
def _clahe(gray):
    return cv2.createCLAHE(2.0,(8,8)).apply(gray)

def _adaptive(gray):
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    if (th==255).mean()>0.92:
        _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (th==0).mean()>0.6:
        th = 255 - th
    return th

def _deskew(bin_img):
    coords = np.column_stack(np.where(bin_img<128))
    if coords.size==0: return bin_img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h,w) = bin_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    rot = cv2.warpAffine(bin_img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle

def _remove_lines(bin_img):
    inv = 255 - bin_img
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(55,1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,55))
    h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    mask = cv2.bitwise_or(h,v)
    clean = cv2.inpaint(inv, mask, 3, cv2.INPAINT_TELEA)
    return 255 - clean

def build_variants(bgr:np.ndarray)->Dict[str,np.ndarray]:
    out={}
    out["original"]=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); out["gray"]=g
    c = _clahe(g); out["clahe"]=c
    b = _adaptive(c); out["binary"]=b
    d,_ = _deskew(b); out["deskew"]=d
    nl = _remove_lines(d); out["no_lines"]=nl
    up = cv2.resize(nl, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC); out["up"]=up
    sharp = cv2.GaussianBlur(up,(0,0),1); sharp = cv2.addWeighted(up,1.6, sharp,-0.6,0); out["sharp"]=sharp
    er = cv2.erode(up, np.ones((2,2),np.uint8), 1); out["erode"]=er
    di = cv2.dilate(up, np.ones((2,2),np.uint8), 1); out["dilate"]=di
    return out

# ============== Tesseract wrappers ==============
def ensure_tesseract(user_path:Optional[str]):
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
    exe = shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd=exe
            pytesseract.get_tesseract_version()
            return True,exe,None
        except Exception as e:
            return False,exe,str(e)
    return False,None,"tesseract not found"

def _lang_ratio(text:str)->float:
    if not text: return 0.0
    good=sum(1 for ch in text if re.match(ALLOW_REGEX, ch))
    return good/max(1,len(text))

def tesseract_full(img_bin:np.ndarray, psm:int=6)->Tuple[str,pd.DataFrame,float]:
    txt = pytesseract.image_to_string(img_bin, config=f"--oem 3 --psm {psm} -l tha+eng")
    df  = pytesseract.image_to_data(img_bin, config=f"--oem 3 --psm {psm} -l tha+eng", output_type=Output.DATAFRAME)
    df  = df.dropna(subset=["text"]).copy()
    conf = df["conf"].astype(float)
    conf = conf[conf>=0]
    mean_conf = float(conf.mean()) if len(conf) else 0.0
    score = 0.6*(mean_conf/100.0) + 0.4*_lang_ratio(txt)
    return txt, df, score

def ocr_best_of(variants:Dict[str,np.ndarray])->Tuple[str,pd.DataFrame,str]:
    best_score=-1; best=( "", pd.DataFrame(), "" )
    # จำกัดรอบเพื่อความเร็ว
    PSMs=[6,4,11]
    ROT=[0,90,180,270]
    ORDER=["sharp","up","no_lines","deskew","binary","erode","dilate"]
    for name in ORDER:
        if name not in variants: continue
        im = variants[name]
        for rot in ROT:
            src = im if rot==0 else cv2.rotate(im,
                    cv2.ROTATE_90_CLOCKWISE if rot==90 else
                    cv2.ROTATE_180 if rot==180 else cv2.ROTATE_90_COUNTERCLOCKWISE)
            for p in PSMs:
                try:
                    txt, df, s = tesseract_full(src, p)
                    if s>best_score:
                        best_score=s; best=(txt, df, f"{name}/psm{p}/rot{rot}")
                except Exception:
                    continue
        if best_score>0.72:  # ได้ดีแล้วพอ
            break
    # รวมผลจาก EasyOCR เพื่อเติมคำ (ตัวเลือก)
    if _HAS_EASYOCR:
        try:
            reader = easyocr.Reader(['th','en'], gpu=False)
            res = reader.readtext(variants["up"])
            extra = "\n".join([r[1] for r in res if r[2]>=0.45])
            if len(extra) > 10:
                best = (best[0] + "\n" + extra, best[1], best[2]+"+easy")
        except Exception:
            pass
    return best

def ocr_numeric(img_bin:np.ndarray)->str:
    return pytesseract.image_to_string(img_bin, config="--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789., ")

# ============== Group to lines ==============
def lines_from_df(df:pd.DataFrame)->pd.DataFrame:
    g=["page_num","block_num","par_num","line_num"]
    agg = df.groupby(g).agg(left=("left","min"), top=("top","min"),
                            right=("left","max"), bottom=("top","max"),
                            height=("height","max")).reset_index()
    texts = df.groupby(g)["text"].apply(lambda s:" ".join([x for x in s if x.strip()])).reset_index(name="text")
    ln = agg.merge(texts,on=g)
    ln["right"] = ln["right"] + df.groupby(g)["width"].max().values
    return ln

# ============== Helpers (fuzzy) ==============
def _norm(s:str)->str: return re.sub(r"[^a-z0-9ก-๙]+","", (s or "").lower())
def _ratio(a,b):
    a,b=_norm(a),_norm(b)
    if not a or not b: return 0.0
    sm = SequenceMatcher(None,a,b).ratio()
    if len(a)<len(b):
        best=0
        for i in range(0,len(b)-len(a)+1):
            best=max(best, SequenceMatcher(None,a,b[i:i+len(a)]).ratio())
        sm=max(sm,best)
    return sm

COMMON_FIX = {
    "quotation no":["quotationno","quotatlonno","quote no","ref no","เลขที่ใบเสนอราคา","quotation number"],
    "date":["วันที่","quotation date","issue date","issued date","doc date","เอกสารลงวันที่"],
    "subtotal":["รวมก่อนภาษี","ยอดก่อนภาษี","net total","amount before vat","sub-total"],
    "vat":["vat7%","ภาษีมูลค่าเพิ่ม","vat amount","tax"],
    "grand total":["ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ","total amount","amount due"]
}
def expand_keys(keys:List[str])->List[str]:
    out=set(keys)
    for k in keys:
        base=_norm(k)
        for canon,alts in COMMON_FIX.items():
            if _norm(canon)==base: out.update(alts)
    return list(out)

def find_line_fuzzy(ln:pd.DataFrame, keys:List[str], cutoff:float=0.7)->Optional[pd.Series]:
    if ln is None or len(ln)==0 or "text" not in ln.columns: return None
    inc = expand_keys(keys)
    scored=[]
    for _,r in ln.iterrows():
        t=r.get("text","")
        sc = max(_ratio(t,k) for k in inc)
        if sc>=cutoff: scored.append((sc,r))
    if not scored: return None
    scored.sort(key=lambda t:(t[0], t[1].get("page_num",0), t[1].get("top",1e9), t[1].get("left",1e9)))
    return scored[-1][1]

def tokens_right(df_words:pd.DataFrame, anchor:pd.Series, max_dx:int=900, dy_factor:float=1.5)->List[str]:
    h=int(anchor["height"])
    top_min=int(anchor["top"]-max(12,h*0.4))
    top_max=int(anchor["top"]+h*dy_factor)
    mask=(df_words["page_num"]==anchor["page_num"]) & \
         (df_words["left"]>anchor["right"]+2) & (df_words["left"]<anchor["right"]+max_dx) & \
         (df_words["top"]>=top_min) & (df_words["top"]<=top_max)
    return df_words[mask].sort_values(["top","left"])["text"].tolist()

# ============== Vendor / Header / Amounts ==============
def _clean_vendor(s:str)->str:
    s=" ".join(s.split())
    m=re.search(r"(?<![A-Za-z]\s)([A-Za-z][A-Za-z '&\.\-]+?(?:Co\.,?\s*Ltd\.|Company\s*Limited))", s, flags=re.I)
    if m:
        v=m.group(1).strip()
        v=re.sub(r"^[A-Za-z]\s+(?=[A-Za-z])","",v)  # ตัดเศษตัวอักษรเดี่ยว (เช่น 'M ')
        v=re.sub(r"\s*,\s*",", ",v)
        v=re.sub(r"\s+Co\.,?\s*Ltd\.?"," Co., Ltd.",v,flags=re.I)
        v=re.sub(r"\s+Company\s+Limited"," Company Limited",v,flags=re.I)
        return re.sub(r"\s{2,}"," ",v).strip()
    m=re.search(r"(บริษัท.+?(?:จำกัด\(มหาชน\)|จำกัด))", s)
    if m: return m.group(1).strip()
    return s.strip()

def extract_vendor(df_words:pd.DataFrame, page_h:int)->Optional[str]:
    ln = lines_from_df(df_words)
    head = ln[ln["top"]<page_h*0.35]
    bad=r"(customer|address|project|quotation|page[:\s]|date[:\s])"
    head = head[~head["text"].str.contains(bad, flags=re.I, regex=True, na=False)]
    text=" ".join(head.sort_values(["top","left"])["text"].tolist())
    text=" ".join(text.split())
    for pat,flg in ((r"(?<![A-Za-z]\s)([A-Za-z][A-Za-z '&\.\-]+?(?:Co\.,?\s*Ltd\.|Company\s*Limited))",re.I),
                    (r"(บริษัท.+?(?:จำกัด\(มหาชน\)|จำกัด))",0)):
        m=re.search(pat,text,flags=flg)
        if m: return _clean_vendor(m.group(1))
    if not head.empty:
        pri = pd.Series(0, index=head.index, dtype=float)
        pri += head["text"].str.contains(r"บริษัท|จำกัด", regex=True).astype(int)*2
        pri += head["text"].str.contains(r"co\.,?\s*ltd\.?|company\s*limited", flags=re.I, regex=True).astype(int)*2
        pri += head["text"].str.contains(r"solutions|consultants|broadband|3bb|askme", flags=re.I, regex=True).astype(int)
        return _clean_vendor(head.loc[pri.idxmax()]["text"])
    return None

def extract_header(df_words:pd.DataFrame)->Tuple[Optional[str],Optional[str]]:
    ln=lines_from_df(df_words)
    qt=dt=None
    ql=find_line_fuzzy(ln,["quotation no","quotation"],cutoff=0.66)
    if ql is not None:
        tx=" ".join(tokens_right(df_words, ql))
        m=re.search(r"\b[A-Z]{1,3}\d{6,}\b",tx)
        if m: qt=m.group(0)
        if qt is None:
            tokens=re.findall(r"[A-Za-z][A-Za-z0-9/_\-\.]{5,}",tx)
            if tokens: qt=max(tokens,key=len).upper()
    dl=find_line_fuzzy(ln,["date","วันที่"],cutoff=0.55)
    if dl is not None:
        dt=parse_date_candidates(" ".join(tokens_right(df_words, dl)))
    if dt is None:
        dt=parse_date_candidates(" ".join(ln["text"].tolist()))
    if qt is None:
        m=re.search(r"\b[A-Z]{1,3}\d{6,}\b"," ".join(ln["text"].tolist()))
        if m: qt=m.group(0)
    return qt, dt

def rightmost_number_on_line(df_words:pd.DataFrame, row:pd.Series)->Optional[float]:
    mask=(df_words["page_num"]==row["page_num"]) & \
         (df_words["block_num"]==row["block_num"]) & \
         (df_words["par_num"]==row["par_num"]) & \
         (df_words["line_num"]==row["line_num"])
    sub=df_words[mask].sort_values("left")
    nums=[]
    for _,r in sub.iterrows():
        if re.fullmatch(r"\d[\d,\.]*",r["text"]): nums.append((r["left"], normalize_number(r["text"])))
    return nums[-1][1] if nums else None

AMT_KEYS = {
    "subtotal":["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","net total","amount before vat"],
    "vat":["vat","vat 7%","ภาษีมูลค่าเพิ่ม","vat amount","tax"],
    "grand":["grand total","รวมทั้งสิ้น","ยอดรวมสุทธิ","ยอดชำระสุทธิ","total amount","amount due"]
}

def extract_amounts(df_words:pd.DataFrame, page_w:int, page_h:int)->Tuple[Optional[float],Optional[float],Optional[float]]:
    ln=lines_from_df(df_words)
    zones=[(0.55,0.58,0.98,0.98),(0.02,0.60,0.48,0.98),(0.50,0.40,0.98,0.98)]
    cand=[]
    def pick(z):
        if z is None or len(z)==0: return (None,None,None)
        g=find_line_fuzzy(z,AMT_KEYS["grand"],0.55)
        v=find_line_fuzzy(z,AMT_KEYS["vat"],0.50)
        s=find_line_fuzzy(z,AMT_KEYS["subtotal"],0.50)
        G=rightmost_number_on_line(df_words,g) if g is not None else None
        V=rightmost_number_on_line(df_words,v) if v is not None else None
        S=rightmost_number_on_line(df_words,s) if s is not None else None
        return (S,V,G)
    for x1,y1,x2,y2 in zones:
        z=ln[(ln["left"]>=page_w*x1)&(ln["right"]<=page_w*x2)&(ln["top"]>=page_h*y1)&(ln["bottom"]<=page_h*y2)]
        cand.append(pick(z))
    cand.append(pick(ln[ln["right"]>page_w*0.45]))
    # best by equation
    best=(None,None,None); best_err=1e18
    for s,v,g in cand:
        if s is None and v is not None and g is not None: s=round(g-v,2)
        if v is None and s is not None and g is not None: v=round(g-s,2)
        if g is None and s is not None and v is not None: g=round(s+v,2)
        if s is None and v is None and g is None: continue
        err=abs((s or 0)+(v or 0)-(g or 0)) if (s and v and g) else 0.09
        if err<best_err: best_err, best=err,(s,v,g)
    # heuristic 7%
    alltxt=" ".join(ln["text"].tolist())
    if re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", alltxt, flags=re.I):
        s,v,g=best
        if g and s and (v is None or v<50): v=round(g-s,2)
        best=(s,v,g)
    return best

# ============== PDF helper ==============
def pdf_to_bgr_list(file_bytes:bytes, dpi:int=300)->List[np.ndarray]:
    out=[]
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pix=p.get_pixmap(dpi=dpi, alpha=False)
            img=np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            out.append(img[:,:,::-1])
    return out

# ============== Google Sheets ==============
def export_to_google_sheets(df:pd.DataFrame, sheet_url:str, service_json:dict, worksheet_name:str="OCR_QT"):
    try:
        import gspread
        gc=gspread.service_account_from_dict(service_json)
        sh=gc.open_by_url(sheet_url)
        try: ws=sh.worksheet(worksheet_name)
        except Exception: ws=sh.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        if not ws.get_all_values(): ws.append_row(list(df.columns))
        for _,row in df.iterrows():
            ws.append_row([("" if v is None else str(v)) for v in row.tolist()])
        return True,"Exported to Google Sheets successfully."
    except Exception as e:
        return False,f"Export failed: {e}"

# ============== UI ==============
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    user_tess = st.text_input("Tesseract path (ถ้าไม่เจอให้ระบุ)", "")
    show_steps = st.checkbox("แสดงภาพ Pre-processing", True)
    worksheet = st.text_input("Worksheet (Google Sheets)", "OCR_QT")
    st.markdown("---")
    st.subheader("🔗 ส่งออก Google ชีท (ตัวเลือก)")
    sheet_url = st.text_input("ลิงก์ Google ชีท (แชร์สิทธิ์แก้ไขให้ Service Account)")
    svc_json = st.file_uploader("อัปโหลด Service Account JSON", type=["json"])

ok, loc, msg = ensure_tesseract(user_tess or None)
st.sidebar.write("**Tesseract:** ", ("✅ "+str(loc)) if ok else ("❌ "+str(msg)))

st.title("🧾 OCR ใบเสนอราคา/บิล → สรุปตาราง (Multi-variant + Auto-score)")
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

        for pidx, bgr in enumerate(pages, start=1):
            variants = build_variants(bgr)
            # แสดงขั้นตอน
            if show_steps:
                cols = st.columns(3)
                for i,k in enumerate(["original","gray","binary","deskew","no_lines","up","sharp"]):
                    with cols[i%3]:
                        img = variants[k]
                        if img.ndim==2: st.image(img, caption=f"{k} (page {pidx})", use_column_width=True, clamp=True)
                        else: st.image(img, caption=f"{k} (page {pidx})", use_column_width=True)

            if not ok:
                st.error("ไม่พบ Tesseract ในระบบ"); continue

            # เลือกผล OCR ที่ดีที่สุด
            raw, df_words, meta = ocr_best_of(variants)
            raw_clean = sanitize_text(raw)

            # สร้าง “ข้อความอ่านง่าย” จากบรรทัด
            try:
                ln = lines_from_df(df_words)
                pretty = "\n".join([" ".join(sanitize_text(x).split()) for x in ln.sort_values(["top","left"])["text"].tolist() if x.strip()])
                if len(pretty) > len(raw_clean)*0.6:
                    raw_clean = pretty
            except Exception:
                pass

            st.caption(f"OCR best: {meta}")
            st.text_area(f"OCR Output (Clean Text) — page {pidx}", value=raw_clean, height=240)

            # ดึงฟิลด์
            page_h, page_w = variants["original"].shape[:2]
            if len(df_words)==0:  # safety
                df_words = pytesseract.image_to_data(variants["up"], config="--oem 3 --psm 6 -l tha+eng", output_type=Output.DATAFRAME).dropna(subset=["text"])
            vendor = extract_vendor(df_words, page_h)
            quo_no, doc_date = extract_header(df_words)
            sub, vat, grand = extract_amounts(df_words, page_w, page_h)

            rec = {
                "file": f"{up.name}#p{pidx}",
                "Vendor / Supplier": vendor,
                "Quotation No.": quo_no,
                "Date": doc_date,
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
    st.download_button("⬇️ ดาวน์โหลด CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ocr_quotation_results.csv", mime="text/csv")
    if sheet_url and svc_json is not None:
        try:
            svc = json.load(svc_json)
            ok2,msg2 = export_to_google_sheets(df, sheet_url, svc, worksheet_name=worksheet)
            (st.success if ok2 else st.error)(msg2)
        except Exception as e:
            st.error(f"อ่าน Service JSON ไม่ได้: {e}")
