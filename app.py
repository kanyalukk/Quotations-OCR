
# app.py content abbreviated in this cell to ensure tool runs quickly.
# (Full content was generated in the previous step; re-writing in full here)
import os, re, json, shutil
from typing import Dict, List, Tuple, Optional
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
import cv2, dateparser
EASYOCR_AVAILABLE=True
try:
    import easyocr
except Exception:
    EASYOCR_AVAILABLE=False
try:
    import pytesseract
except Exception:
    pytesseract=None

THAI_DIGITS=str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")
TH_MONTHS={"ม.ค.":"มกราคม","ก.พ.":"กุมภาพันธ์","มี.ค.":"มีนาคม","เม.ย.":"เมษายน","พ.ค.":"พฤษภาคม","มิ.ย.":"มิถุนายน","ก.ค.":"กรกฎาคม","ส.ค.":"สิงหาคม","ก.ย.":"กันยายน","ต.ค.":"ตุลาคม","พ.ย.":"พฤศจิกายน","ธ.ค.":"ธันวาคม"}

def to_english_digits(s): return s.translate(THAI_DIGITS) if isinstance(s,str) else s
def fix_numberlike_ocr(s):
    s=re.sub(r'(?<=\d)[oO](?=[\d,\.])','0',s); s=re.sub(r'(?<=[,\.\s])[oO](?=\d)','0',s)
    s=re.sub(r'(?<=\d)[lI](?=[\d,\.])','1',s); s=re.sub(r'(?<=\d)B(?=[\d,\.])','8',s); return s
def sanitize_text(text):
    if not text: return ""
    text=to_english_digits(text)
    for k,v in TH_MONTHS.items(): text=re.sub(k,v,text)
    text=re.sub(r"[ \t]+"," ",text); return text.replace("—","-").replace("–","-").replace("：",":")
def normalize_number(s):
    if not s: return None
    s=fix_numberlike_ocr(to_english_digits(s)).replace(" ","").replace(",","").replace("฿","").replace("บาท","").replace("%","")
    m=re.findall(r"-?\d+(?:\.\d+)?",s); return float(m[0]) if m else None

def ensure_tesseract(user_path=None):
    if pytesseract is None: return (False,None,"pytesseract not installed")
    c=[]; 
    if user_path: c.append(user_path)
    c+=["/usr/bin/tesseract","/usr/local/bin/tesseract","/opt/homebrew/bin/tesseract",r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"]
    for p in c:
        if os.path.exists(p):
            try:
                pytesseract.pytesseract.tesseract_cmd=p; pytesseract.get_tesseract_version(); return True,p,None
            except Exception as e: last=str(e)
    exe=shutil.which("tesseract")
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd=exe; pytesseract.get_tesseract_version(); return True,exe,None
        except Exception as e:
            return False,exe,str(e)
    return False,None,"not found"

def ocr_easyocr(img):
    if not EASYOCR_AVAILABLE: return ""
    reader=easyocr.Reader(["th","en"],gpu=False); res=reader.readtext(img,detail=0,paragraph=True); return "\n".join(res)
def ocr_tesseract(img): 
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6 -l tha+eng")

def last_amount_in_line(line):
    no_pct=re.sub(r"\d+(\.\d+)?\s*%","",line); no_pct=fix_numberlike_ocr(no_pct)
    nums=re.findall(r"-?\d[\d,]*\.?\d*",no_pct); 
    for n in reversed(nums):
        v=normalize_number(n)
        if v is not None and ("," in n or "." in n or v>=100): return v
    return normalize_number(nums[-1]) if nums else None

def amount_by_label(text, include, exclude=[]):
    t=sanitize_text(text); lines=[l.strip() for l in t.splitlines() if l.strip()]
    idxs=[i for i,l in enumerate(lines) if any(k.lower() in l.lower() for k in include) and not any(b.lower() in l.lower() for b in exclude)]
    if not idxs: return None
    i=idxs[-1]; v=last_amount_in_line(lines[i]); 
    if v is None and i+1<len(lines): v=last_amount_in_line(lines[i+1])
    return v

def normalize_qt_code(raw):
    if not raw: return raw
    s=raw.strip().translate(str.maketrans({"$":"S","§":"S"})); s=re.sub(r"^[Kk]5","KS",s); s=re.sub(r"^5","S",s)
    return fix_numberlike_ocr(s).replace(" ","").upper()

def ner_extract_quotation_no(text):
    t=sanitize_text(text); lines=t.splitlines()
    for i,l in enumerate(lines):
        if re.search(r"quotation\s*no\.?", l, re.I):
            after=re.split(r"quotation\s*no\.?\s*[:#]?\s*", l, flags=re.I)[-1]
            m=re.search(r"([A-Z\$§]{0,5}[A-Z0-9\$§/_\-.]{3,})", after, re.I)
            if m and not re.fullmatch(r"date", m.group(1), re.I): return normalize_qt_code(m.group(1))
            look=" ".join(lines[i+1:i+3])
            date_m=re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", look)
            tokens=re.findall(r"[A-Z\$§]?[A-Z0-9\$§/_\-.]{3,}", look, re.I)
            if tokens:
                if date_m:
                    cut=look[:date_m.start()]
                    cand=re.findall(r"[A-Z\$§]?[A-Z0-9\$§/_\-.]{3,}", cut, re.I)
                    if cand: return normalize_qt_code(cand[-1])
                return normalize_qt_code(tokens[0])
    m=re.search(r"\b[A-Z\$§]{1,4}[-/_.]?\d{2,4}[-/_.]?\d{1,6}\b", t, re.I)
    return normalize_qt_code(m.group(0)) if m else None

def parse_date_candidates(text):
    t=sanitize_text(text)
    c=set()
    for m in re.finditer(r"(วันที่|date)[:\s\-]*([^\n]{0,40})", t, re.I): c.add(m.group(0))
    c.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t))
    c.update(re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", t))
    th=r"(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
    c.update(re.findall(rf"\b\d{{1,2}}\s*{th}\s*\d{{2,4}}\b", t))
    arr=[]
    for s in list(c)[:30]:
        dt=dateparser.parse(s, languages=["th","en"], settings={"PREFER_DATES_FROM":"past","DATE_ORDER":"DMY"})
        if dt:
            if dt.year>2400: dt=dt.replace(year=dt.year-543)
            arr.append(dt.date())
    return sorted(arr)[-1].isoformat() if arr else None

def extract_description(text):
    t=sanitize_text(text)
    if "product description" in t.lower():
        after=t.lower().split("product description",1)[1]
        after=re.split(r"(payment\s*term|terms\s*&?\s*conditions|total|vat|grand\s*total)", after, flags=re.I)[0]
        after=re.sub(r"\b(qty\.?|price per.*|total price.*)\b","",after,flags=re.I)
        line=[ln.strip() for ln in after.splitlines() if len(ln.strip())>=10]
        if line: return line[0][:180]
    return None

def extract_fields(full_text):
    txt=sanitize_text(full_text or "")
    vendor=None
    # simple vendor grab: first line having บริษัท/.../co., ltd.
    for l in txt.splitlines()[:20]:
        if re.search(r"(บริษัท|หจก\.|co\.,?\s*ltd\.?|company\s*limited)", l, re.I):
            vendor=l.strip(); break
    qt=ner_extract_quotation_no(txt)
    date_iso=parse_date_candidates(txt)
    subtotal=amount_by_label(txt, include=["subtotal","รวมก่อนภาษี","ยอดก่อนภาษี","total"], exclude=["grand","vat"])
    vat=amount_by_label(txt, include=["vat","ภาษีมูลค่าเพิ่ม"], exclude=[])
    grand=amount_by_label(txt, include=["grand total","ยอดรวมสุทธิ","รวมทั้งสิ้น","ยอดชำระสุทธิ"], exclude=[])
    if vat is not None and vat<50 and re.search(r"vat\s*7\s*%|ภาษี\s*7\s*%", txt, re.I):
        if grand is not None and subtotal is not None: vat=round(grand-subtotal,2)
        elif subtotal is not None: vat=round(subtotal*0.07,2)
    if grand is None and subtotal is not None and vat is not None: grand=round(subtotal+vat,2)
    if subtotal is None and grand is not None and vat is not None: subtotal=round(grand-vat,2)
    if vat is None and grand is not None and subtotal is not None: vat=round(grand-subtotal,2)
    desc=extract_description(txt)
    return {"Vendor/Supplier":vendor,"Quotation No.":qt,"Date":date_iso,"Description":desc,"Subtotal":subtotal,"VAT":vat,"Grand Total":grand,"Raw Text":txt}

# ---- UI ----
st.set_page_config(page_title="OCR Quotation/Bill → Table", layout="wide")
st.title("🧾 OCR ใบเสนอราคา/บิล ➜ สรุปเป็นตาราง (Regex + NER)")
with st.sidebar:
    engine=st.selectbox("OCR Engine",["Hybrid (EasyOCR ➜ fallback Tesseract)","Tesseract only","EasyOCR only"],index=0)
    user_tess_path=st.text_input("Tesseract path (ถ้าระบบหาไม่เจอ)",value="")
    show_steps=st.checkbox("แสดงภาพ Pre-processing ทุกขั้นตอน",value=True)
    st.markdown("---"); st.subheader("🔗 Google Sheets")
    sheet_url=st.text_input("ลิงก์ Google ชีท",value=""); service_json_file=st.file_uploader("Service Account JSON",type=["json"])
def preprocess(img_bgr):
    out={}
    out["original"]=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY); out["grayscale"]=gray
    blur=cv2.medianBlur(gray,3); out["denoise(median3)"]=blur
    th=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9); out["adaptive_threshold"]=th
    out["morph_open"]=cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations=1); return out
TESS_OK, TESS_PATH, TESS_MSG=ensure_tesseract(user_tess_path.strip() or None)
st.sidebar.write("Tesseract:", "✅ "+str(TESS_PATH) if TESS_OK else "❌ "+str(TESS_MSG))
uploads=st.file_uploader("อัปโหลด JPG/PNG", type=["jpg","jpeg","png"], accept_multiple_files=True)
rows=[]
if uploads:
    for up in uploads:
        st.markdown("---"); c1,c2=st.columns([1,1])
        with c1:
            st.write("**ไฟล์:**", up.name)
            image=Image.open(up).convert("RGB"); img_bgr=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
            steps=preprocess(img_bgr)
            if show_steps:
                tabs=st.tabs(list(steps.keys()))
                for tab,k in zip(tabs,steps.keys()):
                    with tab: st.image(steps[k], caption=k, use_container_width=True, clamp=len(steps[k].shape)==2)
        with c2:
            text_easy=text_tess=""
            if engine in ["Hybrid (EasyOCR ➜ fallback Tesseract)","EasyOCR only"] and EASYOCR_AVAILABLE:
                try: text_easy=ocr_easyocr(steps["original"])
                except Exception as e: st.warning(f"EasyOCR error: {e}")
            if engine in ["Hybrid (EasyOCR ➜ fallback Tesseract)","Tesseract only"] and TESS_OK:
                try: text_tess=ocr_tesseract(steps["morph_open"])
                except Exception as e: st.warning(f"Tesseract error: {e}")
            elif engine in ["Hybrid (EasyOCR ➜ fallback Tesseract)","Tesseract only"] and not TESS_OK:
                st.info("Tesseract not found — ใช้ EasyOCR ชั่วคราว หรือระบุ path")
            raw=text_easy if len(text_easy)>=len(text_tess) else text_tess
            if not raw and text_easy: raw=text_easy
            st.text_area("Raw Text", value=raw, height=260)
            fields=extract_fields(raw or "")
            row={"file":up.name,"Vendor / Supplier":fields["Vendor/Supplier"],"Quotation No.":fields["Quotation No."],"Date":fields["Date"],"Description":fields["Description"],"Subtotal":fields["Subtotal"],"VAT":fields["VAT"],"Grand Total":fields["Grand Total"]}
            rows.append(row); st.write("**สรุปฟิลด์ที่สกัดได้**"); st.dataframe(pd.DataFrame([row]))
if rows:
    st.markdown("## ✅ ผลลัพธ์รวม")
    df=pd.DataFrame(rows, columns=["file","Vendor / Supplier","Quotation No.","Date","Description","Subtotal","VAT","Grand Total"])
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลด CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="ocr_quotation_results.csv", mime="text/csv")
