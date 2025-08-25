# pages/1_Sổ_tay_CVHT.py
# -----------------------------------------------------------
# Tra cứu PDF từ danh sách có sẵn (manifest) hoặc file tải lên.
# - Phiên bản hoàn thiện với logic tìm kiếm thông minh hơn.
# -----------------------------------------------------------

import re
import unicodedata
import requests
import streamlit as st
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Không cần st.set_page_config() ở các trang con

# ============== Cấu hình mặc định ==============
MANIFEST_URL_DEFAULT = "https://raw.githubusercontent.com/thanvanvan82/cvht-streamlit-app/main/manifest.json"

# ===================== Tiện ích (giữ nguyên) =====================
def _safe_url(u: str) -> str:
    return requests.utils.requote_uri(u)
def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
def _normalize_spaces(s: str) -> str:
    s = s.replace("\u00ad", "").replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()
def _normalize_pdf_text(s: str) -> str:
    s = re.sub(r"-\s*\n\s*", "", s)
    return _normalize_spaces(s)

@st.cache_data(show_spinner="Đang tải manifest...", ttl=3600)
def fetch_manifest(url: str) -> List[Dict[str, str]]:
    try:
        r = requests.get(_safe_url(url), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Không thể tải manifest: {e}")
        return []

@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def download_pdf(url: str) -> bytes:
    headers = {"User-Agent": "st-pdf-app/1.0"}
    r = requests.get(_safe_url(url), headers=headers, timeout=60)
    r.raise_for_status()
    return r.content

def pdf_to_pages(pdf_bytes: bytes) -> List[str]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pages.append(_normalize_pdf_text(page.get_text("text")))
    return pages

@dataclass
class Chunk:
    doc_name: str
    url: Union[str, None]
    page: int
    text: str

def build_chunks(pages: List[str], doc_name: str, url: Union[str, None], chunk_size=1100, overlap=150) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, page_text in enumerate(pages):
        if not page_text: continue
        start, L = 0, len(page_text)
        while start < L:
            end = min(L, start + chunk_size)
            txt = page_text[start:end].strip()
            if txt:
                chunks.append(Chunk(doc_name=doc_name, url=url, page=i + 1, text=txt))
            new_start = end - overlap
            start = new_start if new_start > start else end
    return chunks

@st.cache_resource(show_spinner="Đang tạo chỉ mục...")
def make_index(_chunks_tuple: tuple, remove_diacritics: bool):
    chunks = list(_chunks_tuple)
    corpus = [_strip_diacritics(c.text.lower()) if remove_diacritics else c.text.lower() for c in chunks]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", max_df=0.95, min_df=1)
    X = vec.fit_transform(corpus)
    return vec, X

def search(query: str, chunks: List[Chunk], vec, X, remove_diacritics: bool, exact_match: bool, top_k: int) -> List[Tuple[float, float, Chunk]]:
    if not query.strip(): return []
    q = query.strip().lower()
    q_norm = _strip_diacritics(q) if remove_diacritics else q
    q_vec = vec.transform([q_norm])
    sims = cosine_similarity(q_vec, X).ravel()
    candidate_indices = sims.argsort()[::-1][:max(top_k * 5, 100)]
    results = []
    for idx in candidate_indices:
        raw_score = float(sims[idx])
        if raw_score < 0.01: continue
        ch = chunks[idx]
        hay = _strip_diacritics(ch.text).lower() if remove_diacritics else ch.text.lower()
        key_hit = q_norm in hay
        if exact_match and not key_hit:
            continue
        boost = 0.0
        if key_hit:
            boost = 1.0 if exact_match else 0.2
        sort_score = raw_score + boost
        results.append((sort_score, raw_score, ch))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def highlight(text: str, query: str, remove_diacritics: bool, max_len: int = 350):
    base_text = _strip_diacritics(text).lower() if remove_diacritics else text.lower()
    base_query = _strip_diacritics(query).lower() if remove_diacritics else query.lower()
    idx = base_text.find(base_query)
    if idx < 0:
        snippet = text[:max_len] + ("..." if len(text) > max_len else "")
    else:
        start = max(0, idx - max_len // 3)
        end = min(len(text), idx + len(base_query) + (max_len * 2 // 3))
        snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
    safe_query = re.escape(query)
    pattern = re.compile(f"({safe_query})", flags=re.IGNORECASE)
    snippet = pattern.sub(r"<mark>\1</mark>", snippet)
    return snippet

# ===================== GIAO DIỆN NGƯỜI DÙNG (UI) =====================
st.title("📖 Sổ tay CVHT: Tra cứu Quy chế, Quy định")

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# --- Sidebar ---
st.sidebar.title("Thiết lập tra cứu")
manifest = fetch_manifest(MANIFEST_URL_DEFAULT)

if manifest:
    display_titles = [item.get('title', item['name']) for item in manifest]
    title_to_item_map = {item.get('title', item['name']): item for item in manifest}
    st.sidebar.subheader("1. Chọn văn bản có sẵn")
    selected_titles = st.sidebar.multiselect(
        "Chọn từ danh sách",
        options=display_titles,
        default=[display_titles[0]] if display_titles else []
    )
else:
    st.sidebar.warning("Không tải được danh sách văn bản.")

st.sidebar.subheader("2. Hoặc tải lên file PDF từ máy tính")
uploaded_files = st.sidebar.file_uploader(
    "Chọn một hoặc nhiều file PDF",
    type="pdf",
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Tùy chọn")
remove_diac = st.sidebar.toggle("Tìm kiếm **bỏ dấu** (khuyến nghị)", value=True)
top_k = st.sidebar.slider("Số kết quả hiển thị", 5, 50, 20, 1)

with st.sidebar.expander("Tùy chọn nâng cao"):
    chunk_size = st.sidebar.slider("Kích thước đoạn (ký tự)", 600, 2200, 1100, 50)
    st.sidebar.caption("Đoạn văn bản được cắt nhỏ để tìm kiếm. Lớn hơn cho nhiều ngữ cảnh, nhỏ hơn cho kết quả tập trung.")
    overlap = st.sidebar.slider("Độ chồng lấn (ký tự)", 50, 400, 150, 10)
    st.sidebar.caption("Đảm bảo không bỏ sót thông tin ở chỗ cắt giữa hai đoạn.")

# --- Nút Lập chỉ mục ---
if st.button("📚 Lập chỉ mục các file đã chọn", type="primary"):
    selected_from_manifest = [title_to_item_map[title] for title in selected_titles] if manifest else []
    
    if not selected_from_manifest and not uploaded_files:
        st.warning("Vui lòng chọn hoặc tải lên ít nhất một file PDF.")
    else:
        chunks_all: List[Chunk] = []
        with st.status(f"Đang xử lý {len(selected_from_manifest) + len(uploaded_files)} file...", expanded=True) as status:
            for item in selected_from_manifest:
                try:
                    st.write(f"📥 Đang tải: {item.get('title', item['name'])}")
                    pdf_bytes = download_pdf(item["url"])
                    pages = pdf_to_pages(pdf_bytes)
                    chunks = build_chunks(pages, item.get('title', item['name']), item["url"], chunk_size, overlap)
                    chunks_all.extend(chunks)
                except Exception as e:
                    st.error(f"Lỗi với file {item['name']}: {e}")

            for uploaded_file in uploaded_files:
                try:
                    st.write(f"📄 Đang đọc file tải lên: {uploaded_file.name}")
                    pdf_bytes = uploaded_file.getvalue()
                    pages = pdf_to_pages(pdf_bytes)
                    chunks = build_chunks(pages, uploaded_file.name, None, chunk_size, overlap)
                    chunks_all.extend(chunks)
                except Exception as e:
                    st.error(f"Lỗi với file tải lên {uploaded_file.name}: {e}")

            if not chunks_all:
                status.update(label="Lỗi xử lý file!", state="error", expanded=True)
                st.error("Không thể trích xuất văn bản từ bất kỳ file nào.")
            else:
                status.update(label="Đang tạo chỉ mục tìm kiếm...", state="running")
                vec, X = make_index(tuple(chunks_all), remove_diac)
                st.session_state.vectorizer = vec
                st.session_state.tfidf_matrix = X
                st.session_state.chunks = chunks_all
                st.session_state.index_ready = True
                status.update(label=f"Hoàn tất! Đã lập chỉ mục {len(chunks_all)} đoạn.", state="complete")

# --- Giao diện tìm kiếm (hiển thị sau khi lập chỉ mục) ---
if st.session_state.index_ready:
    st.markdown("---")
    
    with st.form(key="search_form"):
        col1, col2 = st.columns([5, 1])
        with col1:
            q = st.text_input(
                "Nhập từ khóa cần tra",
                key="search_query",
                placeholder="Ví dụ: sinh viên, học phí, rèn luyện..."
            )
            exact_match = st.toggle("Tìm chính xác cụm từ", help="Chỉ hiển thị các đoạn chứa chính xác cụm từ đã gõ.")
        with col2:
            st.write("&nbsp;")
            submitted = st.form_submit_button("🔍 Tìm kiếm")

    if submitted and q.strip():
        results = search(
            q, 
            st.session_state.chunks, 
            st.session_state.vectorizer, 
            st.session_state.tfidf_matrix, 
            remove_diac, 
            exact_match, 
            top_k
        )
        if not results:
            st.warning("Không tìm thấy kết quả phù hợp.")
        else:
            st.subheader(f"Tìm thấy {len(results)} kết quả liên quan:")
            for sort_score, raw_score, ch in results:
                snippet = highlight(ch.text, q, remove_diac)
                with st.container(border=True):
                    source_info = f"**Tài liệu:** `{ch.doc_name}` · **Trang:** {ch.page} · **Mức độ phù hợp:** {raw_score * 100:.1f}%"
                    
                    if ch.url:
                        source_info += f"\n\n**Nguồn:** [Mở PDF tại trang {ch.page}]({ch.url}#page={ch.page})"
                    else:
                        source_info += "\n\n**Nguồn:** Tệp đã tải lên"
                    
                    st.markdown(source_info, unsafe_allow_html=True)
                    st.markdown(f"> {snippet}", unsafe_allow_html=True)
else:
    st.info("👋 Chào bạn! Hãy chọn/tải lên văn bản ở thanh bên trái và bấm **Lập chỉ mục** để bắt đầu.")