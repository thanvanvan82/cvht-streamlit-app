# pages/1_Sổ_tay_CVHT.py
import re
import unicodedata
import requests
import streamlit as st
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

# Import các thư viện cần thiết cho việc xử lý machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============== Cấu hình mặc định ==============
MANIFEST_URL_DEFAULT = "https://raw.githubusercontent.com/thanvanvan82/cvht-streamlit-app/main/manifest.json"

# ===================== Các hàm tiện ích (giữ nguyên không đổi) =====================
@st.cache_data(show_spinner="Đang tải manifest...", ttl=3600)
def fetch_manifest(url: str) -> List[Dict[str, str]]:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Không thể tải manifest: {e}")
        return []

@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def download_pdf(url: str) -> bytes:
    headers = {"User-Agent": "st-pdf-app/1.0"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.content

def pdf_to_pages(pdf_bytes: bytes) -> List[str]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            normalized_text = re.sub(r"-\s*\n\s*", "", text)
            normalized_text = re.sub(r"[ \t]+", " ", normalized_text.replace("\u00ad", "").replace("\u00A0", " "))
            normalized_text = re.sub(r"\s*\n\s*", "\n", normalized_text).strip()
            pages.append(normalized_text)
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
def make_index(_chunks_tuple: tuple):
    chunks = list(_chunks_tuple)
    corpus = [c.text.lower() for c in chunks]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", max_df=0.95, min_df=1)
    X = vec.fit_transform(corpus)
    return vec, X

def search(query: str, chunks: List[Chunk], vec, X, remove_diacritics: bool, exact_match: bool, top_k: int) -> List[Tuple[float, float, Chunk]]:
    if not query.strip(): return []
    q = query.strip().lower()
    if remove_diacritics:
        q = "".join(c for c in unicodedata.normalize('NFD', q) if unicodedata.category(c) != 'Mn')
    
    q_vec = vec.transform([q])
    sims = cosine_similarity(q_vec, X).ravel()
    
    candidate_indices = sims.argsort()[::-1][:max(top_k * 5, 100)]
    results = []
    for idx in candidate_indices:
        raw_score = float(sims[idx])
        if raw_score < 0.01: continue
            
        ch = chunks[idx]
        hay = ch.text.lower()
        if remove_diacritics:
            hay = "".join(c for c in unicodedata.normalize('NFD', hay) if unicodedata.category(c) != 'Mn')
        
        key_hit = q in hay
        
        if exact_match and not key_hit:
            continue
            
        boost = 1.0 if key_hit and exact_match else (0.2 if key_hit else 0.0)
        sort_score = raw_score + boost
        results.append((sort_score, raw_score, ch))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def highlight(text: str, query: str, remove_diacritics: bool, max_len: int = 350):
    if not query: return text[:max_len] + ("..." if len(text) > max_len else "")

    base_text = text.lower()
    base_query = query.lower()
    if remove_diacritics:
        base_text = "".join(c for c in unicodedata.normalize('NFD', base_text) if unicodedata.category(c) != 'Mn')
        base_query = "".join(c for c in unicodedata.normalize('NFD', base_query) if unicodedata.category(c) != 'Mn')

    idx = base_text.find(base_query)
    
    if idx < 0:
        snippet = text[:max_len] + ("..." if len(text) > max_len else "")
    else:
        start = max(0, idx - max_len // 3)
        end = min(len(text), idx + len(base_query) + (max_len * 2 // 3))
        snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
    
    safe_query = re.escape(query)
    pattern = re.compile(f"({safe_query})", flags=re.IGNORECASE)
    highlighted_snippet = pattern.sub(r"<mark>\1</mark>", snippet)
    
    return highlighted_snippet

# ===================== GIAO DIỆN NGƯỜI DÙNG (UI) =====================
st.title("📖 Sổ tay CVHT")

tab_tra_cuu, tab_bieu_mau = st.tabs(["Tra cứu Quy chế, Quy định", "Quy trình, Biểu mẫu"])

with tab_tra_cuu:
    # --- SỬA LỖI 1: Khởi tạo tất cả các biến trạng thái cần thiết ---
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'last_keyword' not in st.session_state:
        st.session_state.last_keyword = ""
    if 'last_remove_diac' not in st.session_state:
        st.session_state.last_remove_diac = True

    # --- Sidebar ---
    with st.sidebar:
        st.title("Tùy chọn")
        st.subheader("Hiển thị & Kỹ thuật")
        top_k = st.slider("Số kết quả hiển thị", 5, 50, 20, 1)
        
        with st.expander("Tùy chọn kỹ thuật"):
            chunk_size = st.slider("Kích thước đoạn (ký tự)", 600, 2200, 1100, 50)
            overlap = st.slider("Độ chồng lấn (ký tự)", 50, 400, 150, 10)

    # --- Giao diện dựa trên trạng thái ---
    if not st.session_state.search_results:
        manifest = fetch_manifest(MANIFEST_URL_DEFAULT)
        if manifest:
            doc_titles = [item.get('title', item['name']) for item in manifest]
            title_to_item_map = {item.get('title', item['name']): item for item in manifest}
            search_options = ["Tất cả văn bản"] + doc_titles
        else:
            title_to_item_map = {}
            search_options = ["Tất cả văn bản"]

        with st.container(border=True):
            with st.form(key="search_form"):
                st.write("**TRA CỨU VĂN BẢN**")
                
                col1, col2 = st.columns(2)
                with col1:
                    keyword = st.text_input("Từ khóa", placeholder="Nhập từ khóa bạn muốn tìm...")
                with col2:
                    selected_doc_title = st.selectbox("Văn bản", options=search_options, index=1)

                col3, col4, col5 = st.columns([2, 1, 1])
                with col3:
                    exact_match = st.checkbox("Tìm chính xác cụm từ", value=False)
                    remove_diac = st.checkbox("Tìm không dấu (khuyến nghị)", value=True)
                with col4:
                    search_submitted = st.form_submit_button("🔍 Tìm kiếm")
                with col5:
                    reset_submitted = st.form_submit_button("🔄 Nhập lại")

                with st.expander("Văn bản bổ sung (Upload file PDF để tra cứu tạm thời)"):
                    uploaded_files = st.file_uploader(
                        "Kéo thả hoặc chọn file", type="pdf",
                        accept_multiple_files=True, label_visibility="collapsed"
                    )
        
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; color: #004b8d;'>HƯỚNG DẪN TRA CỨU</h4>", unsafe_allow_html=True)
            st.markdown("""
            1.  **Từ khóa:** Gõ nội dung cần tìm.
            2.  **Văn bản:** Chọn phạm vi tìm kiếm (mặc định là văn bản đầu tiên).
            3.  **Văn bản bổ sung:** Có thể tải lên file PDF của riêng bạn để tìm kiếm cùng lúc.
            4.  **Tùy chọn:** Tinh chỉnh cách tìm kiếm để có kết quả chính xác hơn.
            5.  Nhấn nút **Tìm kiếm** để xem kết quả.
            """)
        
        if reset_submitted:
            st.session_state.search_results = []
            st.session_state.last_keyword = ""
            
        if search_submitted and keyword:
            # SỬA LỖI 2: Lưu lại TẤT CẢ thông số tìm kiếm vào session_state
            st.session_state.last_keyword = keyword
            st.session_state.last_remove_diac = remove_diac
            
            docs_to_process = []
            if selected_doc_title == "Tất cả văn bản":
                if manifest: docs_to_process.extend(manifest)
            elif manifest:
                docs_to_process.append(title_to_item_map[selected_doc_title])
            
            chunks_all = []
            with st.spinner(f"Đang xử lý {len(docs_to_process) + len(uploaded_files)} file..."):
                for item in docs_to_process:
                    try:
                        pdf_bytes = download_pdf(item["url"])
                        pages = pdf_to_pages(pdf_bytes)
                        chunks = build_chunks(pages, item.get('title', item['name']), item["url"], chunk_size, overlap)
                        chunks_all.extend(chunks)
                    except Exception as e:
                        st.error(f"Lỗi với file {item['name']}: {e}")

                for uploaded_file in uploaded_files:
                    try:
                        pdf_bytes = uploaded_file.getvalue()
                        pages = pdf_to_pages(pdf_bytes)
                        chunks = build_chunks(pages, uploaded_file.name, None, chunk_size, overlap)
                        chunks_all.extend(chunks)
                    except Exception as e:
                        st.error(f"Lỗi với file tải lên {uploaded_file.name}: {e}")

                if chunks_all:
                    vec, X = make_index(tuple(chunks_all))
                    results = search(keyword, chunks_all, vec, X, remove_diac, exact_match, top_k)
                    st.session_state.search_results = results
                    st.rerun()
                else:
                    st.warning("Không có văn bản nào được xử lý.")
                    st.session_state.search_results = []
            
            if not st.session_state.search_results:
                 st.warning("Không tìm thấy kết quả phù hợp.")

    else: # Khi có kết quả để hiển thị
        if st.button("↩️ Trở về giao diện tra cứu"):
            st.session_state.search_results = []
            st.session_state.last_keyword = ""
            st.rerun()

        results = st.session_state.search_results
        st.markdown("---")
        st.subheader(f"Tìm thấy {len(results)} kết quả cho từ khóa \"{st.session_state.last_keyword}\":")
        for sort_score, raw_score, ch in results:
            # SỬA LỖI 3: Luôn đọc các tham số từ session_state để highlight
            snippet = highlight(ch.text, st.session_state.last_keyword, st.session_state.last_remove_diac)
            with st.container(border=True):
                source_info = f"**Tài liệu:** `{ch.doc_name}` · **Trang:** {ch.page} · **Mức độ phù hợp:** {raw_score * 100:.1f}%"
                if ch.url:
                    source_info += f"\n\n**Nguồn:** [Mở PDF tại trang {ch.page}]({ch.url}#page={ch.page})"
                else:
                    source_info += "\n\n**Nguồn:** Tệp đã tải lên"
                st.markdown(source_info, unsafe_allow_html=True)
                st.markdown(f"> {snippet}", unsafe_allow_html=True)

# --- Nội dung cho Tab 2 ---
with tab_bieu_mau:
    st.header("Quy trình xác nhận sinh viên online")
    st.markdown("""
    - **Hướng dẫn xác nhận sinh viên trường đại học thủy lợi theo hình thức online:** [Link](https://tlu.edu.vn/quy-trinh-xac-nhan-sinh-vien-online-cua-truong-dai-hoc-thuy-loi-42898/)
    """)    
    st.header("Danh sách các biểu mẫu")
    st.markdown("""
    - **Các mẫu đơn cho Sinh viên:** [Link](https://tlu.edu.vn/cac-mau-don-cho-sinh-vien-36463/)
    """)