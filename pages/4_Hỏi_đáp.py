# pages/4_Hỏi_đáp.py
import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import requests

# --- THÊM MỚI: Import các thư viện cho RAG ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import tempfile
import os
       
# --- 1. KẾT NỐI VỚI SUPABASE & GOOGLE (Cấu hình cho Streamlit Cloud) ---
MANIFEST_URL_DEFAULT = "https://raw.githubusercontent.com/thanvanvan82/cvht-streamlit-app/main/manifest.json"

@st.cache_resource
def init_supabase_connection():
    """Khởi tạo kết nối Supabase với error handling cho Streamlit Cloud"""
    try:
        # Kiểm tra xem có secrets không
        if "SUPABASE_URL" not in st.secrets:
            st.warning("⚠️ SUPABASE_URL chưa được cấu hình trong Streamlit Cloud secrets")
            return None
            
        if "SUPABASE_KEY" not in st.secrets:
            st.warning("⚠️ SUPABASE_KEY chưa được cấu hình trong Streamlit Cloud secrets")
            return None
            
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        
        # Import supabase client (đảm bảo đã cài trong requirements.txt)
        from supabase import create_client, Client
        
        supabase_client = create_client(url, key)
        #st.success("✅ Kết nối Supabase thành công!")
        return supabase_client
        
    except ImportError:
        st.error("❌ Thiếu package 'supabase'. Thêm 'supabase' vào requirements.txt")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi kết nối Supabase: {e}")
        st.info("💡 Hướng dẫn: Vào Settings > Secrets trong Streamlit Cloud để thêm SUPABASE_URL và SUPABASE_KEY")
        return None

# Khởi tạo Supabase client
supabase: Client = init_supabase_connection()

# Cấu hình Google API
def init_google_api():
    """Khởi tạo Google API với error handling cho Streamlit Cloud"""
    try:
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("❌ GOOGLE_API_KEY chưa được cấu hình trong Streamlit Cloud secrets!")
            st.info("💡 Vào Settings > Secrets để thêm GOOGLE_API_KEY")
            st.stop()
        
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        
        # Import và cấu hình Google Generative AI
        import google.generativeai as genai
        genai.configure(api_key=google_api_key)
        
        #st.success("✅ Google API đã được cấu hình!")
        return google_api_key
        
    except ImportError:
        st.error("❌ Thiếu package 'google-generativeai'. Kiểm tra requirements.txt")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi cấu hình Google API: {e}")
        st.stop()

# Khởi tạo Google API
GOOGLE_API_KEY = init_google_api()
          
# --- 2. CÁC HÀM CŨ (Quản lý FAQ và Lịch sử) ---
@st.cache_data(ttl=600)
def get_faqs():
    if not supabase: return []
    try:
        response = supabase.table("cau_hoi_thuong_gap").select("*").not_.is_("tra_loi", "null").order("luot_hoi", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Lỗi khi tải câu hỏi: {e}")
        return []

def submit_new_question_to_faq(user_info, new_question):
    if not supabase: return
    try:
        normalized_question = new_question.strip().lower()
        existing_q = supabase.table("cau_hoi_thuong_gap").select("id, luot_hoi").eq("cau_hoi", normalized_question).execute()

        if existing_q.data:
            q_id = existing_q.data[0]['id']
            new_count = existing_q.data[0]['luot_hoi'] + 1
            supabase.table("cau_hoi_thuong_gap").update({"luot_hoi": new_count}).eq("id", q_id).execute()
        else:
            supabase.table("cau_hoi_thuong_gap").insert({"cau_hoi": normalized_question, "luot_hoi": 1}).execute()
        
        log_submission(user_info, f"[FAQ Submission] {new_question}")
        
    except Exception as e:
        st.error(f"Đã có lỗi xảy ra khi gửi câu hỏi FAQ: {e}")

# HÀM LOG_SUBMISSION ĐÃ CẬP NHẬT
def log_submission(user_info, question=None, answer=None): # Thêm tham số 'answer=None'
    if not supabase: return
    try:
        # Thêm cột 'tra_loi_moi' vào dictionary
        supabase.table("lich_su_hoi").insert({
            "msv": user_info["msv"], "ho_ten": user_info["ho_ten"], "lop": user_info["lop"],
            "sdt": user_info["sdt"], "email": user_info["email"], "cau_hoi_moi": question,
            "created_at": datetime.now().isoformat(),
            "tra_loi_moi": answer  # Dòng mới được thêm vào
        }).execute()
    except Exception as e:
        st.error(f"Lỗi khi ghi nhận thông tin: {e}")

# --- 3. THÊM MỚI: Các hàm cho RAG (Hỏi đáp với tài liệu) ---
@st.cache_data(show_spinner="Đang tải danh sách tài liệu...", ttl=3600)
def fetch_manifest(url):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def download_pdf(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

@st.cache_resource(show_spinner="Đang xử lý tài liệu đã chọn...")
def create_vector_store_from_manifest(_selected_docs_tuple):
    """
    Hàm này nhận một tuple (hashable) của các tài liệu, xử lý và tạo vector store.
    """
    # Chuyển đổi tuple of frozensets trở lại thành list of dicts
    selected_docs = [dict(item) for item in _selected_docs_tuple]
    
    # SỬA LỖI 1: Thay vì tạo list rỗng cho text, ta tạo list rỗng cho documents
    all_pages = [] 

    for doc_info in selected_docs:
        try:
            pdf_bytes = download_pdf(doc_info['url'])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(pdf_bytes)
                pdf_path = tmpfile.name
            
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load_and_split()
                # SỬA LỖI 2: Thêm trực tiếp các đối tượng Document vào danh sách
                all_pages.extend(pages)
            finally:
                os.remove(pdf_path)
        except Exception as e:
            st.warning(f"Lỗi khi xử lý file {doc_info.get('title', 'N/A')}: {e}")
            
    if not all_pages:
        return None

    # SỬA LỖI 3: Sử dụng phương thức đúng là 'split_documents' với đầu vào là list of Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_chunks_as_docs = text_splitter.split_documents(all_pages)
    
    embedding_model_name = "keepitreal/vietnamese-sbert"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # SỬA LỖI 4: Dùng FAISS.from_documents thay vì from_texts
    vector_store = FAISS.from_documents(split_chunks_as_docs, embedding=embeddings)
    return vector_store
    
def get_conversational_chain():
    prompt_template = """
    Bạn là một trợ lý AI chuyên nghiệp, nhiệm vụ của bạn là trả lời câu hỏi chỉ dựa trên nội dung tài liệu được cung cấp.
    Hãy đọc kỹ bối cảnh và trả lời câu hỏi của người dùng.
    Nếu thông tin không có trong bối cảnh, hãy nói rõ "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."

    Bối cảnh:\n{context}\n
    Câu hỏi:\n{question}\n

    Câu trả lời chi tiết (bằng tiếng Việt):
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- 4. GIAO DIỆN STREAMLIT ---
st.title("🙋‍♂️ Hỏi & Đáp dành cho Sinh viên")

if 'user_info_submitted' not in st.session_state:
    st.session_state.user_info_submitted = False
    st.session_state.user_info = {}

if not st.session_state.user_info_submitted:
    st.warning("Vui lòng cung cấp thông tin của bạn để tiếp tục.")
    with st.form("user_info_form"):
        st.subheader("Thông tin sinh viên")
        msv = st.text_input("Mã số sinh viên (*)")
        ho_ten = st.text_input("Họ và tên (*)")
        lop = st.text_input("Lớp (*)")
        sdt = st.text_input("Số điện thoại")
        email = st.text_input("Email")
        
        submitted = st.form_submit_button("Xác nhận và tiếp tục")
        if submitted:
            if not msv or not ho_ten or not lop:
                st.error("Vui lòng điền đầy đủ các trường có dấu (*).")
            else:
                st.session_state.user_info = {"msv": msv, "ho_ten": ho_ten, "lop": lop, "sdt": sdt, "email": email}
                st.session_state.user_info_submitted = True
                log_submission(st.session_state.user_info)
                st.rerun()
else:
    st.success(f"Chào mừng {st.session_state.user_info['ho_ten']}!")
    
    # --- Tab mới để tách biệt hai chức năng ---
    tab_rag, tab_faq = st.tabs(["Hỏi đáp với tài liệu", "Câu hỏi thường gặp"])
    
    with tab_faq:
        st.header("1. Các câu hỏi thường gặp")
        faqs = get_faqs()
        if not faqs:
            st.info("Hiện chưa có câu hỏi nào được trả lời.")
        else:
            for faq in faqs:
                with st.expander(f"❓ {faq['cau_hoi'].capitalize()}"):
                    st.write(faq['tra_loi'])

        st.markdown("---")
        st.header("2. Góp ý câu hỏi cho mục này")
        st.write("Nếu bạn có một câu hỏi chung mà bạn nghĩ nên có trong mục trên, hãy gửi nó ở đây.")
        with st.form("new_faq_question_form"):
            new_question_text = st.text_area("Nhập câu hỏi của bạn:")
            submit_q_button = st.form_submit_button("Gửi câu hỏi")
            if submit_q_button and new_question_text:
                submit_new_question_to_faq(st.session_state.user_info, new_question_text)
                st.success("Gửi góp ý thành công! Cảm ơn bạn.")
            elif submit_q_button:
                st.warning("Vui lòng nhập câu hỏi.")

    with tab_rag:
        st.header("Hỏi đáp dựa trên các văn bản, quy chế")
        st.write("Chọn các tài liệu bạn muốn hỏi, sau đó đặt câu hỏi. AI sẽ trả lời dựa trên nội dung bạn đã chọn.")
        
        manifest = fetch_manifest(MANIFEST_URL_DEFAULT)
        if manifest:
            doc_options = {item.get('title', item['name']): item for item in manifest}
            selected_titles = st.multiselect(
                "Chọn một hoặc nhiều tài liệu để hỏi:",
                options=list(doc_options.keys()),
                default=[list(doc_options.keys())[0]] if doc_options else [] # Mặc định chọn tài liệu đầu tiên
            )
            
            if selected_titles:
                selected_docs_info = [doc_options[title] for title in selected_titles]
                # Chuyển list of dicts thành tuple of frozensets để có thể cache
                hashable_docs_info = tuple(frozenset(item.items()) for item in selected_docs_info)
                vector_store = create_vector_store_from_manifest(hashable_docs_info)

                if 'rag_messages' not in st.session_state:
                    st.session_state.rag_messages = []

                for message in st.session_state.rag_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if user_question := st.chat_input("Đặt câu hỏi về tài liệu đã chọn..."):
                    if vector_store:
                        st.session_state.rag_messages.append({"role": "user", "content": user_question})
                        with st.chat_message("user"):
                            st.markdown(user_question)
                        
                        with st.chat_message("assistant"):
                            with st.spinner("Đang tìm kiếm trong tài liệu..."):
                                conversation_chain = get_conversational_chain()
                                docs = vector_store.similarity_search(user_question)
                                response = conversation_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                                answer = response["output_text"]
                                st.markdown(answer)
                                st.session_state.rag_messages.append({"role": "assistant", "content": answer})
                                log_submission(st.session_state.user_info, f"[RAG] {user_question}", answer)
                    else:
                        st.warning("Không thể xử lý các tài liệu đã chọn. Có thể file bị lỗi hoặc trống.")
            else:
                st.warning("Vui lòng chọn ít nhất một tài liệu để bắt đầu hỏi đáp.")
        else:
            st.error("Không thể tải danh sách tài liệu từ manifest.")
