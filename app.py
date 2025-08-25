# app.py
# Trang_chủ.py
import streamlit as st

st.set_page_config(
    page_title="Trang chủ - Cố vấn học tập",
    page_icon="🏠",
    layout="wide"
)

st.title("Chào mừng đến với Công cụ hỗ trợ Cố vấn học tập")
st.sidebar.success("Chọn một chức năng ở trên.")

st.markdown(
    """
    Đây là một ứng dụng web được xây dựng để hỗ trợ công tác Cố vấn học tập (CVHT)
    tại Trường Đại học Thủy lợi.

    **👈 Hãy chọn một chức năng từ thanh điều hướng bên trái** để bắt đầu!

    ### Các chức năng hiện có:
    - **Sổ tay CVHT:** Tra cứu nhanh các quy định, quy chế liên quan đến công tác sinh viên.
    - **Biểu mẫu:** (Đang phát triển) Cung cấp các biểu mẫu thường dùng.
    - **Checklist:** (Đang phát triển) Các danh mục công việc cần thực hiện theo từng học kỳ.
    - **Phân tích:** (Đang phát triển) Các công cụ phân tích dữ liệu học tập của sinh viên.
    """
)