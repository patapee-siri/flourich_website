import streamlit as st
import os
from dotenv import load_dotenv
from inference import FlourichAI
import ui

# โหลด Environment Variables จาก .env
load_dotenv()

# ---- CONFIGURATION ----
BERT_MODEL_DIR = r"D:\competition\SSS\Code\LLM_Test\model\final_hf"
# ดึง API Key จาก .env (ถ้าไม่มีจะใช้ค่าว่าง)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") 
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Flourich AI", page_icon="🌱", layout="centered")

if "current_page" not in st.session_state:
    st.session_state.current_page = "debug"

@st.cache_resource(show_spinner="กำลังโหลดโมเดล AI (ใช้เวลาสักครู่)...")
def get_ai_engine():
    # ตรวจสอบว่ามี Key ไหมก่อนรัน
    if not GROQ_API_KEY:
        st.error("ไม่พบ GROQ_API_KEY ในไฟล์ .env กรุณาตรวจสอบ")
        st.stop()
        
    engine = FlourichAI(BERT_MODEL_DIR, GROQ_API_KEY, LLM_MODEL_NAME)
    engine.load_models()
    return engine

def go_to_chat():
    st.session_state.current_page = "chat"

def main():
    ai_engine = get_ai_engine()
    
    if st.session_state.current_page == "debug":
        ui.render_debug_page(
            bert_path=BERT_MODEL_DIR, 
            llm_name=LLM_MODEL_NAME, 
            on_next_click=go_to_chat
        )
    elif st.session_state.current_page == "chat":
        ui.render_chat_page(process_func=ai_engine.process_message)

if __name__ == "__main__":
    main()