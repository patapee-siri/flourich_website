# main.py
import streamlit as st
from inference import FlourichAI
import ui

# ---- CONFIGURATION ----
BERT_MODEL_DIR = r"D:\competition\SSS\Code\LLM_Test\model\final_hf"
GROQ_API_KEY = "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"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Flourich AI", page_icon="🌱", layout="centered")

if "current_page" not in st.session_state:
    st.session_state.current_page = "debug"

@st.cache_resource(show_spinner="กำลังโหลดโมเดล AI (ใช้เวลาสักครู่)...")
def get_ai_engine():
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

# python -m streamlit run D:\competition\SSS\Code\LLM_Test\code\main.py

# 