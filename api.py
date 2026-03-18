# api.py — FastAPI bridge ระหว่าง saferoom.html กับ FlourichAI
# ติดตั้ง: pip install fastapi uvicorn

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import FlourichAI

# โหลด Environment Variables
load_dotenv()

# ── Config ──
BERT_MODEL_DIR = r"D:\competition\SSS\Code\LLM_Test\model\final_hf"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_methods=["*"], 
    allow_headers=["*"]
)

# โหลด Cache
_engine_cache = {}

def get_engine(llm_model: str) -> FlourichAI:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment")
        
    if llm_model not in _engine_cache:
        eng = FlourichAI(BERT_MODEL_DIR, GROQ_API_KEY, llm_model)
        eng.load_models()
        _engine_cache[llm_model] = eng
    return _engine_cache[llm_model]

class ChatRequest(BaseModel):
    message: str
    llm_model: str = "llama-3.3-70b-versatile"

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        engine = get_engine(req.llm_model)
        result = engine.process_message(req.message)
        # ส่งคืน: { label_raw, label_debug, score, strategy, reply }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn api:app --host 0.0.0.0 --port 8000 --reload