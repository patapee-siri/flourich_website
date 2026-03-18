# api.py — FastAPI bridge ระหว่าง saferoom.html กับ FlourichAI
# ติดตั้ง: pip install fastapi uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import FlourichAI

# ── Config เหมือน main.py ──
BERT_MODEL_DIR = r"D:\competition\SSS\Code\LLM_Test\model\final_hf"
GROQ_API_KEY   = "" # ใส่ API Key กันเอาเองนะที่ https://www.groq.com/dashboard/api-keys

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# โหลด BERT ครั้งเดียว
_engine_cache = {}

def get_engine(llm_model: str) -> FlourichAI:
    if llm_model not in _engine_cache:
        eng = FlourichAI(BERT_MODEL_DIR, GROQ_API_KEY, llm_model)
        eng.load_models()
        _engine_cache[llm_model] = eng
    return _engine_cache[llm_model]

class ChatRequest(BaseModel):
    message:   str
    llm_model: str = "llama-3.3-70b-versatile"   # default = Flourich 2.0

@app.post("/chat")
def chat(req: ChatRequest):
    engine = get_engine(req.llm_model)
    return engine.process_message(req.message)
    # ส่งคืน: { label_raw, label_debug, score, strategy, reply }

    # uvicorn api:app --host 0.0.0.0 --port 8000 --reload