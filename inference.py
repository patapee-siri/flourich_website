# inference.py
import torch
from transformers import pipeline as bert_pipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class FlourichAI:
    def __init__(self, bert_model_dir, groq_api_key, llm_model_name):
        self.bert_model_dir = bert_model_dir
        self.groq_api_key = groq_api_key
        self.llm_model_name = llm_model_name
        self.bert_clf = None
        self.chain = None

    def load_models(self):
        """โหลดโมเดลทั้งหมดเข้าสู่หน่วยความจำและสร้าง Prompt Chain"""
        # 1. Load BERT สำหรับวิเคราะห์อารมณ์ (NLU)
        device = 0 if torch.cuda.is_available() else -1
        self.bert_clf = bert_pipeline(
            "text-classification", 
            model=self.bert_model_dir, 
            tokenizer=self.bert_model_dir, 
            device=device, 
            top_k=None
        )
        
        # 2. Load LLM (Groq via LangChain) สำหรับสร้างคำตอบ (NLG)
        llm = ChatOpenAI(
            openai_api_key=self.groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            model_name=self.llm_model_name,
            temperature=0.7
        )
        
        # 3. Create Prompt Chain - แทนตัวเองเป็น Flourich
        template = """คุณคือนักจิตวิทยาชื่อ Flourich ผู้มีความเห็นอกเห็นใจสูง
        ข้อมูลวิเคราะห์จากระบบ NLU:
        - อารมณ์ที่วิเคราะห์ได้: {label}
        - ค่าความมั่นใจ: {score:.4f}
        
        จงตอบกลับผู้ใช้โดยใช้กลยุทธ์ทางจิตวิทยา: {strategy}
        คำตอบของผู้ใช้: "{user_input}"
        คำตอบของ Flourich:"""
        
        prompt = PromptTemplate.from_template(template)
        self.chain = prompt | llm
        return True

    def process_message(self, text):
        """วิเคราะห์ข้อความด้วย NLU และสร้างคำตอบด้วย NLG"""
        # --- 1. วิเคราะห์ด้วย BERT (เอาตัวที่ score สูงสุด) ---
        results = self.bert_clf(text)
        best_res = max(results[0], key=lambda x: x['score'])
        label_raw, score = best_res['label'], best_res['score']
        
        # --- 2. ทำการแมป Label เพื่อการ debug ที่ชัดเจนตามที่คุณต้องการ ---
        # ถ้า Label มีคำว่า "neg" หรือเลข "0" -> แมปเป็น "neg 0"
        # ถ้า Label มีคำว่า "pos" หรือเลข "1" -> แมปเป็น "pos 1"
        if "neg" in label_raw.lower() or "0" in label_raw:
            label_debug = "neg 0"
            strategy = "Validation"
        elif "pos" in label_raw.lower() or "1" in label_raw:
            label_debug = "pos 1"
            strategy = "Empathy"
        else:
            # Fallback เผื่อ Label มีรูปแบบอื่น
            label_debug = label_raw
            strategy = "Validation" if label_raw == "LABEL_0" else "Empathy"
        
        # --- 3. ส่งหา LLM ผ่าน LangChain ---
        response = self.chain.invoke({
            "label": label_debug,
            "score": score,
            "strategy": strategy,
            "user_input": text
        })
        
        return {
            "label_raw": label_raw,
            "label_debug": label_debug,
            "score": score,
            "strategy": strategy,
            "reply": response.content
        }