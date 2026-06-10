from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json

from predict import predict_disease
from chatgpt_service import (
    ask_chatgpt,
    ask_chatgpt_stream,
    build_disease_analysis_prompt,
    validate_plant_image,
)

app = FastAPI(title="Plant Doctor AI", version="2.0")


# ============================================================
# SESSION STORE
# ============================================================

sessions: dict[str, dict] = {}

MAX_HISTORY = 20


def get_session(conversation_id: str) -> dict:
    if conversation_id not in sessions:
        sessions[conversation_id] = {
            "disease": None,
            "history": [],
        }
    return sessions[conversation_id]


def trim_history(history: list) -> list:
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    conversation_id: str
    question: str


class NewConversationResponse(BaseModel):
    conversation_id: str


# ============================================================
# API: Tạo conversation mới
# ============================================================

@app.post("/conversation/new", response_model=NewConversationResponse)
async def new_conversation():
    conv_id = str(uuid.uuid4())
    sessions[conv_id] = {"disease": None, "history": []}
    return {"conversation_id": conv_id}


# ============================================================
# API 1: NHẬN DIỆN BỆNH
# ============================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = None,
):
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    session = get_session(conversation_id)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File ảnh rỗng")

    # ── BƯỚC 1: Validate ảnh bằng GPT Vision ──────────────────
    try:
        validation = validate_plant_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi kiểm tra ảnh: {str(e)}")

    if not validation["is_plant"]:
        # Ảnh không phải cây — trả về thông báo, không chạy model
        return {
            "conversation_id": conversation_id,
            "disease": None,
            "solution": validation["message"],
        }

    # ── BƯỚC 2: Nhận diện bệnh (chỉ chạy nếu ảnh hợp lệ) ─────
    try:
        disease = predict_disease(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi nhận diện: {str(e)}")

    session["disease"] = disease

    analysis_prompt = build_disease_analysis_prompt(disease)

    session["history"].append({
        "role": "user",
        "content": f"[Người dùng đã chụp ảnh lá cây] AI nhận diện bệnh: {disease}\n\nHãy phân tích chi tiết bệnh này."
    })

    try:
        solution = ask_chatgpt(session["history"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi ChatGPT: {str(e)}")

    session["history"].append({
        "role": "assistant",
        "content": solution
    })
    session["history"] = trim_history(session["history"])

    return {
        "conversation_id": conversation_id,
        "disease": disease,
        "solution": solution,
    }


# ============================================================
# API 2: CHAT thường (fallback, không streaming)
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    session = get_session(req.conversation_id)

    if session["disease"] and not _is_about_disease(req.question, session["disease"]):
        user_content = f"[Ngữ cảnh: cây đang mắc bệnh {session['disease']}]\n{req.question}"
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})

    try:
        answer = ask_chatgpt(session["history"])
    except Exception as e:
        session["history"].pop()
        raise HTTPException(status_code=500, detail=f"Lỗi ChatGPT: {str(e)}")

    session["history"].append({"role": "assistant", "content": answer})
    session["history"] = trim_history(session["history"])

    return {
        "conversation_id": req.conversation_id,
        "answer": answer,
    }


# ============================================================
# API 3: CHAT STREAMING (SSE)
# ============================================================

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    session = get_session(req.conversation_id)

    if session["disease"] and not _is_about_disease(req.question, session["disease"]):
        user_content = f"[Ngữ cảnh: cây đang mắc bệnh {session['disease']}]\n{req.question}"
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})

    history_snapshot = list(session["history"])

    async def event_generator():
        full_response = []
        try:
            for chunk in ask_chatgpt_stream(history_snapshot):
                full_response.append(chunk)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

            complete = "".join(full_response)
            session["history"].append({"role": "assistant", "content": complete})
            session["history"] = trim_history(session["history"])

        except Exception as e:
            yield f"data: {json.dumps('⚠️ Lỗi: ' + str(e), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            if session["history"]:
                session["history"].pop()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# API 4: XOÁ SESSION
# ============================================================

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id in sessions:
        del sessions[conversation_id]
    return {"status": "deleted"}


# ============================================================
# HELPER
# ============================================================

def _is_about_disease(question: str, disease: str) -> bool:
    q_lower = question.lower()
    d_lower = disease.lower()
    keywords = ["bệnh", "thuốc", "xử lý", "điều trị", "triệu chứng", "nguyên nhân"]
    return any(k in q_lower for k in keywords) or any(
        word in q_lower for word in d_lower.split()
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(sessions)}
