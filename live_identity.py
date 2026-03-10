"""
Speaker Identity System — FastAPI WebSocket Server
Stable mode: processes fixed 1-second audio windows (same logic as old script)
"""

import asyncio
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.spatial.distance import cosine
from starlette.websockets import WebSocketState

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    logging.warning("sherpa_onnx not installed — running in MOCK mode")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SETTINGS (same as your old script)
# ─────────────────────────────────────────────
SAMPLE_RATE        = 16000
BUFFER_SECONDS     = 1.0
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * BUFFER_SECONDS)

MATCH_THRESHOLD   = 0.60
NEW_SPEAKER_LIMIT = 0.80

DB_PATH    = Path("data/speaker_db.pkl")
MODEL_PATH = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

SPEAKER_LABELS = {
    0: "HOST (You)",
    1: "GUEST 1",
    2: "GUEST 2",
    3: "GUEST 3"
}

# ─────────────────────────────────────────────
# IDENTITY MANAGER (unchanged)
# ─────────────────────────────────────────────
class IdentityManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.known_embeddings = []
        self.speaker_ids = []
        self.load_db()

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump({"embeddings": self.known_embeddings, "ids": self.speaker_ids}, f)

    def load_db(self):
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
                self.known_embeddings = data["embeddings"]
                self.speaker_ids = data["ids"]
            log.info("Loaded %d embeddings", len(self.known_embeddings))

    def delete_db(self):
        self.known_embeddings = []
        self.speaker_ids = []
        if self.db_path.exists():
            self.db_path.unlink()
        log.info("Speaker database cleared")

    def get_speaker(self, new_embedding: np.ndarray):
        if len(self.known_embeddings) == 0:
            self.known_embeddings.append(new_embedding)
            self.speaker_ids.append(0)
            return 0, 0.0

        best_score = 1.0
        best_id = -1

        for i, stored in enumerate(self.known_embeddings):
            score = cosine(new_embedding, stored)
            if score < best_score:
                best_score = score
                best_id = self.speaker_ids[i]

        if best_score < MATCH_THRESHOLD:
            return best_id, best_score
        elif best_score > NEW_SPEAKER_LIMIT:
            new_id = len(set(self.speaker_ids))
            self.known_embeddings.append(new_embedding)
            self.speaker_ids.append(new_id)
            self.save_db()
            return new_id, 0.0
        else:
            return best_id, best_score


# ─────────────────────────────────────────────
# CONNECTION STATE — fixed window buffer
# ─────────────────────────────────────────────
class ConnectionState:
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)


# ─────────────────────────────────────────────
# EXTRACTOR
# ─────────────────────────────────────────────
def build_extractor():
    if not SHERPA_AVAILABLE:
        return None
    if not Path(MODEL_PATH).exists():
        log.warning("Model not found — MOCK mode")
        return None
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODEL_PATH)
    log.info("Model loaded")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


extractor = build_extractor()
manager   = IdentityManager()

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="Speaker Identity System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():

    index = static_dir / "index.html"
    # FIX 1: encoding="utf-8" — Windows defaults to cp1252 which crashes on emoji/special chars
    return HTMLResponse(
        index.read_text(encoding="utf-8") if index.exists()
        else "<h2>Place index.html inside /static</h2>"
    )


@app.delete("/delete-db")
async def delete_db():
    manager.delete_db()
    return JSONResponse({"status": "ok"})


# ─────────────────────────────────────────────
# AUDIO PROCESSING — OLD LOGIC (unchanged)
# ─────────────────────────────────────────────
async def process_audio(raw_bytes: bytes, state: ConnectionState):
    return await asyncio.get_event_loop().run_in_executor(None, _sync_process, raw_bytes, state)


def _sync_process(raw_bytes: bytes, state: ConnectionState):
    samples = np.frombuffer(raw_bytes, dtype=np.float32)

    # accumulate into buffer
    state.buffer = np.concatenate([state.buffer, samples])

    # wait for full 1-second window
    if len(state.buffer) < SAMPLES_PER_WINDOW:
        return None

    # take exactly 1-second window, keep remainder
    window = state.buffer[:SAMPLES_PER_WINDOW]
    state.buffer = state.buffer[SAMPLES_PER_WINDOW:]

    # same loudness check as old script
    if np.linalg.norm(window) < 7.0:
        return None

    # ── MOCK mode ──
    if extractor is None:
        emb = np.random.rand(192).astype(np.float32)
        speaker_id, score = manager.get_speaker(emb)
        name = SPEAKER_LABELS.get(speaker_id, f"Speaker {speaker_id}")
        conf = int((1 - score) * 100) if score > 0 else 100
        return {
            "type": "result",
            "speaker": name,
            "confidence": conf,
            "speaker_count": len(set(manager.speaker_ids)),
            "mock": True,
        }

    # ── REAL mode ──
    stream_obj = extractor.create_stream()
    stream_obj.accept_waveform(SAMPLE_RATE, window)
    embedding = extractor.compute(stream_obj)

    speaker_id, score = manager.get_speaker(np.array(embedding))
    name = SPEAKER_LABELS.get(speaker_id, f"Speaker {speaker_id}")
    conf = int((1 - score) * 100) if score > 0 else 100

    if conf <= 30:
        return None

    log.info("Speaker: %s | Confidence: %d%%", name, conf)

    return {
        "type": "result",
        "speaker": name,
        "confidence": conf,
        "speaker_count": len(set(manager.speaker_ids)),
        "mock": False,
    }


# ─────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = ConnectionState()
    log.info("Client connected: %s", ws.client.host)

    try:
        while True:
            # FIX 2: check state before receiving + catch RuntimeError on disconnect
            if ws.client_state != WebSocketState.CONNECTED:
                break

            try:
                data = await ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            if data.get("type") == "websocket.disconnect":
                break

            if data.get("text"):
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    continue
                if msg.get("cmd") == "reset":
                    state.reset()
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({"type": "reset_ok"})
                elif msg.get("cmd") == "delete_db":
                    manager.delete_db()
                    state.reset()
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({"type": "db_deleted"})

            elif data.get("bytes"):
                result = await process_audio(data["bytes"], state)
                if result and ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(result)

    except Exception as exc:
        log.warning("WS error: %s", exc)
    finally:
        log.info("Client disconnected: %s", ws.client.host)