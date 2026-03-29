import os
import re
import threading
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = APP_DIR / "model_adapter"
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "120"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))

app = Flask(__name__)

_model = None
_tokenizer = None
_model_error = None
_model_lock = threading.Lock()


def _set_torch_runtime() -> None:
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(TORCH_THREADS)
    except Exception:
        pass


def _load_model_once() -> None:
    global _model, _tokenizer, _model_error
    if _model is not None or _model_error is not None:
        print("Model already loaded")
        return

    with _model_lock:
        if _model is not None or _model_error is not None:
            print("Model already loaded")
            return
        try:
            print("Loading model...")

            _set_torch_runtime()
            _tokenizer = AutoTokenizer.from_pretrained(
                str(ADAPTER_DIR), use_fast=True, legacy=False
            )
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
            peft_model = peft_model.merge_and_unload()
            peft_model.eval()

            _model = peft_model

            print("Model loaded")
        except Exception as exc:
            print(f"Failed to load model: {exc}")

            _model_error = f"{type(exc).__name__}: {exc}"


SYSTEM_NOTE = (
    "You are MedBot, a simple educational medical assistant. "
    "Give clear, short, easy-to-understand answers. "
    "Do not claim to diagnose or replace a doctor. "
    "If symptoms sound severe or urgent, tell the user to seek medical care quickly."
)


def build_prompt(user_question: str) -> str:
    user_question = user_question.strip()
    return (
        "### Instruction:\n"
        f"{SYSTEM_NOTE}\n"
        "Answer the following medical question clearly and briefly.\n\n"
        f"### Input:\n{user_question}\n\n"
        "### Response:\n"
    )


def clean_response(full_text: str) -> str:
    text = full_text
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1]
    text = text.strip()
    text = re.split(r"\n###\s", text)[0].strip()
    return text or "Sorry, I could not generate a useful response."


def generate_medbot_response(question: str) -> str:
    if _model_error is not None:
        return (
            "The model could not be loaded on this server. "
            f"Startup error: {_model_error}"
        )
    if _model is None or _tokenizer is None:
        return "The model is still loading. Please try again in a moment."

    prompt = build_prompt(question)
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with _model_lock:
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )

    decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_response(decoded)


@app.route("/", methods=["GET"])
def index():
    print("Request for index page")

    return render_template("index.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    print("Request for health check")

    return jsonify(
        {
            "status": "ok" if _model_error is None else "error",
            "model_loaded": _model is not None,
            "model_error": _model_error,
            "base_model": BASE_MODEL_ID,
        }
    ), (200 if _model_error is None else 500)


@app.route("/chat", methods=["POST"])
def chat():
    print("Request for chat")
    start_time = time.perf_counter()

    _load_model_once()

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    print("Request:", message)

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    reply = generate_medbot_response(message)
    elapsed_seconds = time.perf_counter() - start_time
    elapsed_display = f"{elapsed_seconds:.2f}s"

    print(f"Response ({elapsed_display}):", reply)

    return jsonify(
        {
            "reply": reply,
            "time_spent_seconds_display": elapsed_display,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "9978"))
    app.run(host="0.0.0.0", port=port, debug=False)
