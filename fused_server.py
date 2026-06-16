#!/usr/bin/env python3
"""
Fallback OpenAI-Compatible API Server for 5G Core/RAN Intelligent Diagnostic Engine.
Loads the fused Llama-3.3-70B model using standard HF Transformers.
Bypasses native vLLM compilation/Triton version incompatibilities.
"""

import os
import sys
import time
import torch
import json
import threading
import warnings
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from rag_pipeline.spec_retriever import SpecRetriever, build_augmented_system_prompt

warnings.filterwarnings("ignore")

app = FastAPI(title="5G Core/RAN Diagnostic Engine - Fused Model Server")

# Add CORS middleware to allow Streamlit connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/workspace/telco_expert_llama3_3_70b_merged"

# Global model pointers
model = None
tokenizer = None
retriever = None

@app.on_event("startup")
def load_model():
    global model, tokenizer, retriever
    print("=" * 70)
    print("🚀 Mounting Fused Llama-3.3-70B Model into HBM3 Lanes (Standard HF Loader)...")
    print("=" * 70)
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        print(f"✓ Fused model and tokenizer loaded successfully in {time.time() - start_time:.1f}s!")
        
        # Load RAG retriever
        print("📥 Initializing 3GPP Specification RAG Index (Auto-detection mode)...")
        retriever = SpecRetriever()
        print("✓ SpecRetriever initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to load model or retriever from {MODEL_PATH}: {e}")
        sys.exit(1)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.4)
    max_tokens = body.get("max_tokens", 700)
    stream = body.get("stream", False)
    repetition_penalty = body.get("frequency_penalty", 1.2)

    # ── RAG Context Augmentation ──
    system_msg_idx = -1
    user_query = ""
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_msg_idx = i
        elif msg.get("role") == "user":
            user_query = msg.get("content", "")

    if system_msg_idx != -1 and user_query and retriever:
        base_system_prompt = messages[system_msg_idx]["content"]
        
        # Prevent double-augmentation if client/frontend already performed retrieval
        if "--- RETRIEVED 3GPP SPECIFICATION CONTEXT ---" in base_system_prompt:
            print("🔍 [RAG] Prompt already contains 3GPP spec context. Skipping server-side RAG.")
        else:
            # Perform retrieval
            start_rag = time.time()
            results = retriever.retrieve(user_query, top_k=3)
            print(f"🔍 [RAG] Query: '{user_query[:60]}...' -> Found {len(results)} spec citations (took {time.time() - start_rag:.3f}s).")
            for idx, r in enumerate(results):
                print(f"    - [{idx+1}] {r['spec_id']} Section {r['section']} (Score: {r['relevance_score']:.4f})")
            
            augmented_system_prompt = build_augmented_system_prompt(
                base_system_prompt,
                user_query,
                retriever,
                top_k=3
            )
            messages[system_msg_idx]["content"] = augmented_system_prompt

    # Reconstruct prompt using Chat Template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # ── Non-Streaming Mode ──
    if not stream:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 0.05 else False
            )
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }]
        }

    # ── Streaming Mode ──
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=True if temperature > 0.05 else False
    )

    # Launch generation in background thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate_events():
        # Yield initial role delta
        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}}]})}\n\n"
        for text in streamer:
            chunk = {
                "choices": [{
                    "delta": {
                        "content": text
                    }
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
