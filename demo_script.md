# 🎬 Demo Recording Script — 5G Core/RAN Intelligent Diagnostic Engine

> **Total Duration**: 5-7 minutes  
> **Format**: Screen recording with voice narration  
> **Tools needed**: OBS/screen recorder, microphone

---

## 🎞️ SCENE 1: Title & Problem Statement (0:00 - 0:45)

### SLIDE: Title Card
> **Show**: PPT Slide — Project title, team name, AMD branding

### 🎙️ VOICEOVER:
> "When a 5G network drops 50,000 calls during a concert at a stadium, a Network Operations Center engineer has roughly 15 minutes to diagnose the root cause before it becomes front-page news.
>
> Today, these engineers manually grep through gigabytes of protocol logs, cross-reference 3GPP specifications, and rely on years of tribal knowledge. This process can take hours.
>
> Our solution — the 5G Core/RAN Intelligent Diagnostic Engine — gives them an instant, 3GPP-specification-compliant root cause analysis in seconds, powered by a 70-billion parameter AI model fine-tuned specifically for telecom protocol engineering, running on AMD's Instinct MI300X GPU."

### SLIDE: Problem-Solution Summary
> **Show**: Before/After comparison slide

---

## 🎞️ SCENE 2: Architecture Overview (0:45 - 1:45)

### SLIDE: Technical Architecture Diagram
> **Show**: Architecture flow diagram (from training_analysis charts)

### 🎙️ VOICEOVER:
> "Let me walk you through our technical architecture.
>
> We start with Meta's Llama 3.3, a 70-billion parameter instruction-tuned model. At full precision, this model would require over 140 gigabytes of VRAM — but through 4-bit quantization, we compress it to approximately 38 gigabytes, fitting comfortably within the MI300X's 192 gigabytes of HBM3 memory.
>
> We then apply QLoRA — Quantized Low-Rank Adaptation — adding just 207 million trainable parameters, only 0.29% of the total model. This is efficient, precise, and avoids catastrophic forgetting of the base model's capabilities.
>
> What makes our approach unique is the **curriculum learning strategy**: we train in two distinct phases.
>
> **Phase 1** trains on 3GPP technical specification data — TSG documents and TeleQnA datasets — teaching the model protocol-level reasoning.
>
> **Phase 2** trains on real-world telecom customer service transcripts from African mobile operators, teaching the model to understand how real network problems are described by customers and field engineers.
>
> Finally, we use a novel **matrix interpolation merge** — a raw element-wise mathematical blend across all 1,120 LoRA parameter matrices — to combine both knowledge domains into a single unified expert brain."

---

## 🎞️ SCENE 3: Training Results (1:45 - 2:30)

### SLIDE: Training Loss Curves
> **Show**: training_curves.png chart

### 🎙️ VOICEOVER:
> "Here are our actual training results from the MI300X.
>
> In Phase 1, the 3GPP domain training, you can see the loss dropping from 3.7 to approximately 0.93 over 300 training steps. The noisy convergence pattern is characteristic of diverse technical specification data — the model is learning a wide range of protocol concepts.
>
> In Phase 2, the conversational training converges much faster — from 4.3 down to 0.14 in just 150 steps. This rapid convergence tells us the base model already had strong conversational abilities; we're fine-tuning it specifically for telecom diagnostic dialogue patterns.
>
> The key insight: curriculum learning lets us specialize the model in stages, preventing the simpler conversational patterns from overwriting the complex 3GPP protocol knowledge."

### SLIDE: Model Statistics
> **Show**: model_stats.png chart

---

## 🎞️ SCENE 4: Live Demo — Diagnostic Engine (2:30 - 4:30)

### 🖥️ ACTION: Switch to Streamlit App (app_v2.py running)

### 🎙️ VOICEOVER:
> "Now let me show you the engine in action. This is our enterprise diagnostic workbench, running live on the AMD MI300X."

### Demo Query 1: Customer Complaint
> **Type/click**: "My phone drops data service completely to zero whenever I walk near the central metro station entrance. It stays offline for a minute then jumps back on LTE."

### 🎙️ VOICEOVER (while streaming):
> "Watch the streaming response in real-time. The model is generating token by token on the MI300X GPU.
>
> Notice how it immediately identifies this as a handover failure scenario, references the correct 3GPP specification — TS 38.331 — and provides a structured protocol root cause trace.
>
> It identifies the RRCReconfiguration failure, analyzes the Xn interface signaling, checks the gNB-CU/DU synchronization, and provides specific remediation steps. This level of 3GPP-compliant analysis would take a Tier-3 engineer 30-60 minutes to produce manually."

### Demo Query 2: Technical Log Input
> **Type/click**: "Users in sector 3 experiencing intermittent data drops during handover between gNB-CU and gNB-DU. F1 interface latency spikes observed."

### 🎙️ VOICEOVER:
> "Now let's try a more technical engineering input. Notice the model automatically escalates to a deeper protocol analysis, referencing the F1 interface split architecture from TS 38.401, and providing CU-DU synchronization diagnostics."

### Demo Query 3: General Question (shows dual-mode)
> **Type/click**: "What is the difference between 5G NSA and SA architecture?"

### 🎙️ VOICEOVER:
> "The model also functions as a clean engineering assistant for general queries. Here it's not in diagnostic mode — it answers cleanly and concisely. The dual-mode routing happens automatically through the 70B model's attention mechanism."

---

## 🎞️ SCENE 5: Evaluation & Benchmarks (4:30 - 5:30)

### SLIDE: Benchmark Results
> **Show**: Radar chart — Base vs Fine-tuned model comparison

### 🎙️ VOICEOVER:
> "We built a quantitative evaluation framework to prove our fine-tuning actually works.
>
> We ran 30 diverse telecom diagnostic queries through both the raw base Llama model and our fine-tuned version, scoring each response on four dimensions:
>
> **3GPP Compliance** — Does it reference correct specification numbers?
> **Protocol Accuracy** — Does it identify the right protocol layers and procedures?
> **Structural Quality** — Does it produce organized, diagnostic-format responses?
> **Hallucination Rate** — Does it fabricate non-existent specifications?
>
> The fine-tuned model significantly outperforms the base model across all four dimensions, with the most dramatic improvement in structural quality and 3GPP compliance."

### SLIDE: Per-category comparison bar chart

---

## 🎞️ SCENE 6: RAG Augmentation (5:30 - 6:00)

### SLIDE: RAG Architecture
> **Show**: Diagram of RAG pipeline

### 🎙️ VOICEOVER:
> "To take this further, we also built a Retrieval-Augmented Generation layer. When a diagnostic query comes in, we retrieve the most relevant 3GPP specification excerpts from our curated knowledge base and inject them directly into the model's context window.
>
> This means our model doesn't just recall spec numbers from training — it has access to actual specification text at inference time, enabling precise, verifiable citations. Fine-tuning teaches the model *how to think* about telecom problems; RAG provides *what to reference*."

---

## 🎞️ SCENE 7: AMD Platform & Future Work (6:00 - 6:45)

### SLIDE: Hardware Utilization
> **Show**: hardware_utilization.png

### 🎙️ VOICEOVER:
> "None of this would be possible without the AMD Instinct MI300X. With 192 gigabytes of HBM3 memory, we can load a 70-billion parameter model in 4-bit quantization with substantial headroom for inference. On an NVIDIA A100 with 80GB, this model simply wouldn't fit with the same performance characteristics.
>
> The ROCm 7.0 software stack provided seamless PyTorch integration, and the Triton 3.0 kernel compiler enabled efficient attention computation."

### SLIDE: Future Roadmap
### 🎙️ VOICEOVER:
> "Looking ahead, our roadmap includes:
> - **Multi-modal input**: Accept network topology diagrams and PCB captures as visual input
> - **Real-time telemetry integration**: Connect to live gNodeB SNMP feeds for proactive diagnosis
> - **Production serving**: Deploy via vLLM on MI300X for multi-tenant NOC environments
> - **Continuous learning**: Feedback loop from resolved tickets to keep the model current
>
> Thank you for watching. We believe this demonstrates the power of AMD's MI300X hardware combined with intelligent fine-tuning to solve real enterprise telecom challenges."

### SLIDE: Thank You / Team Credits

---

## ⏱️ Timing Summary

| Scene | Duration | Content |
|---|---|---|
| 1. Title & Problem | 45s | Problem framing, why this matters |
| 2. Architecture | 60s | Technical deep-dive |
| 3. Training Results | 45s | Loss curves, curriculum strategy |
| 4. Live Demo | 120s | 3 demo queries in the app |
| 5. Evaluation | 60s | Benchmark charts |
| 6. RAG | 30s | Retrieval augmentation |
| 7. AMD & Future | 45s | Hardware story, roadmap |
| **Total** | **~6:45** | |

---

## 🎯 Key Phrases to Emphasize (for judges)

- "70 billion parameter model fine-tuned with QLoRA on AMD MI300X"
- "Curriculum learning: domain knowledge first, then conversational realism"
- "Novel matrix interpolation merge across 1,120 LoRA parameter matrices"
- "3GPP-specification-compliant root cause analysis in seconds"
- "192 gigabytes of HBM3 enables models impossible on consumer GPUs"
- "RAG-augmented fine-tuned diagnostic engine"
- "Quantitative evaluation proves measurable improvement over base model"
