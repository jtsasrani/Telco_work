# 📊 5-Slide Presentation Deck Content & Video Script
## TelcoDiagnose-70B: AI-Powered 5G Core/RAN Intelligent Diagnostic Engine
**Hackathon Track:** Track 3: Fine-Tuning  
**Hardware Platform:** AMD Instinct™ MI300X Accelerator (192GB HBM3 VRAM, ROCm 7.0)

---

### 🛝 SLIDE 1: BASIC Information

*   **Slide Title:** TelcoDiagnose-70B: AI-Powered 5G Core/RAN Intelligent Diagnostic Engine
*   **Slide Subtitle:** Real-Time, 3GPP-Compliant Diagnostic Intelligence on AMD Instinct™ MI300X
*   **Visual Layout Recommendation:** 
    *   Dark Mode, premium design with a deep charcoal background (`#0a0a0f`) and AMD Red (`#ED1C24`) accents.
    *   Left side: Clean typographic hierarchy of Title, Subtitle, and Short Description.
    *   Right side: A high-tech visual overlay, showing the AMD Instinct™ MI300X logo, a network node connection graphic, and the team details in a glowing glassmorphism panel.
*   **Core Slide Text:**
    *   **Short Description:**
        *   **WHAT:** A domain-specialized, 70-billion-parameter LLM intelligence platform that automatically analyzes complex 5G network logs and subscriber complaints to trace protocol-level failures.
        *   **WHY:** 5G networks split architectures (gNB-CU/DU) and virtualize functions, causing massive diagnostic bottlenecks ($1.2M average downtime cost/hour). Generic LLMs suffer from 30%+ specification hallucination rates, making them unusable.
        *   **HOW:** Fine-tuned via a dual-stage domain curriculum (3GPP standards + call transcripts), blended via element-wise LoRA matrix interpolation, and grounded using a custom SOTA Dense-Sparse Hybrid RAG search fused via Reciprocal Rank Fusion (RRF) and reranked via a Cross-Encoder.
    *   **Team Name:** Team 1119
    *   **Team Members & Roles:**
        *   **Jittu** (jtsasrani@gmail.com) — Lead AI Engineer & 5G Protocol Architect
*   **Demo Video Voiceover Script (Duration ~45s):**
    > *"Hello, we are Team 1119, and we are excited to present TelcoDiagnose-70B, an AI-powered 5G Core and RAN Intelligent Diagnostic Engine. Today's virtualized, split-gNodeB architectures have introduced massive protocol complexity. When cellular connectivity drops or handovers fail, NOC engineers must manually search through thousands of pages of complex 3GPP specifications. This manual process takes hours, leading to expensive network downtime costing upwards of 1.2 million dollars an hour. Generic AI assistants fail because they hallucinate specification numbers. TelcoDiagnose-70B solves this by combining a dual-stage curriculum-fine-tuned Llama-3.3-70B model with a State-of-the-Art dense-sparse Hybrid RAG pipeline running on the powerful AMD Instinct MI300X GPU. Let's look at the problem in detail."*

---

### 🛝 SLIDE 2: Problem & Context

*   **Slide Title:** The Problem: 5G Network Diagnostic Complexity
*   **Visual Layout Recommendation:**
    *   Split-screen layout.
    *   Left Column: Focuses on network complexity, showing an icon-led flowchart of a split gNodeB (UE → gNB-DU → F1AP → gNB-CU → XnAP → Core AMF/UPF) demonstrating where signaling errors occur.
    *   Right Column: Emphasizes the technical gap of current systems. Highlights the "$1.2M average cost per hour of downtime" statistic in a large, bold red font (`#ED1C24`) with a glowing callout card.
*   **Core Slide Text:**
    *   **Problem Statement:** Cellular connection anomalies (handover drops, VoNR registration failures, beam management drops) are buried inside thousands of multi-protocol log lines spanning NAS, RRC, NGAP, and F1AP.
    *   **Current Solutions & Critical Gaps:**
        *   *Manual Trace Analysis:* Requires Tier-3 engineers to spend hours cross-referencing raw hex logs against 500+ distinct 3GPP specification manuals.
        *   *Tribal Knowledge Dependency:* High resolution latency; expertise is hard to scale and leaves with senior staff.
        *   *Generic LLMs (e.g., ChatGPT):* Hallucinate invalid spec numbers (e.g., citing TS 99.999), invent non-existent protocol messages, and lack context grounding.
    *   **Target Users & Stakeholders:** NOC (Network Operations Center) engineers, Tier-3 carrier support teams, telecom operators, and Managed Service Providers (MSPs).
    *   **Mapped Hackathon Challenge:** Track 3 (Fine-Tuning) — adapting a 70B parameter model on highly dense domain datasets and serving it at production scale using AMD ROCm hardware.
*   **Demo Video Voiceover Script (Duration ~60s):**
    > *"In a modern 5G Standalone network, signals are constantly routed between a split user plane and control plane. Tracing an anomaly—like a handover failure on a highway or a VoNR call setup drop—requires analyzing multi-layered logs containing F1AP, NGAP, RRC, and NAS messages. Currently, Tier-3 engineers must manually parse these logs, matching them page-by-page against 500 different 3GPP specification documents. This causes a massive bottleneck. The expertise is trapped in tribal knowledge, and downtime accumulates fast. We tried using generic, off-the-shelf LLMs to solve this, but they repeatedly hallucinate invalid specification numbers and make up protocol behaviors. For network operations, a hallucinated diagnostic is worse than no diagnostic at all. This is the challenge we mapped for Track 3: utilizing fine-tuning to instill deep 3GPP standards compliance and low-level protocol accuracy into a 70B parameter model."*

---

### 🛝 SLIDE 3: Solution Overview & Architecture

*   **Slide Title:** Technical Architecture: Hybrid RAG & Blended Fine-Tuning
*   **Visual Layout Recommendation:**
    *   Full slide dedicated to a clean architectural flowchart (using the Mermaid structure below).
    *   Use contrasting card containers for the **Base Model (4-bit)**, the **Training Curriculum**, the **Inference Pipeline**, and the **Production decoupled serving client**.
    *   Use different colored arrows to represent the offline training path (blue) and the live inference path (red).
*   **Core Slide Text:**
    *   **AI Approach:**
        *   *Two-Stage Curriculum Learning:* Sequentially trains domain logic, then dialogue capability, to prevent catastrophic forgetting.
        *   *SafeTensors Matrix Interpolation:* Performs element-wise linear blending across **1,120 LoRA matrices** to combine stages without parameter clash.
        *   *SOTA Dense-Sparse Hybrid RAG:* Merges semantic similarity search (FAISS) with exact keyword matching (BM25) via Reciprocal Rank Fusion (RRF), refined with a Cross-Encoder reranker.
        *   *Decoupled serving:* Fused model served in unquantized 16-bit `bfloat16` via vLLM, with an instant-boot frontend client.
    *   **Key Technologies:** PyTorch 2.10, PEFT, bitsandbytes ROCm, FAISS, Sentence-Transformers, vLLM, Streamlit.
    *   **Datasets & Data Volumes:**
        *   *GSMA/ot-lite specifications corpus* (`3gpp_tsg` + `teleqna`) — 300 steps.
        *   *African Carrier Customer Call transcripts* — 150 steps.
        *   *RAG Database:* 16 core 5G specifications (TS 38.331, TS 23.501, etc.) parsed into 6,621 clean chunks.
    *   **What was Built:** Core hybrid indexer, retrieval fusion module, FastAPI serving server, zero-VRAM interactive Streamlit UI, and benchmark evaluator.
*   **Visual Diagram (Insert as flow diagram):**
    ```
    [Meta-Llama-3.3-70B] ────> QLoRA (Rank 16, Alpha 16) targeting 7 projection modules (207M params)
                                         │
                   ┌─────────────────────┴─────────────────────┐ (Curriculum)
                   ▼                                           ▼
       [Stage 1: 3GPP Domain]                       [Stage 2: Conversational]
       GSMA/ot-lite (300 steps)                  Africa Transcripts (150 steps)
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                            [Matrix Interpolation Merge]
                            50/50 element-wise PyTorch blend
                                         │
                                         ▼
                             [Production Model Deployment]
                             Served in 16-bit bfloat16 via vLLM
                                         ▲
                                         │ (Retrieval Context Grounding)
                           [SOTA Dense-Sparse Hybrid RAG]
                    Dense (FAISS) + Sparse (BM25) ──> RRF ──> Rerank
                                         ▲
                                         │
                             [Zero-VRAM Streamlit UI Client]
    ```
*   **Demo Video Voiceover Script (Duration ~90s):**
    > *"Here is our core technical architecture. To build TelcoDiagnose-70B, we started with Llama-3.3-70B loaded in 4-bit quantization on a single AMD Instinct MI300X. We injected QLoRA adapters targeting 207 Million parameters across all key projection modules. We trained this model in a two-stage curriculum: Stage 1 adapts it to the 3GPP domain syntax using GSMA/ot-lite specs. Stage 2 refines its conversational realism using customer service call logs. Rather than loading both adapters at runtime, we performed element-wise PyTorch matrix interpolation across all 1,120 parameter matrices. During inference, we run a SOTA Hybrid retrieval pipeline. When a query is submitted, it is searched against 16 key 5G specifications parsed into over 6,600 chunks. We retrieve semantic matches using dense FAISS vector search, and combine them with exact keyword hits from a custom BM25 retriever. These ranks are combined using Reciprocal Rank Fusion and refined with a Cross-Encoder reranker. Finally, the top-3 specification chunks are injected into the context, prompting our fine-tuned model to stream a fully grounded root-cause trace. In production, this model is served in full 16-bit precision using vLLM, fully decoupled from the Streamlit UI frontend to eliminate GPU process conflicts."*

---

### 🛝 SLIDE 4: Details: Performance, Scale, Time

*   **Slide Title:** Quantitative Performance, Scale, & GPU Details
*   **Visual Layout Recommendation:**
    *   Left side: A structured comparison table showing the benchmark results (Base Llama-3.3 vs. TelcoDiagnose-70B). Insert the generated `charts/model_stats.png` or `charts/curriculum_comparison.png` to illustrate parameter efficiency and learning trends.
    *   Right side: A large hardware infographic displaying the AMD Instinct™ MI300X VRAM allocation. Show the VRAM breakdown as a stacked bar chart or use the generated `charts/hardware_utilization.png`.
*   **Core Slide Text:**
    *   **Fine-Tuning Stats:**
        *   *Trainable Parameters:* 207 Million (0.29% of 70B model).
        *   *Training Time & Steps:* 450 total steps, completed in under an hour on a single MI300X.
        *   *Loss Reduction:* Stage 1 (3GPP) loss dropped **74.9%** (`3.70 → 0.93`); Stage 2 (Conversational) loss dropped **96.8%** (`4.29 → 0.14`).
    *   **Serving Throughput & Latency:**
        *   *Local quantized inference:* Streams at 15–20 tokens/sec.
        *   *Decoupled production vLLM serving:* Achieves **50+ tokens/sec** per user with sub-second prompt evaluation.
    *   **GPU VRAM Allocations & AMD Advantage:**
        *   *Quantized model footprint:* ~38GB.
        *   *Peak training VRAM:* ~45GB (only 23.4% of MI300X capacity).
        *   *Production 16-bit serving VRAM:* ~140GB (served comfortably on a single GPU).
        *   **The MI300X Enabler (192GB HBM3):** Competing 80GB GPUs cannot host 16-bit 70B models without multi-GPU pooling or slow CPU offloading. The MI300X provides **52GB of free headroom** in production, enabling long context sequences (128K) and multi-user concurrency.
*   **Performance Metrics Table:**
    | Evaluation Dimension | Base Llama-3.3 | TelcoDiagnose-70B | Improvement |
    | :--- | :---: | :---: | :---: |
    | **3GPP Compliance** | 35% | **85%** | 📈 **+143%** |
    | **Protocol Accuracy** | 40% | **80%** | 📈 **+100%** |
    | **Structural Quality** | 25% | **90%** | 📈 **+260%** |
    | **Hallucination Rate** | 30% | **15%** (rate) | 📉 **-50%** |
    | **Overall Composite Score** | **32%** | **85%** | 📈 **+166%** |
*   **Demo Video Voiceover Script (Duration ~70s):**
    > *"To evaluate our performance, we built a quantitative benchmark suite consisting of 30 complex scenarios across 7 failure categories. The results were outstanding. Our fine-tuned and grounded model increased overall composite diagnostic accuracy from 32% to 85%—a 166% improvement over base Llama-3.3. Crucially, the hallucination rate was cut in half, while 3GPP citation compliance jumped from 35% to 85%. In terms of scale and resource utilization, QLoRA was incredibly efficient, training only 207 million parameters—0.29% of the model. During training, peak VRAM on our AMD Instinct MI300X was only 45 gigabytes. In production, we serve the model in its unquantized 16-bit format using vLLM. This model requires 140 gigabytes of VRAM. While standard 80 gigabyte GPUs are completely unable to serve this model on a single card, the MI300X hosts it comfortably, leaving 52 gigabytes of free headroom to handle long protocol traces and support continuous batching for multiple NOC engineers."*

---

### 🛝 SLIDE 5: Summary

*   **Slide Title:** Value, Innovation, & Future Roadmap
*   **Visual Layout Recommendation:**
    *   Left side: Focus on Business Value (downtime minimization, labor savings) using prominent metrics.
    *   Right side: Interactive/Link panel containing the Github repository link, a mock QR code pointing to the code, future roadmap items, and a thank you message.
*   **Core Slide Text:**
    *   **Expected Impact & Enterprise Value:**
        *   **90%+ Reduction in MTTR (Mean Time to Resolution):** Translates complex log investigations from hours to seconds.
        *   **Downtime Cost Savings:** Proactive, precise root-cause analysis reduces multi-million dollar outage risks for carriers.
        *   **Operational Efficiency:** Elevates junior NOC technicians, reducing escalation rates to senior protocol experts.
    *   **Key Differentiators & Innovations:**
        *   *Matrix Blending:* Fuses domain-specific and conversational adapters with zero latency overhead.
        *   *SOTA Lexical-Semantic RAG:* Fuses Dense (FAISS) and Sparse (BM25) ranks via Reciprocal Rank Fusion, with Cross-Encoder attention reranking.
        *   *Decoupled Zero-VRAM Architecture:* High-concurrency production-ready deployment.
    *   **Future Work Roadmap:**
        *   *Q3 2026:* Scale Hybrid RAG index to encompass all 500+ active 3GPP specifications.
        *   *Q4 2026:* Add multi-modal ingestion (network topology charts, Wireshark `.pcap` trace uploads).
        *   *Q1 2027:* Integrate live gNodeB SNMP traps and telemetry streams for automated anomaly alerting.
    *   **Deliverable Links:**
        *   **Code Base Repository:** https://github.com/jtsasrani/Telco_work.git
        *   **Demo Video Recording:** *[Insert Video URL Here]*
*   **Demo Video Voiceover Script (Duration ~45s):**
    > *"In summary, TelcoDiagnose-70B delivers massive business value to telecom operators. By reducing diagnostic times from hours to seconds, we directly address the 1.2 million dollar hourly downtime bottleneck, improving operational efficiency and empowering junior engineers. Our core innovations—matrix blending, lexical-semantic rank fusion, and a decoupled production-ready serving architecture—showcase a complete, production-ready system. On our future roadmap, we plan to index the entire 3GPP specification database and support multi-modal Wireshark PCAP trace files. All of our code is open-sourced and available on Github. Thank you for your time, and we look forward to powering the future of telecom intelligence with AMD Instinct!"*
