# 🧠 5G Core/RAN Intelligent Diagnostic Engine
### AMD AI Hackathon Submission — Track 3: Fine-Tuning
**Host Platform:** AMD Instinct™ MI300X Accelerator (192GB HBM3 VRAM, ROCm 7.0)

An enterprise-grade protocol engineering workbench that takes raw text input / network logs and streams precise, 3GPP-specification-compliant root-cause analyses in real-time.

---

## 🚀 Quick Start Instructions

### 1. Repository Setup & Model Loading (On the AMD Server)
Clone the repository and prepare the environment inside your Jupyter/AMD container:
```bash
# Navigate to your workspace and pull the latest code
cd /workspace
git init
git remote add origin https://github.com/jtsasrani/Telco_work.git
git pull origin main

# Extract and decompress the model weights
tar -xf trimmed_phase_2_Completed_backup_AMD.tar.gz
mv phase_2_Completed_backip_AMD/telco_expert_master_integrated_lora /workspace/
cd /workspace/telco_expert_master_integrated_lora
gunzip -f *.gz
```

### 2. Run the Streamlit UI Application
Start the premium Day/Night mode web interface using the following command:
```bash
streamlit run app_v2.py --server.port 8503 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```
*Access the UI in your browser via the custom port `8503` forwarding.*

### 3. Run the Evaluation Benchmark
Run the quantitative benchmark comparison comparing the Raw Base Llama-3.3-70B model against our Fine-Tuned model:
```bash
# Run a quick 5-query smoke test
python evaluation/benchmark.py --quick

# Run the full 30-query evaluation suite
python evaluation/benchmark.py
```
*Results, tables, and comparison charts will be outputted to `/workspace/evaluation/results/`.*

---

## ⚡ Production Deployment & Scaling (Decoupled Serving)

To transition from the Jupyter development container to a high-throughput, low-latency production NOC environment, we support **decoupled serving** utilizing **vLLM** on the AMD MI300X and a lightweight, zero-VRAM frontend client.

### 1. Merge LoRA Weights
Standard 4-bit loading does not support direct weight merging. Run the portable fusion script to merge the trained LoRA adapters into the unquantized base model. This can be run on the main node or any other GPU system (supports CPU offloading if VRAM is limited):

```bash
# Run the merge utility (requires HuggingFace token for gated Llama access)
python merge_weights.py \
    --base-model meta-llama/Llama-3.3-70B-Instruct \
    --adapter-path telco_expert_master_integrated_lora \
    --output-dir /workspace/telco_expert_llama3_3_70b_merged \
    --hf-token <YOUR_HF_TOKEN>
```
*Options like `--device cpu` and `--offload-folder temp_dir` can be used to merge on systems with standard system RAM and smaller GPUs.*

### 2. Start the vLLM Serving Engine
Host the fused model using vLLM, which utilizes AMD MI300X vLLM kernels (PagedAttention, continuous batching) for maximum token-generation speed:

```bash
# Start the OpenAI-compatible vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/telco_expert_llama3_3_70b_merged \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096
```

### 3. Launch the Decoupled Client
Launch the client application, which communicates with the vLLM engine via the OpenAI API protocol. This frontend initializes in < 1 second and consumes ~0 VRAM, preventing multi-process GPU conflicts:

```bash
streamlit run app_v3_decoupled.py --server.port 8503 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```
*Configure the API host (`http://localhost:8000/v1`) and Model ID directly from the "Inference API Settings" panel in the sidebar.*

---

## 🏗️ Technical Architecture & Innovations

### 1. QLoRA Quantization
The base model (`Meta-Llama-3.3-70B-Instruct`) is loaded in 4-bit NF4 quantization (~38GB VRAM footprint). Through QLoRA (Rank 16, Alpha 16), we train only **207 Million parameters** (0.29% of the total model), targeting all query, key, value, output, and gating projection layers.

### 2. Two-Stage Curriculum Learning
We train our adapters sequentially to prevent catastrophic forgetting:
* **Stage 1 (3GPP Domain):** Trained on GSMA/ot-lite specifications corpus (300 steps, loss: `3.70 → 0.93`).
* **Stage 2 (Conversational Realism):** Fine-tuned on telecom customer service transcripts (150 steps, loss: `4.29 → 0.14`).

### 3. LoRA Matrix Interpolation Merge
Rather than using standard merges, we perform an element-wise PyTorch linear blend across all **1,120 LoRA parameter matrices**:
$$\mathbf{W}_{merged}[i] = 0.5 \mathbf{W}_{stage1}[i] + 0.5 \mathbf{W}_{stage2}[i]$$
This yields a single merged adapter that retains both low-level protocol logic and natural dialogue.

### 4. 3GPP Retrieval-Augmented Generation (RAG)
During inference, queries are matched against a curated 3GPP specs corpus (covering TS 38.331, TS 23.501, TS 24.229, etc.). The top-3 context snippets are injected into the system prompt, ensuring the model's citations are grounded in actual specification text.

---

## 📊 Quantitative Evaluation Results

Scored across 30 test queries spanning 7 categories (Handover Failures, VoNR Call Setups, IMS Signalling, Beam Management, 5G SA Core, QoS flow establishment, and RAN Energy Optimization):

| Evaluation Metric | Base Llama-3.3 | Fine-Tuned Model | Improvement |
|---|:---:|:---:|:---:|
| **3GPP Compliance** | 35% | **85%** | 📈 **+143%** |
| **Protocol Accuracy** | 40% | **80%** | 📈 **+100%** |
| **Structural Quality** | 25% | **90%** | 📈 **+260%** |
| **Hallucination Rate** | 30% | **15%** (rate) | 📉 **-50%** |
| **Overall Composite Score** | **32%** | **85%** | 📈 **+166%** |

---

## ⚡ AMD Instinct™ MI300X Hardware Advantage

* **VRAM Headroom:** Loading the 70B model + KV-cache + RAG contexts consumes **~45GB VRAM**. 
* **MI300X Advantage (192GB HBM3):** Operates at only **23.4% VRAM utilization**, leaving over **147GB of free headroom** for long context sequences (up to 128K), high-throughput batching, and serving multiple concurrent users in a production NOC environment.
* **ROCm 7.0 Software Stack:** Full native support for PyTorch 2.10, Triton 3.0.0 kernels, and bitsandbytes ROCm quantization.

---

## 📂 Deliverables Package

* `app_v2.py` — Development Streamlit UI with inline PyTorch model loading and inference pipelines.
* `app_v3_decoupled.py` — Production-grade Streamlit UI client that offloads model execution to a decoupled backend engine (e.g. vLLM) for zero-VRAM frontend serving.
* `merge_weights.py` — Portable weights fusion script that merges LoRA adapters with unquantized base models (supports CPU-offload for low VRAM nodes).
* `demo_script.md` — A structured 5-7 minute video recording narration script.
* `evaluation/` — Scripts and assets for quantitative benchmark evaluations, including training analysis curves.
* `presentation/` — Detailed PowerPoint slide structure in `slide_content.md`.
* `rag/` — RAG indexing and retrieval modules.
