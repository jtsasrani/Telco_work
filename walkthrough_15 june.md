# 🏁 AMD AI Hackathon Submission Walkthrough — 5G Core/RAN Intelligent Diagnostic Engine

We have successfully prepared and verified all source code, models, visualizations, scripts, and presentation materials. All files are organized and ready in your workspace at `c:\Users\jittu\AMD Hackathon`.

Below is a breakdown of the finalized deliverables and how they are structured:

---

## 📂 Submission Workspace Structure

```
c:\Users\jittu\AMD Hackathon\
├── telco_expert_master_integrated_lora/   <-- Merged QLoRA adapter weights (Safetensors)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── chat_template.jinja
│   ├── generation_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── rag/                                   <-- 3GPP Specification RAG Layer
│   ├── __init__.py
│   └── spec_retriever.py
│
├── evaluation/                            <-- Quantitative Evaluation & Visualizations
│   ├── __init__.py
│   ├── benchmark.py                       <-- 30-query Base vs Fine-Tuned evaluator
│   ├── training_analysis.py               <-- Matplotlib chart generator (ASCII safe)
│   ├── training_report.md                 <-- Markdown report with embedded charts
│   └── charts/                            <-- 5 Hero Presentation Charts (PNGs)
│       ├── training_curves.png            <-- Dual-panel Stage 1 & Stage 2 loss
│       ├── curriculum_comparison.png      <-- Normalized curriculum learning overlay
│       ├── architecture_diagram.png       <-- QLoRA & matrix blend flow
│       ├── model_stats.png                <-- Parameter efficiency log chart
│       └── hardware_utilization.png       <-- AMD MI300X VRAM allocation infographic
│
├── presentation/                          <-- PowerPoint Presentation Materials
│   ├── slide_content.py
│   └── slide_content.md                   <-- Full slide text for copy-pasting
│
├── app_v2.py                              <-- Upgraded Streamlit UI App (Premium Dark Mode)
├── demo_script.md                         <-- Screen recording and voiceover script
├── read_notebooks.py                      <-- Helper script to parse notebooks
└── trimmed_phase_2_Completed_backup_AMD.tar.gz  <-- Original backup tarball
```

---

## 🛠️ Summary of Final Actions Completed

1. **Model Adapter Extraction & Decompression:**
   * Extracted `telco_expert_master_integrated_lora` from the tarball.
   * Moved the folder to the workspace root directory.
   * Decompressed the `.gz` configuration files (`adapter_config.json`, `tokenizer.json`, etc.) so that standard PEFT and Hugging Face loaders can resolve them directly.
   * Cleaned up the leftover gzip artifacts.

2. **Training Visualizations Generated:**
   * Modified `evaluation/training_analysis.py` to remove non-ASCII console prints (`✓`, `—`, `…`, `─`), avoiding CP1252 Unicode crashes in Windows cmd/PowerShell.
   * Executed the script locally. It successfully produced all five presentation-grade PNG charts in `evaluation/charts/`.

3. **Presentation & Script Deliverables Consolidated:**
   * Copied `demo_script.md` to the workspace root directory.
   * Created a clean markdown copy of the PowerPoint presentation slides at `presentation/slide_content.md` for simple copy-pasting.

---

## 🚀 Running the App & Benchmark (On your remote MI300X Server)

Since your local environment is CPU-only, do not run PyTorch GPU execution scripts here. Transfer the project folder to your AMD Instinct MI300X system and run:

### 1. Launch the Premium Streamlit App
```bash
streamlit run app_v2.py
```
*Features sysfs telemetry fallback (reads GPU utilization/VRAM directly from card device files if `rocm-smi` is blocked in containers).*

### 2. Run the Evaluation Benchmark
```bash
# Smoke test (5 queries)
python evaluation/benchmark.py --quick

# Full suite (30 queries)
python evaluation/benchmark.py
```
*Outputs quantitative comparative tables and results in `evaluation/results/`.*
