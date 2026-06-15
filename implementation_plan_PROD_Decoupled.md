# 🏆 Decoupled Production Serving and Weights Fusing Plan

We are establishing the production serving structure on the branch `feature/production-decoupling`. This architecture transitions our hackathon prototype into an enterprise-grade setup: decoupling the Streamlit UI from direct model weights ownership (offloading inference to an optimized engine like vLLM) and providing a portable, resource-aware weights fusing script.

## User Review Required

> [!IMPORTANT]
> **Base Model Gating & Token Access**:
> Merging LoRA adapters requires loading the unquantized base model (`meta-llama/Llama-3.3-70B-Instruct` or similar) in `bfloat16`. Since this model is gated on Hugging Face, you must have a valid Hugging Face User Access Token with read permissions for the repository.
> 
> **Fusing Hardware Requirements**:
> A 70B model loaded in `bfloat16` consumes ~140 GB of RAM. The script supports CPU-offload merging so it can run on systems with smaller VRAM (e.g., A100 80GB, RTX 4090, or other nodes) as long as there is sufficient system RAM (~200 GB) for holding parameters during CPU processing.

## Open Questions

> [!WARNING]
> None at the moment. The plan covers all bases for both high-memory GPU nodes and memory-constrained offload configurations.

## Proposed Changes

---

### Weights Fusing Utility

#### [NEW] [merge_weights.py](file:///c:/Users/jittu/AMD%20Hackathon/merge_weights.py)
A command-line script to load the base model and LoRA adapter weights, fuse them, and save the resulting unquantized model in `bfloat16` format.

**Key Features:**
- CLI arguments for base model, adapter path, output directory, and HF token.
- Offload folder options to support merging on systems with lower VRAM by moving tensor states to CPU memory.
- Validates directories, VRAM requirements, and authentication prior to running.

---

### Streamlit Decoupled Web Application

#### [NEW] [app_v3_decoupled.py](file:///c:/Users/jittu/AMD%20Hackathon/app_v3_decoupled.py)
A lightweight production-ready iteration of the Streamlit dashboard that offloads inference to a vLLM server via the OpenAI-compatible API.

**Key Features:**
- **Zero VRAM Footprint**: Streamlit loads instantly (under 1 second) and acts strictly as a front-end client, preventing out-of-memory conflicts.
- **Configurable Endpoints**: Sidebar elements to specify the backend host (e.g., `http://localhost:8000/v1`) and the model identifier.
- **Identical Premium UI**: Maintains all custom theme toggles (Light/Dark), last-query statistics, system prompts, carrier profiles, and diagnostic log parsing.
- **Unified GPU Telemetry**: Continues using local sysfs metrics or CLI tools (`amd-smi`/`rocm-smi`) assuming the vLLM server and Streamlit are hosted on the same node.

---

### Project Documentation

#### [MODIFY] [README.md](file:///c:/Users/jittu/AMD%20Hackathon/README.md)
Update documentation to explain:
1. How to run the `merge_weights.py` script to merge adapters.
2. How to start the production vLLM backend server on the AMD MI300X using the merged weights.
3. How to launch the decoupled Streamlit client (`app_v3_decoupled.py`) pointing to the vLLM backend.

## Verification Plan

### Automated Verification
We will run static checking on the files to ensure syntax and imports are valid.
```powershell
python -m py_compile merge_weights.py
python -m py_compile app_v3_decoupled.py
```

### Manual Verification
1. We will verify the command-line options of `merge_weights.py` using `--help` flag locally.
2. We will inspect the UI structure of `app_v3_decoupled.py` to ensure it uses the OpenAI API client cleanly.
3. The user will run the actual merge script and the Streamlit app on the GPU node.
