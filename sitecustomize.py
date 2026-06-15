# 🧬 5G Core/RAN Intelligent Diagnostic Engine — Triton Compatibility Patch
# This file is automatically loaded by Python at startup when in the PYTHONPATH.

try:
    import triton.language as tl
    if not hasattr(tl, "constexpr_function"):
        # Inject fallback pass-through decorator for constexpr_function
        # Since Llama-3.3-70B is dense, FusedMoE kernels are never executed.
        # This allows vLLM to pass import-level inspection without crashing.
        tl.constexpr_function = lambda x: x
        print("🧬 [AMD GPU Compatibility Patch] Injected fallback for tl.constexpr_function")
except Exception:
    pass
