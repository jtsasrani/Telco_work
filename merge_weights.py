#!/usr/bin/env python3
"""
Merge Weights Utility for 5G Core/RAN Intelligent Diagnostic Engine.
This script fuses PEFT LoRA adapters with the unquantized Llama-3.3-70B base model.
Supports CPU RAM offloading for merging on hardware nodes with limited VRAM.
"""

import os
import argparse
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuses PEFT LoRA adapters into a base causal language model."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name or path of the base unquantized Llama model (e.g., meta-llama/Llama-3.3-70B-Instruct or unsloth/Llama-3.3-70B-Instruct)."
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="telco_expert_master_integrated_lora",
        help="Path to the directory containing the adapter weights and config."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="telco_expert_llama3_3_70b_merged",
        help="Path where the merged model will be saved."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face User Access Token (required for gated Llama models). Defaults to HF_TOKEN env var."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for loading. Use 'cpu' or 'auto' with offloading for low VRAM nodes."
    )
    parser.add_argument(
        "--offload-folder",
        type=str,
        default="offload_weights",
        help="Temporary directory to offload weights if VRAM/RAM is tight."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🧬 5G Core/RAN Intelligent Diagnostic Engine — Weights Fusion Utility")
    print("=" * 70)
    print(f"Base Model:   {args.base_model}")
    print(f"Adapter Path: {args.adapter_path}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Device Map:   {args.device}")
    print("=" * 70)

    # 1. Verification of paths
    if not os.path.exists(args.adapter_path):
        print(f"❌ Error: Adapter path '{args.adapter_path}' does not exist.")
        sys.exit(1)
        
    config_path = os.path.join(args.adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at '{config_path}'. Verify the adapter path.")
        sys.exit(1)
        
    # Check HF Token for gated model
    if not args.hf_token and "meta-llama" in args.base_model:
        print("⚠️ Warning: Accessing 'meta-llama/Llama-3.3-70B-Instruct' usually requires a Hugging Face token.")
        print("   Please pass --hf-token or set the HF_TOKEN environment variable if loading fails.")

    print("🚀 Importing PyTorch and HuggingFace libraries...")
    start_time = time.time()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"❌ Error: Missing required dependencies. Run: pip install torch transformers peft safetensors")
        print(f"Details: {e}")
        sys.exit(1)

    print(f"✓ Libraries loaded successfully (took {time.time() - start_time:.1f}s).")
    
    # Check CUDA availability
    cuda_avail = torch.cuda.is_available()
    print(f"System CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU mode. Loading Llama-3.3-70B in bfloat16 requires ~140GB of system RAM.")

    # Configure device map and offloading
    device_map = args.device
    if device_map == "cuda" and not cuda_avail:
        print("⚠️ CUDA requested but not available. Falling back to device_map='auto'.")
        device_map = "auto"
        
    offload_kwargs = {}
    if device_map == "auto" or device_map == "cpu":
        os.makedirs(args.offload_folder, exist_ok=True)
        offload_kwargs = {
            "offload_folder": args.offload_folder,
        }

    print("\n📥 Step 1: Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            token=args.hf_token if args.hf_token else None,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded.")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        sys.exit(1)

    print("\n📥 Step 2: Loading Base Model in bfloat16 (this might take several minutes)...")
    try:
        # Load base model in bfloat16 format
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token=args.hf_token if args.hf_token else None,
            trust_remote_code=True,
            **offload_kwargs
        )
        print("✓ Base model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        print("Tips: Verify your Hugging Face Token, internet connection, and available RAM/VRAM.")
        sys.exit(1)

    print("\n📥 Step 3: Loading and attaching PEFT LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            device_map=device_map
        )
        print("✓ Adapters attached successfully.")
    except Exception as e:
        print(f"❌ Failed to attach adapter model: {e}")
        sys.exit(1)

    print("\n⚙️ Step 4: Fusing/merging LoRA adapter weights into base model...")
    merge_start = time.time()
    try:
        # Perform merge & unload
        model = model.merge_and_unload()
        print(f"✓ Model merged and unloaded successfully (took {time.time() - merge_start:.1f}s).")
    except Exception as e:
        print(f"❌ Failed to merge and unload adapter weights: {e}")
        sys.exit(1)

    print(f"\n💾 Step 5: Saving the fused model to: {args.output_dir}...")
    save_start = time.time()
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save model
        model.save_pretrained(
            args.output_dir,
            safe_serialization=True
        )
        # Save tokenizer
        tokenizer.save_pretrained(args.output_dir)
        print(f"✓ Fused model and tokenizer saved successfully (took {time.time() - save_start:.1f}s).")
    except Exception as e:
        print(f"❌ Failed to save merged model: {e}")
        sys.exit(1)

    # Clean up offload folder if it's empty
    try:
        if os.path.exists(args.offload_folder) and not os.listdir(args.offload_folder):
            os.rmdir(args.offload_folder)
    except:
        pass

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("🎉 Weight Fusion Process Completed Successfully!")
    print(f"Total time elapsed: {total_time/60:.2f} minutes")
    print(f"Output Directory:   {os.path.abspath(args.output_dir)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
