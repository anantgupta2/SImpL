"""Merge a saturated CoT LoRA adapter into the base model, producing a warm-start
base for staged (CoT-then-understanding) training. The merged base + a fresh
zero-init adapter reproduces the CoT-saturated policy exactly at step 0.

Usage:
    python scripts/claude/merge_cot_adapter.py <base_model> <adapter_dir> <out_dir>
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"[merge] base={base}\n[merge] adapter={adapter}\n[merge] out={out}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[merge] device={device}")
tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"[merge] DONE -> {out}")