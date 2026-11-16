"""llmprune: magnitude, attention-head and layer pruning of LLaMA-2 with LoRA recovery.

Companion code for the Master's thesis "Pruning Large Language Models" (M. Farzand, Istanbul Aydin
University). The modules that need PyTorch (modeling, importance, pruning, data, evaluation, benchmark,
lora, pipeline, selftest) are imported lazily, so ``import llmprune`` and the pure-Python modules
(selection, counting, metrics, config, status, report) work on a machine without a GPU stack.
"""

__version__ = "1.0.0"
