"""
LLM Fine-tuning with LoRA/QLoRA using HuggingFace PEFT.

Supports:
- LoRA: Low-Rank Adaptation (fp16/fp32)
- QLoRA: Quantized LoRA (int4 via bitsandbytes, CUDA only)

Interview Topics:
  - LoRA: rank decomposition, target modules, alpha scaling
  - QLoRA: NF4 quantization, double quantization, paged optimizers
  - Full fine-tune vs LoRA vs Prompt Tuning — trade-offs
  - Training dynamics: gradient accumulation, warmup, checkpointing
  - Evaluation: perplexity, task accuracy, generation quality
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import yaml
import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
except ImportError:
    AutoModelForCausalLM = None

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
        TaskType,
    )
except ImportError:
    LoraConfig = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None


class LLMFineTuner:
    """
    Fine-tunes language models with LoRA/QLoRA for visual reasoning tasks.

    LoRA (Low-Rank Adaptation):
    ──────────────────────────
    Instead of updating all W weights (d×d matrix), LoRA adds:
      W' = W + ΔW = W + BA
    Where B is d×r and A is r×d (r << d)

    Parameters: 2 × d × r instead of d × d
    Example: d=4096, r=16 → 131K vs 16.7M parameters (0.8%)

    Interview: "How does LoRA work?"
    1. Original weights W are frozen
    2. Two small matrices B (d×r) and A (r×d) are added
    3. Output = Wx + BAx (original + low-rank update)
    4. Alpha/r scaling: ΔW = (α/r) × BA
    5. Merge: at inference time combined as W' = W + BA

    QLoRA extends LoRA with:
    - 4-bit NF4 quantization of base model (bitsandbytes)
    - Double quantization (quantize the quantization constants)
    - Paged optimizers (offload optimizer states to CPU)
    → Enables fine-tuning 65B models on single 48GB GPU
    """

    def __init__(self, config_path: str = "configs/llm_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, model_name: str = None) -> tuple:
        """
        Load base model + tokenizer with optional quantization.

        Interview: "Types of quantization?"
        - FP32: 4 bytes/param — full precision, baseline
        - FP16: 2 bytes/param — half precision, minimal quality loss
        - BF16: 2 bytes/param — better range than FP16 (Ampere+ GPUs)
        - INT8: 1 byte/param — bitsandbytes LLM.int8()
        - INT4/NF4: 0.5 bytes/param — QLoRA, significant compression
        """
        if AutoModelForCausalLM is None:
            raise ImportError("transformers required: pip install transformers")

        model_name = model_name or self.config["model"]["name"]
        print(f"Loading model: {model_name}")
        print(f"  Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional quantization
        model_kwargs = {"trust_remote_code": True}

        quant_config = self.config.get("quantization", {})
        use_quant = quant_config.get("enabled", False) and self.device == "cuda"

        if use_quant:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=quant_config.get("load_in_4bit", True),
                    bnb_4bit_compute_dtype=getattr(
                        torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16")
                    ),
                    bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                print("  Quantization: 4-bit NF4 (QLoRA mode)")
            except ImportError:
                print("  Warning: bitsandbytes not available, using fp32")
        else:
            if self.device == "mps":
                model_kwargs["torch_dtype"] = torch.float32  # MPS prefers fp32
            else:
                model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if use_quant:
            self.model = prepare_model_for_kbit_training(self.model)

        # Move to device if not using device_map
        if "device_map" not in model_kwargs and self.device != "cpu":
            self.model = self.model.to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Memory: ~{total_params * 4 / 1e9:.2f} GB (fp32)")

        return self.model, self.tokenizer

    def apply_lora(self) -> "PeftModel":
        """
        Apply LoRA adapters to the model.

        Interview: "LoRA hyperparameters?"
        - r (rank): 4-64, lower = fewer params, higher = more capacity
          Rule of thumb: r=8 for simple tasks, r=32-64 for complex
        - alpha: scaling factor, typically 2×r
          Effective scale = alpha/r (if alpha=32, r=16 → scale=2)
        - target_modules: which layers get LoRA
          - Attention: q_proj, k_proj, v_proj, o_proj (most common)
          - MLP: gate_proj, up_proj, down_proj (more params, sometimes helpful)
        - dropout: 0.05-0.1 for regularization
        """
        if LoraConfig is None:
            raise ImportError("peft required: pip install peft")
        if self.model is None:
            raise ValueError("Load model first")

        lora_cfg = self.config["lora"]
        config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, config)

        # Print LoRA info
        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"\nLoRA applied:")
        print(f"  Rank (r): {lora_cfg['r']}")
        print(f"  Alpha: {lora_cfg['alpha']}")
        print(f"  Target modules: {lora_cfg['target_modules']}")
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

        return self.model

    def prepare_dataset(self, data_path: str = None) -> tuple:
        """
        Load and tokenize instruction dataset.

        Returns train and eval datasets ready for Trainer.
        """
        if Dataset is None:
            raise ImportError("datasets required: pip install datasets")

        data_path = data_path or self.config["dataset"]["path"]
        with open(data_path) as f:
            raw_data = json.load(f)

        # Format into text
        formatted = []
        for sample in raw_data:
            text = (
                f"<|im_start|>system\n"
                f"You are a visual reasoning assistant that analyzes CV pipeline outputs.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{sample['instruction']}\n\n{sample['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{sample['output']}<|im_end|>"
            )
            formatted.append({"text": text})

        # Create HF dataset
        dataset = Dataset.from_list(formatted)

        # Tokenize
        max_length = self.config["model"]["max_length"]

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

        # Split
        train_ratio = self.config["dataset"]["train_split"]
        split = tokenized.train_test_split(test_size=1 - train_ratio, seed=42)

        print(f"\nDataset prepared:")
        print(f"  Total samples: {len(raw_data)}")
        print(f"  Train: {len(split['train'])}")
        print(f"  Eval: {len(split['test'])}")
        print(f"  Max length: {max_length}")

        return split["train"], split["test"]

    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "output/llm_finetuned",
    ) -> dict:
        """
        Train with HuggingFace Trainer.

        Interview: "Training dynamics?"
        - Gradient accumulation: simulate larger batch on limited GPU
          effective_batch = batch_size × gradient_accumulation_steps
        - Warmup: gradually increase LR to avoid early instability
        - Cosine LR schedule: smooth decay, often better than step decay
        - Max grad norm: clip gradients to prevent explosion
        - Checkpointing: save every N steps for recovery
        """
        cfg = self.config["training"]

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg["max_grad_norm"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            # Platform-specific
            fp16=cfg.get("fp16", False) and self.device == "cuda",
            bf16=cfg.get("bf16", False) and self.device == "cuda",
            dataloader_pin_memory=self.device == "cuda",
            # Disable features that cause issues on MPS
            use_cpu=self.device == "cpu",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # causal LM, not masked LM
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        print(f"\n{'='*60}")
        print("LLM Fine-tuning (LoRA)")
        print(f"{'='*60}")
        print(f"  Epochs: {cfg['epochs']}")
        print(f"  Batch size: {cfg['batch_size']}")
        print(f"  Gradient accum: {cfg['gradient_accumulation_steps']}")
        print(f"  Effective batch: {cfg['batch_size'] * cfg['gradient_accumulation_steps']}")
        print(f"  Learning rate: {cfg['learning_rate']}")
        print(f"  LR scheduler: {cfg['lr_scheduler_type']}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time

        # Evaluate
        eval_result = self.trainer.evaluate()

        metrics = {
            "train_loss": round(train_result.training_loss, 4),
            "eval_loss": round(eval_result["eval_loss"], 4),
            "perplexity": round(math.exp(eval_result["eval_loss"]), 4),
            "train_runtime": round(training_time, 2),
            "train_samples_per_second": round(
                train_result.metrics.get("train_samples_per_second", 0), 2
            ),
            "total_steps": train_result.global_step,
        }

        print(f"\nTraining complete:")
        print(f"  Train loss: {metrics['train_loss']}")
        print(f"  Eval loss: {metrics['eval_loss']}")
        print(f"  Perplexity: {metrics['perplexity']}")
        print(f"  Time: {metrics['train_runtime']:.0f}s")

        return metrics

    def save_model(self, output_dir: str = "output/llm_finetuned"):
        """
        Save LoRA adapters (not full model — much smaller).

        Interview: "How are LoRA models saved?"
        - Only adapter weights are saved (a few MBs)
        - Base model is kept separate (unchanged)
        - Inference: base model + adapter merge → full model
        - Advantage: multiple adapters, same base model
        """
        if self.model is None:
            raise ValueError("No model to save")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

        # Show saved files
        for f in sorted(Path(output_dir).iterdir()):
            size = f.stat().st_size / 1024
            unit = "KB" if size < 1024 else "MB"
            size = size if size < 1024 else size / 1024
            print(f"  {f.name}: {size:.1f} {unit}")

    def load_finetuned(
        self,
        adapter_path: str,
        base_model: str = None,
    ):
        """Load base model + LoRA adapters for inference."""
        base_model = base_model or self.config["model"]["name"]

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32 if self.device == "mps" else torch.float16,
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(base, adapter_path)

        if self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Loaded fine-tuned model from: {adapter_path}")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response with fine-tuned model."""
        if self.model is None:
            raise ValueError("No model loaded")

        gen_config = self.config.get("evaluation", {}).get("generate", {})

        formatted = (
            f"<|im_start|>system\n"
            f"You are a visual reasoning assistant that analyzes CV pipeline outputs.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(formatted, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def compare_base_vs_finetuned(self, test_prompts: list) -> list:
        """
        Compare base model vs fine-tuned model outputs.

        Interview: "How do you evaluate fine-tuning results?"
        - Quantitative: perplexity, eval loss, BLEU/ROUGE
        - Qualitative: A/B comparison, human preference
        - Task-specific: accuracy on visual reasoning questions
        - Overfitting check: train loss << eval loss → overfit
        """
        results = []
        for prompt in test_prompts:
            response = self.generate(prompt)
            results.append({
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response,
            })
            print(f"\nPrompt: {prompt[:80]}...")
            print(f"Response: {response[:200]}...")
        return results
