"""Generation backends behind a single interface.

    apply_template(messages) -> str         # one templated prompt
    generate(prompts, max_new_tokens, temperature, n) -> List[List[str]]
        outer list  = one entry per prompt
        inner list  = n sampled completions for that prompt (n>=1)

Backends:
  * OpenAICompatGenerator - OpenAI-compatible chat API (e.g. NVIDIA
                      integrate.api.nvidia.com): inference runs in the cloud, so
                      NO local GPU is needed — runs from a laptop over HTTP.
  * VLLMGenerator   - fast batched inference (preferred on Kaggle 2xT4)
  * HFGenerator     - transformers + bitsandbytes 4-bit fallback
  * OracleMock      - no model; returns gold wrapped in noisy JSON (tests the
                      WHOLE prompt->parse->write->eval pipeline locally, no GPU)
  * EmptyMock       - returns {"answers": []} for everything (reproduces the
                      "predict nothing" 0.203 floor through the real pipeline)

Only the mock backends import nothing heavy, so the pipeline is exercisable on
a laptop. vLLM / torch are imported lazily inside the GPU backends.
"""
from __future__ import annotations
import json
from typing import Dict, List, Tuple

# delimiter used by mock backends to smuggle subject/relation through the
# (subject -> templated prompt -> generate) path. Plain ASCII, never appears
# in a real subject string.
MOCK_DELIM = " |##REL##| "


class BaseGenerator:
    def apply_template(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def generate(self, prompts: List[str], max_new_tokens: int,
                 temperature: float = 0.0, n: int = 1) -> List[List[str]]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Mock backends (no GPU) — for local plumbing tests.
# --------------------------------------------------------------------------- #
def _field(messages: List[Dict[str, str]], prefix: str) -> str:
    for line in messages[-1]["content"].splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


class EmptyMock(BaseGenerator):
    def apply_template(self, messages):
        return "x"

    def generate(self, prompts, max_new_tokens, temperature=0.0, n=1):
        return [['{"answers": []}'] * n for _ in prompts]


class OracleMock(BaseGenerator):
    """Returns the gold answer (first alias) for each (subject,relation),
    wrapped in deliberately messy text (code fence + trailing prose) so the
    parser is exercised. gold_map: {(subject, relation): [first_alias,...]}."""
    def __init__(self, gold_map: Dict[Tuple[str, str], List[str]]):
        self.gold_map = gold_map

    def apply_template(self, messages):
        return _field(messages, "Subject:") + MOCK_DELIM + _field(messages, "Relation:")

    def generate(self, prompts, max_new_tokens, temperature=0.0, n=1):
        out = []
        for key in prompts:
            subj, _, rel = key.partition(MOCK_DELIM)
            gold = self.gold_map.get((subj, rel), [])
            blob = "```json\n" + json.dumps({"answers": gold}, ensure_ascii=False) + "\n```\nDone."
            out.append([blob] * n)
        return out


# --------------------------------------------------------------------------- #
# OpenAI-compatible cloud API (NVIDIA build.nvidia.com / integrate.api.nvidia.com).
# No local GPU: inference runs in the cloud, driven over HTTP from a laptop.
# --------------------------------------------------------------------------- #
class OpenAICompatGenerator(BaseGenerator):
    def __init__(self, model_path: str,
                 base_url: str = "https://integrate.api.nvidia.com/v1",
                 api_key_env: str = "NVIDIA_API_KEY",
                 max_workers: int = 4, max_retries: int = 6, timeout: float = 120.0):
        import os
        from openai import OpenAI                       # lazy; no torch needed
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"set the {api_key_env} environment variable (your nvapi-... key)")
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        self.model = model_path
        self.max_workers = max_workers
        self.max_retries = max_retries

    def apply_template(self, messages):
        return messages                                 # chat API consumes messages directly

    def _call(self, messages, max_new_tokens, temperature):
        """One chat completion with exponential backoff on rate-limit/transient errors."""
        import time
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=float(temperature), max_tokens=max_new_tokens, n=1)
                return r.choices[0].message.content or ""
            except Exception as e:                      # 429 / 5xx / timeouts
                if attempt == self.max_retries - 1:
                    print(f"    !! API call failed after retries: {type(e).__name__}", flush=True)
                    return ""
                time.sleep(min(delay, 30.0)); delay *= 2
        return ""

    def generate(self, prompts, max_new_tokens, temperature=0.0, n=1):
        from concurrent.futures import ThreadPoolExecutor
        # one task per (prompt, sample): uniform handling of n>1 self-consistency
        tasks = [(i, j) for i in range(len(prompts)) for j in range(n)]
        out: List[List[str]] = [["" for _ in range(n)] for _ in prompts]

        def work(t):
            i, j = t
            out[i][j] = self._call(prompts[i], max_new_tokens, temperature)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            list(ex.map(work, tasks))
        return out


# --------------------------------------------------------------------------- #
# vLLM backend (preferred on Kaggle).
# --------------------------------------------------------------------------- #
class VLLMGenerator(BaseGenerator):
    def __init__(self, model_path: str, tensor_parallel_size: int = 1,
                 quantization: str | None = None, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.92, dtype: str = "auto"):
        from vllm import LLM                       # lazy import
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._tmpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
        # Qwen3 etc. expose a thinking switch; disable it for terse JSON output.
        try:
            if "enable_thinking" in self.tokenizer.apply_chat_template.__doc__ or True:
                self._tmpl_kwargs["enable_thinking"] = False
                _ = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "hi"}], **self._tmpl_kwargs)
        except TypeError:
            self._tmpl_kwargs.pop("enable_thinking", None)
        self.llm = LLM(
            model=model_path, tensor_parallel_size=tensor_parallel_size,
            quantization=quantization, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, dtype=dtype,
            trust_remote_code=True,
        )

    def apply_template(self, messages):
        return self.tokenizer.apply_chat_template(messages, **self._tmpl_kwargs)

    def generate(self, prompts, max_new_tokens, temperature=0.0, n=1):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature,
                            top_p=0.95 if temperature > 0 else 1.0,
                            max_tokens=max_new_tokens)
        outs = self.llm.generate(prompts, sp)
        return [[o.text for o in req.outputs] for req in outs]


# --------------------------------------------------------------------------- #
# transformers + bitsandbytes 4-bit fallback.
# --------------------------------------------------------------------------- #
class HFGenerator(BaseGenerator):
    def __init__(self, model_path: str, load_in_4bit: bool = True,
                 device_map: str = "auto", dtype: str = "float16"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.torch = torch
        td = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map, quantization_config=qcfg,
            torch_dtype=td, trust_remote_code=True)
        self.model.eval()

    def apply_template(self, messages):
        try:  # Qwen3 etc.: force thinking OFF so we get terse JSON, not <think> blocks
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, max_new_tokens, temperature=0.0, n=1, batch_size=8):
        torch = self.torch
        results: List[List[str]] = []
        do_sample = temperature > 0
        eff_bs = max(1, batch_size // max(1, n))   # n return-seqs per prompt -> shrink batch to fit VRAM
        for i in range(0, len(prompts), eff_bs):
            batch = prompts[i:i + eff_bs]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=0.95 if do_sample else None,
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id)
            gen = gen[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for j in range(len(batch)):
                results.append(texts[j * n:(j + 1) * n])
        return results
