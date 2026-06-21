"""Generation backends behind a single interface.

    apply_template(messages) -> str         # one templated prompt
    generate(prompts, max_new_tokens, temperature, n) -> List[List[str]]
        outer list  = one entry per prompt
        inner list  = n sampled completions for that prompt (n>=1)

Backends:
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
    def __init__(self, model_path: str, load_in_4bit: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", quantization_config=qcfg,
            torch_dtype=torch.float16, trust_remote_code=True)
        self.model.eval()

    def apply_template(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, max_new_tokens, temperature=0.0, n=1, batch_size=8):
        torch = self.torch
        results: List[List[str]] = []
        do_sample = temperature > 0
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
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
