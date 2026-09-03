from typing import Any, Dict, List

from . import default_agents
from parallel_synthesis.prompts import (
    CONTEXT_QA_TASKS,
    PRETRAINING_TASKS,
    build_text_mas_prompt,
)
from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.utils.utils import (
    context_qa_match,
    extract_after_think,
    extract_context_qa_prediction,
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    run_with_timeout,
)
import argparse

class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.default_num_parallel_agents = max(
            1,
            int(getattr(args, "parallel_kv_num_parallel_agents", 3) if args else 3),
        )
        self.agents = default_agents(self.default_num_parallel_agents)
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task

    @staticmethod
    def _agent_ref_index(role: str) -> int:
        role_text = str(role or "").strip().lower()
        legacy = {"planner": 0, "critic": 1, "refiner": 2}
        if role_text in legacy:
            return legacy[role_text]
        if role_text.startswith("sub_agent_"):
            suffix = role_text[len("sub_agent_") :]
            if suffix.isdigit():
                return max(0, int(suffix) - 1)
        return -1

    def _num_parallel_agents_for_items(self, items: List[Dict[str, Any]]) -> int:
        _ = items
        return self.default_num_parallel_agents

    def _agents_for_items(self, items: List[Dict[str, Any]]) -> List[Any]:
        return default_agents(self._num_parallel_agents_for_items(items))

    @classmethod
    def _reference_context_for_agent(cls, item: Dict, role: str) -> str:
        idx = cls._agent_ref_index(role)
        if idx < 0:
            return ""
        contexts = item.get("agent_reference_contexts")
        if isinstance(contexts, list) and idx < len(contexts):
            return str(contexts[idx])
        references = item.get("references", [])
        if isinstance(references, list) and idx < len(references):
            ref = references[idx] if isinstance(references[idx], dict) else {}
            citation = str(ref.get("citation", "")).strip()
            abstract = str(ref.get("abstract", "")).strip()
            if citation:
                return f"[{citation}]\n{abstract}"
            return abstract
        return ""

    def _agent_prompt_context(self, item: Dict, role: str, default_context: str) -> str:
        if role == "judger":
            return default_context
        if self.task in PRETRAINING_TASKS:
            return self._reference_context_for_agent(item, role)
        return default_context

    def _run_batch_impl(self, items: List[Dict], *, use_vllm: bool) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_ttft_sec = None

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        agents = self._agents_for_items(items)
        for agent in agents:
            batch_messages = [
                build_text_mas_prompt(
                    role=agent.role,
                    question=item["question"],
                    context=self._agent_prompt_context(item, agent.role, contexts[idx]),
                    args=self.args,
                )
                for idx, item in enumerate(items)
            ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if use_vllm:
                generated_texts = self.model.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            else:
                generated_texts, _, timing_metrics = self.model.generate_text_batch(
                    input_ids,
                    attention_mask,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    return_timings=True,
                )
                if agent.role == "judger":
                    ttft = timing_metrics.get("ttft_sec")
                    if ttft is not None:
                        batch_ttft_sec = ttft

            for idx in range(batch_size):

                text_out = generated_texts[idx].strip()
                formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if agent.role != "judger":

                    contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                    }
                )
            # import pdb; pdb.set_trace()

        results: List[Dict] = []
        for idx, item in enumerate(items):
            raw_final_text = final_texts[idx]
            final_text = extract_after_think(raw_final_text)

            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                if pred is None:
                    ok = False
                    error_msg = f'No numeric answer parsed. Pred: {pred}, Gold: {gold}'
                else:
                    try:
                        pred_int = int(pred)
                        gold_int = int(gold)
                        ok = (pred_int == gold_int)
                        error_msg = None
                    except (ValueError, TypeError):
                        ok = False
                        error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                if self.task in PRETRAINING_TASKS:
                    pred = final_text.strip()
                    gold = str(item.get("gold", "")).strip()
                    ok = (pred == gold) if (pred and gold) else False
                    error_msg = None
                elif self.task in CONTEXT_QA_TASKS:
                    pred = extract_context_qa_prediction(raw_final_text)
                    gold = str(item.get("gold", "")).strip()
                    ok = context_qa_match(item, pred)
                    error_msg = None
                else:
                    pred = normalize_answer(extract_gsm8k_answer(final_text))
                    gold = item.get("gold", "")
                    ok = (pred == gold) if (pred and gold) else False
                    error_msg = None

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": raw_final_text,
                    "agents": agent_traces[idx],
                    "ttft_sec": batch_ttft_sec,
                    "correct": ok,
                }
            )
        return results

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        return self._run_batch_impl(items, use_vllm=False)

    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        return self._run_batch_impl(items, use_vllm=True)

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
