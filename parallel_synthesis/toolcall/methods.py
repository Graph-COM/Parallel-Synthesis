import argparse
from typing import Any, Dict, List, Optional

import torch

from parallel_synthesis.methods import default_agents
from parallel_synthesis.methods.parallel_kv import ParallelKV
from parallel_synthesis.models import ModelWrapper, _past_length

from .prompts import (
    build_baseline_user_prompt,
    build_parallel_kv_judger_prompt,
    build_textmas_role_prompt,
    build_tool_system_prompt,
)
from .react_agent import ToolCallingReactAgent
from .scoring import extract_answer_text, free_form_exact_match
from .tools import ToolRegistry


class _ToolCallingMethodBase:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        generate_bs: int,
        max_tool_steps: int,
        tools: ToolRegistry,
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, int(generate_bs))
        self.max_tool_steps = max(1, int(max_tool_steps))
        self.tools = tools
        self.args = args
        self.print_intermediate = bool(getattr(args, "print_intermediate", False))
        self.print_intermediate_max_chars = int(
            getattr(args, "print_intermediate_max_chars", 1200)
        )
        self.task = getattr(args, "task", "")

    def _new_react_agent(self, *, verbose_prefix: str = "") -> ToolCallingReactAgent:
        return ToolCallingReactAgent(
            self.model,
            self.tools,
            max_steps=self.max_tool_steps,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=self.print_intermediate,
            verbose_max_chars=self.print_intermediate_max_chars,
            verbose_prefix=verbose_prefix,
        )

    def _system_prompt(self) -> str:
        return build_tool_system_prompt(self.tools.schemas())

    @staticmethod
    def _serialize_tool_calls(logs: List[Any]) -> List[Dict[str, Any]]:
        out = []
        for x in logs:
            out.append(
                {
                    "name": str(getattr(x, "name", "")),
                    "arguments": dict(getattr(x, "arguments", {})),
                    "result_preview": str(getattr(x, "result_preview", "")),
                }
            )
        return out

    @staticmethod
    def _serialize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for msg in messages or []:
            out.append(
                {
                    "role": str(msg.get("role", "")),
                    "content": str(msg.get("content", "")),
                }
            )
        return out

    @staticmethod
    def _serialize_text_list(values: List[Any]) -> List[str]:
        return [str(v) for v in (values or [])]

    @staticmethod
    def _build_result(
        *,
        item: Dict[str, Any],
        pred_text: str,
        raw_text: str,
        agents: List[Dict[str, Any]],
        ttft_sec: Optional[float] = None,
        cache_prepare_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        gold = str(item.get("gold", ""))
        ok = free_form_exact_match(pred_text, gold)
        return {
            "question": item.get("question", ""),
            "gold": gold,
            "solution": item.get("solution", gold),
            "prediction": pred_text,
            "raw_prediction": raw_text,
            "agents": agents,
            "ttft_sec": ttft_sec,
            "cache_prepare_sec": cache_prepare_sec,
            "correct": ok,
        }

class BaselineToolCallingMethod(_ToolCallingMethodBase):
    method_name = "baseline_toolcall"

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        results: List[Dict[str, Any]] = []
        system_prompt = self._system_prompt()
        for sample_idx, item in enumerate(items, start=1):
            user_prompt = build_baseline_user_prompt(
                question=str(item.get("question", "")),
                files=list(item.get("files", [])),
            )
            run = self._new_react_agent(
                verbose_prefix=f"[baseline][sample {sample_idx}] "
            ).run(system_prompt=system_prompt, user_prompt=user_prompt)
            pred = extract_answer_text(run.final_answer)
            agents = [
                {
                    "name": "SingleAgent",
                    "role": "singleagent",
                    "input": user_prompt,
                    "output": run.final_output,
                    "tool_calls": self._serialize_tool_calls(run.tool_calls),
                    "messages": self._serialize_messages(run.messages),
                    "step_prompts": self._serialize_text_list(run.step_prompts),
                }
            ]
            results.append(
                self._build_result(
                    item=item,
                    pred_text=pred,
                    raw_text=run.final_output,
                    agents=agents,
                )
            )
        return results

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_batch([item])[0]


class TextMASToolCallingMethod(_ToolCallingMethodBase):
    method_name = "text_mas_toolcall"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_num_parallel_agents = max(
            1,
            int(getattr(self.args, "parallel_kv_num_parallel_agents", 3) if self.args else 3),
        )
        self.agents = default_agents(self.default_num_parallel_agents)

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        results: List[Dict[str, Any]] = []
        system_prompt = self._system_prompt()

        for sample_idx, item in enumerate(items, start=1):
            context = ""
            traces: List[Dict[str, Any]] = []
            final_raw = ""
            final_answer = ""
            judger_ttft_sec: Optional[float] = None
            for agent in self.agents:
                prompt = build_textmas_role_prompt(
                    role=agent.role,
                    question=str(item.get("question", "")),
                    context=context,
                    files=list(item.get("files", [])),
                )
                run = self._new_react_agent(
                    verbose_prefix=f"[text_mas][sample {sample_idx}][{agent.role}] "
                ).run(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    measure_first_step_ttft=(agent.role == "judger"),
                )
                role_output_full = run.final_output.strip()
                role_answer = run.final_answer.strip()
                if not role_output_full:
                    role_output_full = role_answer

                traces.append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompt,
                        "output": role_output_full,
                        "final_answer": role_answer,
                        "tool_calls": self._serialize_tool_calls(run.tool_calls),
                        "messages": self._serialize_messages(run.messages),
                        "step_prompts": self._serialize_text_list(run.step_prompts),
                    }
                )

                if agent.role != "judger":
                    context += f"[{agent.name}]:\n{role_output_full}\n\n"
                else:
                    final_raw = run.final_output
                    final_answer = run.final_answer
                    judger_ttft_sec = run.first_step_ttft_sec

            pred = extract_answer_text(final_answer or final_raw)
            results.append(
                self._build_result(
                    item=item,
                    pred_text=pred,
                    raw_text=final_raw,
                    agents=traces,
                    ttft_sec=judger_ttft_sec,
                )
            )

        return results

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_batch([item])[0]


class ParallelKVToolCallingMethod(ParallelKV):
    """
    Tool-calling ParallelKV variant.

    Only cache extraction differs from the shared ParallelKV path:
    tool-enabled `sub_agent_i` workers run through the ReAct loop, and we keep
    each worker's final-turn generation cache. Rope shifting, optional mapper
    loading, cache concatenation, judger position offset, and final decoding all
    come from the shared ParallelKV implementation.
    """

    method_name = "parallel_kv_toolcall"

    def __init__(
        self,
        model: ModelWrapper,
        *,
        judger_max_new_tokens: int,
        temperature: float,
        top_p: float,
        generate_bs: int,
        max_tool_steps: int,
        tools: ToolRegistry,
        args: argparse.Namespace,
    ) -> None:
        super().__init__(
            model,
            judger_max_new_tokens=judger_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generate_bs=generate_bs,
            args=args,
        )
        self.max_tool_steps = max(1, int(max_tool_steps))
        self.tools = tools
        self.args = args
        self.print_intermediate = bool(getattr(args, "print_intermediate", False))
        self.print_intermediate_max_chars = int(
            getattr(args, "print_intermediate_max_chars", 1200)
        )

    def _new_react_agent(self, *, verbose_prefix: str = "") -> ToolCallingReactAgent:
        return ToolCallingReactAgent(
            self.model,
            self.tools,
            max_steps=self.max_tool_steps,
            max_new_tokens=self.judger_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=self.print_intermediate,
            verbose_max_chars=self.print_intermediate_max_chars,
            verbose_prefix=verbose_prefix,
        )

    def _system_prompt(self) -> str:
        return build_tool_system_prompt(self.tools.schemas())

    def _score_final_text(self, item: Dict[str, Any], raw_final_text: str):
        pred = extract_answer_text(raw_final_text)
        gold = str(item.get("gold", ""))
        ok = free_form_exact_match(pred, gold)
        return pred, gold, ok

    def _collect_parallel_agent_past(
        self,
        items: List[Dict[str, Any]],
        *,
        include_traces: bool = False,
    ):
        self._ensure_hf_backend()
        if len(items) != 1:
            raise ValueError(
                "Tool-calling parallel_kv collects final-turn agent caches per sample. "
                "run_batch should route this method through singleton items."
            )

        item = items[0]
        traces: Optional[List[List[Dict[str, Any]]]] = [[]] if include_traces else None
        parallel_agent_past: List[Dict[str, Any]] = []
        model_obj = self.model.model
        was_training = bool(model_obj.training)
        system_prompt = self._system_prompt()

        with self._disable_lora_for_non_judger():
            model_obj.eval()
            try:
                context = ""
                for agent in self._agents_for_items(items):
                    if agent.role == "judger":
                        continue
                    prompt = build_textmas_role_prompt(
                        role=agent.role,
                        question=str(item.get("question", "")),
                        context=context,
                        files=list(item.get("files", [])),
                    )
                    run = self._new_react_agent(
                        verbose_prefix=f"[parallel_kv][{agent.role}] "
                    ).run(system_prompt=system_prompt, user_prompt=prompt)
                    text_out = run.final_output.strip()
                    text_answer = run.final_answer.strip()
                    if not text_out:
                        text_out = text_answer

                    if traces is not None:
                        traces[0].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": prompt,
                                "output": text_out,
                                "final_answer": text_answer,
                                "tool_calls": _ToolCallingMethodBase._serialize_tool_calls(run.tool_calls),
                                "messages": _ToolCallingMethodBase._serialize_messages(run.messages),
                                "step_prompts": _ToolCallingMethodBase._serialize_text_list(run.step_prompts),
                            }
                        )

                    generation_meta = run.last_generation or {}
                    generated_len = int(generation_meta.get("generated_len", 0) or 0)
                    prompt_lens = generation_meta.get("prompt_lens")
                    past_key_values = generation_meta.get("past_key_values")
                    if (
                        generated_len > 0
                        and prompt_lens is not None
                        and past_key_values is not None
                    ):
                        agent_output_past = self._truncate_past(past_key_values, generated_len)
                        agent_output_past = self._ensure_cache_object(agent_output_past)
                        if agent_output_past is not None and _past_length(agent_output_past) > 0:
                            parallel_agent_past.append(
                                {
                                    "past": agent_output_past,
                                    "role": agent.role,
                                    "prompt_lens": prompt_lens.to(self.model.device),
                                }
                            )

                    context += f"[{agent.name}]:\n{text_out}\n\n"
            finally:
                if was_training:
                    model_obj.train()
                else:
                    model_obj.eval()

        return parallel_agent_past, traces

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        results: List[Dict[str, Any]] = []
        system_prompt = self._system_prompt()

        for sample_idx, item in enumerate(items, start=1):
            # 1. Collect parallel agent caches
            parallel_agent_past, traces = self._collect_parallel_agent_past(
                [item], include_traces=True
            )
            if traces is None:
                traces = [[]]

            # 2. Prepare merged worker cache as a fixed latent prefix for every judger turn
            dummy_mask = torch.ones((1, 1), dtype=torch.long, device=self.model.device)
            past_for_decoding, position_offset, cache_prepare_sec = self._prepare_parallel_judger_past_with_timing(
                parallel_agent_past, judger_mask=dummy_mask
            )

            # 3. Run judger through the ReAct loop. Each turn rebuilds the
            #    current text chat state, while the parallel-agent cache stays
            #    attached as a fixed latent prefix.
            user_prompt = build_parallel_kv_judger_prompt(
                str(item.get("question", "")),
                list(item.get("files", [])),
            )
            run = self._new_react_agent(
                verbose_prefix=f"[parallel_kv][sample {sample_idx}][judger] "
            ).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                persistent_prefix_past_key_values=past_for_decoding,
                persistent_prefix_position_offset=position_offset,
                measure_first_step_ttft=True,
            )
            total_ttft_sec = self._add_prepare_time_to_ttft(
                run.first_step_ttft_sec,
                cache_prepare_sec,
            )

            # 4. Build result
            pred = extract_answer_text(run.final_answer)
            gold = str(item.get("gold", ""))
            ok = free_form_exact_match(pred, gold)

            traces[0].append(
                {
                    "name": "Judger",
                    "role": "judger",
                    "input": user_prompt,
                    "output": run.final_output,
                    "final_answer": run.final_answer,
                    "tool_calls": _ToolCallingMethodBase._serialize_tool_calls(run.tool_calls),
                    "messages": _ToolCallingMethodBase._serialize_messages(run.messages),
                    "step_prompts": _ToolCallingMethodBase._serialize_text_list(run.step_prompts),
                }
            )

            results.append(
                {
                    "question": item.get("question", ""),
                    "gold": gold,
                    "solution": item.get("solution", gold),
                    "prediction": pred,
                    "raw_prediction": run.final_output,
                    "agents": traces[0],
                    "ttft_sec": total_ttft_sec,
                    "cache_prepare_sec": cache_prepare_sec,
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_batch([item])[0]
