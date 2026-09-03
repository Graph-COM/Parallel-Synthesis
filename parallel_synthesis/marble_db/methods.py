import argparse
import json
from typing import Any, Dict, List, Tuple

import torch

from parallel_synthesis.toolcall.methods import _ToolCallingMethodBase, ParallelKVToolCallingMethod
from parallel_synthesis.toolcall.tools import ToolRegistry

from . import prompts
from .scoring import score_marble_db_prediction
from .tools import build_marble_db_tool_registry


class PlannerResult:
    def __init__(
        self,
        assignments: Dict[str, str],
        raw_output: str,
        source: str,
        messages: List[Dict[str, str]],
        prompt_text: str,
    ) -> None:
        self.assignments = assignments
        self.raw_output = raw_output
        self.source = source
        self.messages = messages
        self.prompt_text = prompt_text


def _json_parse(input_str: str) -> Dict[str, Any]:
    payload = str(input_str or "")
    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = payload[start : end + 1]
    return json.loads(payload)


class _MarbleDBShared:
    session_factory: Any

    def _set_session_tools(self, session: Any) -> None:
        self.tools = build_marble_db_tool_registry(session)

    def _system_prompt(self) -> str:
        return prompts.build_db_tool_system_prompt(self.tools.schemas())

    def _worker_system_prompt(self, agent_meta: Dict[str, Any]) -> str:
        return prompts.build_worker_system_prompt(
            agent_meta=agent_meta,
            tool_schemas=self.tools.schemas(),
        )

    @staticmethod
    def _judger_system_prompt() -> str:
        return ""

    def _generate_plain_text(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, str]:
        prompt_text, input_ids, attention_mask, _ = self.model.prepare_chat_input(
            messages,
            add_generation_prompt=True,
        )
        if self.model.use_vllm:
            text = self.model.vllm_generate_text_batch(
                [prompt_text],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )[0]
            return text.strip(), prompt_text
        generated, _ = self.model.generate_text_batch(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return generated[0].strip(), prompt_text

    @staticmethod
    def _fallback_assignments(item: Dict[str, Any]) -> Dict[str, str]:
        assignments: Dict[str, str] = {}
        for agent in item.get("agents_meta", []) or []:
            agent_id = str(agent.get("agent_id", "")).strip()
            profile = str(agent.get("profile", "")).strip()
            if agent_id:
                assignments[agent_id] = profile
        return assignments

    def _normalize_assignments(
        self,
        item: Dict[str, Any],
        tasks: Dict[str, Any],
    ) -> Dict[str, str]:
        normalized = self._fallback_assignments(item)
        for key, value in (tasks or {}).items():
            agent_id = str(key).strip()
            if not agent_id or agent_id not in normalized:
                continue
            normalized[agent_id] = str(value).strip() or normalized[agent_id]
        return normalized

    def _fixed_profile_plan(self, item: Dict[str, Any]) -> PlannerResult:
        assignments = self._fallback_assignments(item)
        raw_output = json.dumps(
            {"tasks": assignments, "continue": True},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return PlannerResult(
            assignments=assignments,
            raw_output=raw_output,
            source="fixed_profile",
            messages=[],
            prompt_text="",
        )

    def _plan_item(self, item: Dict[str, Any]) -> PlannerResult:
        messages = prompts.build_planner_messages(
            question=str(item.get("question", "")),
            current_progress=str(item.get("planner_initial_progress", "Starting the simulation.")),
            agents_meta=list(item.get("agents_meta", []) or []),
        )
        raw_output, prompt_text = self._generate_plain_text(
            messages,
            max_new_tokens=int(getattr(self.args, "planner_max_new_tokens", 1024)),
            temperature=float(getattr(self.args, "planner_temperature", 0.7)),
            top_p=float(getattr(self.args, "planner_top_p", 1.0)),
        )
        try:
            parsed = _json_parse(raw_output)
            tasks = parsed.get("tasks", {})
            assignments = self._normalize_assignments(item, tasks if isinstance(tasks, dict) else {})
            source = "planner_model"
        except Exception:
            assignments = self._fallback_assignments(item)
            source = "fallback_profile"
        return PlannerResult(
            assignments=assignments,
            raw_output=raw_output,
            source=source,
            messages=[
                {"role": msg["role"], "content": str(msg["content"])}
                for msg in messages
            ] + [{"role": "assistant", "content": raw_output}],
            prompt_text=prompt_text,
        )

    @staticmethod
    def _planner_trace(planner: PlannerResult) -> Dict[str, Any]:
        if planner.source == "fixed_profile":
            return {
                "name": "Planner",
                "role": "planner",
                "input": "[fixed agent profiles from dataset]",
                "output": planner.raw_output,
                "final_answer": planner.raw_output,
                "tool_calls": [],
                "messages": [],
                "step_prompts": [],
                "planning_source": planner.source,
                "task_assignments": planner.assignments,
            }
        return {
            "name": "Planner",
            "role": "planner",
            "input": planner.prompt_text,
            "output": planner.raw_output,
            "final_answer": planner.raw_output,
            "tool_calls": [],
            "messages": planner.messages,
            "step_prompts": [planner.prompt_text],
            "planning_source": planner.source,
            "task_assignments": planner.assignments,
        }

    def _serialize_worker_trace(
        self,
        *,
        agent_id: str,
        prompt: str,
        run: Any,
    ) -> Dict[str, Any]:
        text_out = str(run.final_output or "").strip()
        text_answer = str(run.final_answer or "").strip()
        if not text_out:
            text_out = text_answer
        return {
            "name": agent_id,
            "role": agent_id,
            "input": prompt,
            "output": text_out,
            "final_answer": text_answer,
            "tool_calls": _ToolCallingMethodBase._serialize_tool_calls(run.tool_calls),
            "messages": _ToolCallingMethodBase._serialize_messages(run.messages),
            "step_prompts": _ToolCallingMethodBase._serialize_text_list(run.step_prompts),
        }

    def _build_result(
        self,
        *,
        item: Dict[str, Any],
        planner: PlannerResult,
        traces: List[Dict[str, Any]],
        raw_prediction: str,
        ttft_sec: Any = None,
        cache_prepare_sec: Any = None,
    ) -> Dict[str, Any]:
        scored = score_marble_db_prediction(raw_prediction, item)
        return {
            "benchmark": "marble_db",
            "task_id": item.get("task_id", 0),
            "question": item.get("question", ""),
            "gold": list(scored["gold_root_causes"]),
            "solution": list(scored["gold_root_causes"]),
            "prediction": list(scored["predicted_root_causes"]),
            "predicted_root_causes": list(scored["predicted_root_causes"]),
            "gold_root_causes": list(scored["gold_root_causes"]),
            "raw_prediction": raw_prediction,
            "task_score": float(scored["task_score"]),
            "hit_count": int(scored["hit_count"]),
            "correct": bool(scored["correct"]),
            "task_assignments": dict(planner.assignments),
            "planner_source": planner.source,
            "agents": traces,
            "ttft_sec": ttft_sec,
            "cache_prepare_sec": cache_prepare_sec,
        }


class TextMASMarbleDBMethod(_MarbleDBShared, _ToolCallingMethodBase):
    method_name = "text_mas_marble_db"

    def __init__(
        self,
        model,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        generate_bs: int,
        max_tool_steps: int,
        args: argparse.Namespace,
        session_factory: Any,
    ) -> None:
        super().__init__(
            model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generate_bs=generate_bs,
            max_tool_steps=max_tool_steps,
            tools=ToolRegistry([]),
            args=args,
        )
        self.session_factory = session_factory

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        results: List[Dict[str, Any]] = []
        for sample_idx, item in enumerate(items, start=1):
            planner = self._fixed_profile_plan(item)
            traces: List[Dict[str, Any]] = [self._planner_trace(planner)]
            with self.session_factory(item) as session:
                self._set_session_tools(session)
                judger_system_prompt = self._judger_system_prompt()
                worker_results: List[Dict[str, Any]] = []
                prompt_module = prompts
                for agent_meta in item.get("agents_meta", []) or []:
                    agent_id = str(agent_meta.get("agent_id", "")).strip()
                    worker_system_prompt = self._worker_system_prompt(agent_meta)
                    prompt = prompt_module.build_worker_user_prompt(
                        agent_meta=agent_meta,
                        assigned_task=str(planner.assignments.get(agent_id, agent_meta.get("profile", ""))),
                        all_agents_meta=list(item.get("agents_meta", []) or []),
                    )
                    run = self._new_react_agent(
                        verbose_prefix=f"[marble_db][text_mas][sample {sample_idx}][{agent_id}] "
                    ).run(system_prompt=worker_system_prompt, user_prompt=prompt)
                    trace = self._serialize_worker_trace(
                        agent_id=agent_id,
                        prompt=prompt,
                        run=run,
                    )
                    traces.append(trace)
                    worker_results.append({agent_id: trace["output"]})

                summary_text = prompt_module.render_agents_results_summary(worker_results)
                judger_prompt = prompt_module.build_textmas_judger_prompt(
                    question=str(item.get("question", "")),
                    summary_text=summary_text,
                    labels=[
                        str(x).strip()
                        for x in (item.get("labels", []) or [])
                        if str(x).strip()
                    ],
                    max_root_causes=int(item.get("number_of_labels_pred", 2) or 2),
                )
                judger_run = self._new_react_agent(
                    verbose_prefix=f"[marble_db][text_mas][sample {sample_idx}][judger] "
                ).run(
                    system_prompt=judger_system_prompt,
                    user_prompt=judger_prompt,
                    measure_first_step_ttft=True,
                )
                traces.append(
                    {
                        "name": "Judger",
                        "role": "judger",
                        "input": judger_prompt,
                        "output": judger_run.final_output,
                        "final_answer": judger_run.final_answer,
                        "tool_calls": _ToolCallingMethodBase._serialize_tool_calls(judger_run.tool_calls),
                        "messages": _ToolCallingMethodBase._serialize_messages(judger_run.messages),
                        "step_prompts": _ToolCallingMethodBase._serialize_text_list(judger_run.step_prompts),
                    }
                )
                results.append(
                    self._build_result(
                        item=item,
                        planner=planner,
                        traces=traces,
                        raw_prediction=str(judger_run.final_output),
                        ttft_sec=judger_run.first_step_ttft_sec,
                    )
                )
        return results

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_batch([item])[0]


class ParallelKVMarbleDBMethod(_MarbleDBShared, ParallelKVToolCallingMethod):
    method_name = "parallel_kv_marble_db"

    def __init__(
        self,
        model,
        *,
        judger_max_new_tokens: int,
        temperature: float,
        top_p: float,
        generate_bs: int,
        max_tool_steps: int,
        args: argparse.Namespace,
        session_factory: Any,
    ) -> None:
        super().__init__(
            model,
            judger_max_new_tokens=judger_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generate_bs=generate_bs,
            max_tool_steps=max_tool_steps,
            tools=ToolRegistry([]),
            args=args,
        )
        self.session_factory = session_factory

    def _collect_worker_past_with_traces(
        self,
        *,
        item: Dict[str, Any],
        planner: PlannerResult,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        self._ensure_hf_backend()
        parallel_agent_past: List[Dict[str, Any]] = []
        worker_traces: List[Dict[str, Any]] = []
        worker_results: List[Dict[str, Any]] = []

        model_obj = self.model.model
        was_training = bool(model_obj.training)
        with self._disable_lora_for_non_judger():
            model_obj.eval()
            try:
                prompt_module = prompts
                for agent_meta in item.get("agents_meta", []) or []:
                    agent_id = str(agent_meta.get("agent_id", "")).strip()
                    worker_system_prompt = self._worker_system_prompt(agent_meta)
                    prompt = prompt_module.build_worker_user_prompt(
                        agent_meta=agent_meta,
                        assigned_task=str(planner.assignments.get(agent_id, agent_meta.get("profile", ""))),
                        all_agents_meta=list(item.get("agents_meta", []) or []),
                    )
                    run = self._new_react_agent(
                        verbose_prefix=f"[marble_db][parallel_kv][{agent_id}] "
                    ).run(system_prompt=worker_system_prompt, user_prompt=prompt)
                    trace = self._serialize_worker_trace(
                        agent_id=agent_id,
                        prompt=prompt,
                        run=run,
                    )
                    worker_traces.append(trace)
                    worker_results.append({agent_id: trace["output"]})

                    generation_meta = run.last_generation or {}
                    generated_len = int(generation_meta.get("generated_len", 0) or 0)
                    prompt_lens = generation_meta.get("prompt_lens")
                    past_key_values = generation_meta.get("past_key_values")
                    if (
                        generated_len > 0
                        and prompt_lens is not None
                        and past_key_values is not None
                    ):
                        agent_output_past = self._truncate_past(
                            past_key_values,
                            generated_len,
                        )
                        agent_output_past = self._ensure_cache_object(agent_output_past)
                        if agent_output_past is not None and self.model is not None:
                            parallel_agent_past.append(
                                {
                                    "past": agent_output_past,
                                    "role": agent_id,
                                    "prompt_lens": prompt_lens.to(self.model.device),
                                }
                            )
            finally:
                if was_training:
                    model_obj.train()
                else:
                    model_obj.eval()
        return parallel_agent_past, worker_traces, worker_results

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        results: List[Dict[str, Any]] = []
        for sample_idx, item in enumerate(items, start=1):
            planner = self._fixed_profile_plan(item)
            traces: List[Dict[str, Any]] = [self._planner_trace(planner)]
            with self.session_factory(item) as session:
                self._set_session_tools(session)
                judger_system_prompt = self._judger_system_prompt()
                parallel_agent_past, worker_traces, _ = self._collect_worker_past_with_traces(
                    item=item,
                    planner=planner,
                )
                prompt_module = prompts
                traces.extend(worker_traces)

                dummy_mask = torch.ones((1, 1), dtype=torch.long, device=self.model.device)
                past_for_decoding, position_offset, cache_prepare_sec = (
                    self._prepare_parallel_judger_past_with_timing(
                        parallel_agent_past,
                        judger_mask=dummy_mask,
                    )
                )
                judger_prompt = prompt_module.build_parallel_kv_judger_prompt(
                    question=str(item.get("question", "")),
                    labels=[
                        str(x).strip()
                        for x in (item.get("labels", []) or [])
                        if str(x).strip()
                    ],
                    max_root_causes=int(item.get("number_of_labels_pred", 2) or 2),
                )
                judger_run = self._new_react_agent(
                    verbose_prefix=f"[marble_db][parallel_kv][sample {sample_idx}][judger] "
                ).run(
                    system_prompt=judger_system_prompt,
                    user_prompt=judger_prompt,
                    persistent_prefix_past_key_values=past_for_decoding,
                    persistent_prefix_position_offset=position_offset,
                    measure_first_step_ttft=True,
                )
                total_ttft_sec = self._add_prepare_time_to_ttft(
                    judger_run.first_step_ttft_sec,
                    cache_prepare_sec,
                )
                traces.append(
                    {
                        "name": "Judger",
                        "role": "judger",
                        "input": judger_prompt,
                        "output": judger_run.final_output,
                        "final_answer": judger_run.final_answer,
                        "tool_calls": _ToolCallingMethodBase._serialize_tool_calls(judger_run.tool_calls),
                        "messages": _ToolCallingMethodBase._serialize_messages(judger_run.messages),
                        "step_prompts": _ToolCallingMethodBase._serialize_text_list(judger_run.step_prompts),
                    }
                )
                results.append(
                    self._build_result(
                        item=item,
                        planner=planner,
                        traces=traces,
                        raw_prediction=str(judger_run.final_output),
                        ttft_sec=total_ttft_sec,
                        cache_prepare_sec=cache_prepare_sec,
                    )
                )
        return results

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_batch([item])[0]
