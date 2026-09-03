import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from parallel_synthesis.models import ModelWrapper

from .prompts import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
)
from .tools import ToolRegistry

CODE_OPEN = "<code>"
CODE_CLOSE = "</code>"


class ToolCallLog:
    def __init__(self, name: str, arguments: Dict[str, Any], result_preview: str) -> None:
        self.name = name
        self.arguments = arguments
        self.result_preview = result_preview


class AgentRunResult:
    def __init__(
        self,
        final_output: str,
        final_answer: str,
        messages: List[Dict[str, str]],
        tool_calls: Optional[List[ToolCallLog]] = None,
        step_prompts: Optional[List[str]] = None,
        last_generation: Optional[Dict[str, Any]] = None,
        first_step_ttft_sec: Optional[float] = None,
    ) -> None:
        self.final_output = final_output
        self.final_answer = final_answer
        self.messages = messages
        self.tool_calls = list(tool_calls or [])
        self.step_prompts = list(step_prompts or [])
        self.last_generation = last_generation
        self.first_step_ttft_sec = first_step_ttft_sec


class ToolCallingReactAgent:
    """DeepResearch-style ReAct loop implemented on top of ModelWrapper."""

    def __init__(
        self,
        model: ModelWrapper,
        tools: ToolRegistry,
        *,
        max_steps: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        verbose: bool = False,
        verbose_max_chars: int = 1200,
        verbose_prefix: str = "",
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = bool(verbose)
        self.verbose_max_chars = max(80, int(verbose_max_chars))
        self.verbose_prefix = str(verbose_prefix)

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(f"{self.verbose_prefix}{message}")

    def _vclip(self, text: str) -> str:
        s = str(text)
        if len(s) <= self.verbose_max_chars:
            return s
        return s[: self.verbose_max_chars] + "\n... [truncated]"

    def _generate_one(
        self,
        messages: List[Dict[str, str]],
        *,
        return_timings: bool = False,
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        prompt, input_ids, attention_mask, _ = self.model.prepare_chat_input(
            messages, add_generation_prompt=True
        )
        if self.model.use_vllm:
            generated = self.model.vllm_generate_text_batch(
                [prompt],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )[0]
            return generated, prompt, None
        if return_timings:
            generated_batch, past_key_values, generated_ids_list, timing_metrics = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                return_generated_ids=True,
                return_timings=True,
            )
        else:
            generated_batch, past_key_values, generated_ids_list = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                return_generated_ids=True,
            )
            timing_metrics = None
        generated_len = 0
        if generated_ids_list:
            generated_len = int(generated_ids_list[0].shape[0])
        generation_meta = {
            "past_key_values": past_key_values,
            "prompt_lens": attention_mask.sum(dim=1).detach().clone(),
            "generated_len": generated_len,
            "ttft_sec": None if timing_metrics is None else timing_metrics.get("ttft_sec"),
        }
        return generated_batch[0], prompt, generation_meta

    def _generate_one_with_cache(
        self,
        messages: List[Dict[str, str]],
        past_key_values: Any,
        position_offset: Any,
        *,
        return_timings: bool = False,
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """Generation with a fixed KV prefix prepended to the current chat prompt."""
        prompt, input_ids, attention_mask, _ = self.model.prepare_chat_input(
            messages, add_generation_prompt=True
        )
        if return_timings:
            generated_batch, _, timing_metrics = self.model.generate_text_batch_with_offset(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                past_key_values=past_key_values,
                position_offset=position_offset,
                return_timings=True,
            )
        else:
            generated_batch, _ = self.model.generate_text_batch_with_offset(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                past_key_values=past_key_values,
                position_offset=position_offset,
            )
            timing_metrics = None
        return generated_batch[0], prompt, {
            "ttft_sec": None if timing_metrics is None else timing_metrics.get("ttft_sec")
        }

    @staticmethod
    def _extract_answer(text: str) -> Optional[str]:
        m = re.search(
            re.escape(ANSWER_OPEN) + r"([\s\S]*?)" + re.escape(ANSWER_CLOSE),
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        return m.group(1).strip()

    @staticmethod
    def _parse_json_relaxed(text: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        try:
            obj = ast.literal_eval(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        recovered = ToolCallingReactAgent._recover_malformed_tool_call(text)
        if recovered is not None:
            return recovered

        return None

    @staticmethod
    def _parse_qwen_xml_tool_call(text: str) -> Optional[Dict[str, Any]]:
        payload = str(text).strip()

        fn_match = re.search(
            r"<function=([^>\n]+)>\s*([\s\S]*?)\s*</function>",
            payload,
            flags=re.IGNORECASE,
        )
        if not fn_match:
            fn_match = re.search(
                r"<function=([^>\n]+)>\s*([\s\S]*)",
                payload,
                flags=re.IGNORECASE,
            )
        if not fn_match:
            return None

        name = fn_match.group(1).strip()
        body = fn_match.group(2)
        if not name:
            return None

        arguments: Dict[str, Any] = {}
        for pm in re.finditer(
            r"<parameter=([^>\n]+)>\s*([\s\S]*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            body,
            flags=re.IGNORECASE,
        ):
            key = pm.group(1).strip()
            value_raw = pm.group(2).strip()
            if not key:
                continue

            value: Any = value_raw
            # Allow non-string arguments emitted as JSON literals by template.
            if value_raw:
                try:
                    value = json.loads(value_raw)
                except Exception:
                    value = value_raw
            arguments[key] = value

        return {"name": name, "arguments": arguments}

    @staticmethod
    def _extract_relaxed_string_field(payload: str, field_name: str) -> Optional[str]:
        m = re.search(rf'["\']{re.escape(field_name)}["\']\s*:\s*', payload)
        if not m:
            return None
        i = m.end()
        while i < len(payload) and payload[i].isspace():
            i += 1
        if i >= len(payload) or payload[i] not in {"'", '"'}:
            return None
        quote = payload[i]
        start = i + 1

        # Accept malformed strings that may contain unescaped quotes.
        # We keep scanning until a quote whose trailing text looks like
        # a field/value boundary or object closure.
        j = start
        escaped = False
        while j < len(payload):
            ch = payload[j]
            if escaped:
                escaped = False
                j += 1
                continue
            if ch == "\\":
                escaped = True
                j += 1
                continue
            if ch == quote:
                tail = payload[j + 1 :]
                if re.match(
                    r'^\s*(?:,\s*["\'][^"\']+["\']\s*:.*)?\s*\}\s*\}\s*$',
                    tail,
                    flags=re.DOTALL,
                ):
                    return payload[start:j]
            j += 1

        # Fallback: use the last quote before the final closing braces.
        close = re.search(r"\}\s*\}\s*$", payload)
        end_bound = close.start() if close else len(payload)
        k = payload.rfind(quote, start, end_bound)
        if k > start:
            return payload[start:k]
        return None

    @staticmethod
    def _recover_malformed_tool_call(text: str) -> Optional[Dict[str, Any]]:
        payload = str(text).strip()
        name_match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', payload)
        if not name_match:
            return None
        name = name_match.group(1).strip()
        if not name:
            return None

        arguments: Dict[str, Any] = {}
        command = ToolCallingReactAgent._extract_relaxed_string_field(payload, "command")
        if command is not None:
            arguments["command"] = command
        elif re.search(r'["\']arguments["\']\s*:\s*\{', payload):
            # If arguments object exists but cannot be parsed, still return a
            # best-effort empty arguments payload so the caller can decide.
            arguments = {}
        else:
            # No recognizable arguments payload.
            return None

        return {"name": name, "arguments": arguments}

    def _extract_tool_calls(self, text: str) -> List[Tuple[str, Dict[str, Any], Optional[str]]]:
        calls: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
        for m in re.finditer(
            re.escape(TOOL_CALL_OPEN) + r"([\s\S]*?)" + re.escape(TOOL_CALL_CLOSE),
            text,
            flags=re.IGNORECASE,
        ):
            payload = m.group(1).strip()
            code = None
            cm = re.search(
                re.escape(CODE_OPEN) + r"([\s\S]*?)" + re.escape(CODE_CLOSE),
                payload,
                flags=re.IGNORECASE,
            )
            if cm:
                code = cm.group(1).strip()
                payload = (payload[: cm.start()] + payload[cm.end() :]).strip()

            req = self._parse_json_relaxed(payload)
            if req is None:
                req = self._parse_qwen_xml_tool_call(payload)
            if req is None:
                continue

            name = str(req.get("name", "")).strip()
            arguments = req.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}

            if code:
                arguments = dict(arguments)
                arguments["code"] = code

            if not name:
                continue
            calls.append((name, arguments, code))
        return calls

    @staticmethod
    def _trim_assistant_content(content: str) -> str:
        content = str(content)
        pos = content.find(TOOL_RESPONSE_OPEN)
        if pos >= 0:
            return content[:pos].strip()
        return content.strip()

    @staticmethod
    def _preview(text: str, n: int = 300) -> str:
        t = str(text).strip()
        if len(t) <= n:
            return t
        return t[:n] + "... [truncated]"

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        persistent_prefix_past_key_values: Any = None,
        persistent_prefix_position_offset: Any = None,
        measure_first_step_ttft: bool = False,
    ) -> AgentRunResult:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_calls: List[ToolCallLog] = []
        step_prompts: List[str] = []
        last_generation: Optional[Dict[str, Any]] = None
        first_step_ttft_sec: Optional[float] = None

        self._vprint(f"start_react max_steps={self.max_steps}")
        self._vprint("user_prompt:\n" + self._vclip(user_prompt))

        for step_idx in range(1, max(1, self.max_steps) + 1):
            self._vprint(f"step={step_idx}")
            if persistent_prefix_past_key_values is not None:
                raw, prompt_text, generation_meta = self._generate_one_with_cache(
                    messages,
                    persistent_prefix_past_key_values,
                    persistent_prefix_position_offset,
                    return_timings=bool(measure_first_step_ttft and step_idx == 1),
                )
            else:
                raw, prompt_text, generation_meta = self._generate_one(
                    messages,
                    return_timings=bool(measure_first_step_ttft and step_idx == 1),
                )
            raw = raw.strip()
            last_generation = generation_meta
            if step_idx == 1 and generation_meta is not None:
                ttft = generation_meta.get("ttft_sec")
                if ttft is not None:
                    first_step_ttft_sec = float(ttft)
            step_prompts.append(str(prompt_text))
            self._vprint("exact_prompt:\n" + str(prompt_text))
            assistant_content = self._trim_assistant_content(raw)
            messages.append({"role": "assistant", "content": assistant_content})
            self._vprint("assistant_output:\n" + self._vclip(assistant_content))

            reqs = self._extract_tool_calls(assistant_content)
            if reqs:
                self._vprint(f"tool_calls_in_turn={len(reqs)}")
                for call_idx, (tool_name, tool_args, _) in enumerate(reqs, start=1):
                    self._vprint(
                        f"tool_call[{call_idx}]: "
                        + self._vclip(json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False))
                    )
                    try:
                        tool_result = self.tools.call(tool_name, tool_args)
                    except Exception as exc:  # defensive: avoid breaking loop on tool errors
                        tool_result = f"Tool execution failed: {exc}"
                    self._vprint(f"tool_response[{call_idx}]:\n" + self._vclip(tool_result))

                    tool_calls.append(
                        ToolCallLog(
                            name=tool_name,
                            arguments=tool_args,
                            result_preview=self._preview(tool_result),
                        )
                    )
                    obs = f"{TOOL_RESPONSE_OPEN}\n{tool_result}\n{TOOL_RESPONSE_CLOSE}"
                    messages.append({"role": "user", "content": obs})
                continue

            ans = self._extract_answer(assistant_content)
            if ans is not None:
                self._vprint("final_answer:\n" + self._vclip(ans))
                return AgentRunResult(
                    final_output=assistant_content,
                    final_answer=ans,
                    messages=messages,
                    tool_calls=tool_calls,
                    step_prompts=step_prompts,
                    last_generation=last_generation,
                    first_step_ttft_sec=first_step_ttft_sec,
                )

            self._vprint("no_tool_call_or_answer -> stop")
            return AgentRunResult(
                final_output=assistant_content,
                final_answer=assistant_content,
                messages=messages,
                tool_calls=tool_calls,
                step_prompts=step_prompts,
                last_generation=last_generation,
                first_step_ttft_sec=first_step_ttft_sec,
            )

        last = messages[-1]["content"] if messages else ""
        ans = self._extract_answer(last) or last
        self._vprint("max_steps_reached final_answer:\n" + self._vclip(ans))
        return AgentRunResult(
            final_output=last,
            final_answer=ans,
            messages=messages,
            tool_calls=tool_calls,
            step_prompts=step_prompts,
            last_generation=last_generation,
            first_step_ttft_sec=first_step_ttft_sec,
        )
