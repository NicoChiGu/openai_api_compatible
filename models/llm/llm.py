import json
import re
from contextlib import suppress
from typing import Mapping, Optional, Union, Generator, List
from urllib.parse import urljoin

import requests
import httpx
from openai import OpenAI

from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    I18nObject,
    ModelFeature,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
    AssistantPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel

class OpenAILargeLanguageModel(OAICompatLargeLanguageModel):
    # 预编译正则提高性能
    _THINK_PATTERN = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)
    _NEEDS_MAX_COMPLETION_TOKENS_PATTERN = re.compile(r"^(o1|o3|gpt-5|gpt-4o-6-reasoning)", re.IGNORECASE)

    def _get_timeout(self, credentials: dict) -> tuple[int, int]:
        """安全解析超时配置"""
        def safe_int(key, default):
            val = credentials.get(key)
            if val is None or str(val).strip() == "":
                return default
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return default
        return (safe_int("connect_timeout", 10), safe_int("read_timeout", 300))

    def _build_openai_client(self, credentials: dict) -> OpenAI:
        """构建带自定义超时的客户端"""
        timeout_config = self._get_timeout(credentials)
        return OpenAI(
            api_key=credentials.get("api_key"),
            base_url=credentials.get("endpoint_url"),
            default_headers=credentials.get("extra_headers") or {},
            http_client=httpx.Client(
                timeout=httpx.Timeout(
                    connect=timeout_config[0],
                    read=timeout_config[1],
                    write=timeout_config[1],
                    pool=timeout_config[0],
                )
            )
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        endpoint_model = credentials.get("endpoint_model_name") or model
        param_pref = credentials.get("token_param_name", "auto")
        use_max_completion = (
            param_pref == "max_completion_tokens"
            or (param_pref == "auto" and bool(self._NEEDS_MAX_COMPLETION_TOKENS_PATTERN.match(endpoint_model)))
        )

        try:
            if use_max_completion:
                self._retry_with_safe_min_tokens(model, credentials)
            else:
                super().validate_credentials(model, credentials)
        except CredentialsValidateFailedError as e:
            msg = str(e)
            if any(x in msg for x in ["max_output_tokens", "integer_below_min_value"]):
                self._retry_with_safe_min_tokens(model, credentials)
            elif any(x in msg for x in ["budget_tokens", "thinking"]):
                self._retry_with_thinking_disabled(model, credentials)
            else:
                raise

    def _retry_with_safe_min_tokens(self, model: str, credentials: dict) -> None:
        client = self._build_openai_client(credentials)
        endpoint_model = credentials.get("endpoint_model_name") or model
        mode = credentials.get("mode", "chat")
        
        # 再次确认参数名
        param_pref = credentials.get("token_param_name", "auto")
        use_max_completion = (
            param_pref == "max_completion_tokens" 
            or (param_pref == "auto" and bool(self._NEEDS_MAX_COMPLETION_TOKENS_PATTERN.match(endpoint_model)))
        )

        try:
            if mode == "chat":
                kwargs = {"model": endpoint_model, "messages": [{"role": "user", "content": "ping"}], "stream": False}
                if use_max_completion:
                    kwargs["max_completion_tokens"] = 16
                else:
                    kwargs["max_tokens"] = 16
                client.chat.completions.create(**kwargs)
            else:
                client.completions.create(model=endpoint_model, prompt="ping", max_tokens=16, stream=False)
        except Exception as e:
            raise CredentialsValidateFailedError(f"API验证失败: {str(e)}")

    def _retry_with_thinking_disabled(self, model: str, credentials: dict) -> None:
        timeout_config = self._get_timeout(credentials)
        headers = {"Content-Type": "application/json"}
        if api_key := credentials.get("api_key"):
            headers["Authorization"] = f"Bearer {api_key}"
        base_url = credentials["endpoint_url"].rstrip("/") + "/"
        validate_max_tokens = int(credentials.get("validate_credentials_max_tokens") or 5)
        
        data = {
            "model": credentials.get("endpoint_model_name", model),
            "max_tokens": validate_max_tokens,
            "thinking": {"type": "disabled"},
        }
        mode = credentials.get("mode", "chat")
        endpoint = urljoin(base_url, "chat/completions" if mode == "chat" else "completions")
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=timeout_config)
            if response.status_code != 200:
                raise CredentialsValidateFailedError(f"HTTP {response.status_code}: {response.text}")
        except Exception as ex:
            raise CredentialsValidateFailedError(f"网络连接错误: {str(ex)}")

    def get_customizable_model_schema(self, model: str, credentials: Mapping | dict) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)
        new_params = [
            ParameterRule(
                name="connect_timeout",
                label=I18nObject(en_US="Connect Timeout", zh_Hans="连接超时"),
                help=I18nObject(en_US="Seconds", zh_Hans="连接建立超时时间"),
                type=ParameterType.NUMBER, required=False, default=10
            ),
            ParameterRule(
                name="read_timeout",
                label=I18nObject(en_US="Read Timeout", zh_Hans="读取超时"),
                help=I18nObject(en_US="Seconds", zh_Hans="模型响应超时时间"),
                type=ParameterType.NUMBER, required=False, default=300
            )
        ]
        existing_names = {p.name for p in entity.parameter_rules}
        for p in new_params:
            if p.name not in existing_names:
                entity.parameter_rules.append(p)
        return entity

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        
        # 1. 处理 JSON Schema 注入逻辑
        if model_parameters.get("response_format") == "json_schema":
            json_schema_str = model_parameters.get("json_schema")
            if json_schema_str:
                instruction = f"Your response must be a JSON object validating against: {json_schema_str}"
                existing_sys = next((p for p in prompt_messages if p.role == PromptMessageRole.SYSTEM), None)
                if existing_sys:
                    existing_sys.content = instruction + "\n\n" + existing_sys.content
                else:
                    prompt_messages.insert(0, SystemPromptMessage(content=instruction))

        # 2. 修正 o1 等模型的 token 参数
        param_pref = credentials.get("token_param_name", "auto")
        if (param_pref == "max_completion_tokens" or 
            (param_pref == "auto" and self._NEEDS_MAX_COMPLETION_TOKENS_PATTERN.match(model))):
            if "max_tokens" in model_parameters:
                model_parameters.setdefault("max_completion_tokens", model_parameters.pop("max_tokens"))

        # 3. 处理思考模式开关
        enable_thinking = model_parameters.get("enable_thinking", True)

        # 4. 调用基类
        result = super()._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

        # 5. 如果关闭了思考模式，执行后置过滤
        if enable_thinking is False:
            return self._filter_thinking_stream(result) if stream else self._filter_thinking_result(result)
        
        return result

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        """支持 reasoning_content 字段的流式包装"""
        reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content")
        content_piece = delta.get("content") or ""
        if reasoning_piece:
            if not is_reasoning:
                return f"<think>\n{reasoning_piece}", True
            return str(reasoning_piece), True
        elif is_reasoning:
            return f"\n</think>{content_piece}", False
        return content_piece, False

    def _filter_thinking_result(self, result: LLMResult) -> LLMResult:
        if result.message and result.message.content:
            result.message.content = self._THINK_PATTERN.sub("", result.message.content, count=1)
        return result

    def _filter_thinking_stream(self, stream: Generator) -> Generator:
        buffer, in_thinking, thinking_started = "", False, False
        for chunk in stream:
            if chunk.delta and chunk.delta.message and chunk.delta.message.content:
                content = chunk.delta.message.content
                buffer += content
                if not thinking_started and buffer.startswith("<think>"):
                    in_thinking = thinking_started = True
                if in_thinking and "</think>" in buffer:
                    end_idx = buffer.find("</think>") + len("</think>")
                    while end_idx < len(buffer) and buffer[end_idx].isspace(): end_idx += 1
                    buffer, in_thinking, thinking_started = buffer[end_idx:], False, False
                    if buffer:
                        chunk.delta.message.content = buffer
                        buffer = ""
                        yield chunk
                    continue
                if not in_thinking:
                    yield chunk
                    buffer = ""
            else:
                yield chunk
