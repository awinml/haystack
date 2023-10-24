import os
from unittest.mock import patch, Mock
from copy import deepcopy

import pytest
from anthropic.types.completion import Completion as AnthropicCompletion

from haystack.preview.components.generators.anthropic.claude import ClaudeGenerator
from haystack.preview.components.generators.anthropic.claude import default_streaming_callback


def mock_anthropic_response(prompt: str, model: str = "claude-instant-1") -> AnthropicCompletion:
    response = f"Generated response for this prompt: {prompt}"
    completion_response = AnthropicCompletion()
    completion_response.completion = response
    completion_response.stop_reason = "stop_sequence"
    completion_response.model = model

    return completion_response


def mock_anthropic_stream_response(prompt: str, model: str = "claude-instant-1") -> AnthropicCompletion:
    response = f"Generated response for this prompt: {prompt}"
    tokens = response.split()

    for token in tokens[:-1]:
        completion_response = AnthropicCompletion()
        completion_response.completion = token
        completion_response.stop_reason = None
        completion_response.model = model
        yield completion_response

    completion_response = AnthropicCompletion()
    completion_response.completion = tokens[-1]
    completion_response.stop_reason = "stop_sequence"
    completion_response.model = model
    yield completion_response


class TestClaudeGenerator:
    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ClaudeGenerator expects an ANTHROPIC API key"):
            ClaudeGenerator()

    @pytest.mark.unit
    def test_init_default_parameters(self):
        component = ClaudeGenerator(api_key="test-api-key")
        assert component.api_key == "test-api-key"
        assert component.model_name == "claude-instant-1"
        assert component.max_tokens_to_sample == 256
        assert component.use_async_client is False
        assert component.max_retries == 3
        assert component.timeout == 600
        assert component.streaming_callback is None
        assert component.api_base_url == "https://api.anthropic.com/v1/complete"
        assert component.model_parameters == {}

    @pytest.mark.unit
    def test_init_with_custom_parameters(self):
        callback = lambda x: x
        component = ClaudeGenerator(
            api_key="test-api-key",
            model_name="claude-instant-1",
            max_tokens_to_sample=10,
            use_async_client=True,
            max_retries=2,
            timeout=20,
            streaming_callback=callback,
            api_base_url="test-base-url",
            temperature=0.5,
            top_p=0.1,
            top_k=5,
            stop_sequences=["###", "---"],
        )
        assert component.api_key == "test-api-key"
        assert component.model_name == "claude-instant-1"
        assert component.max_tokens_to_sample == 10
        assert component.use_async_client == True
        assert component.max_retries == 2
        assert component.timeout == 20
        assert component.streaming_callback == callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {
            "temperature": 0.5,
            "top_p": 0.1,
            "top_k": 5,
            "stop_sequences": ["###", "---"],
        }

    @pytest.mark.unit
    def test_to_dict_default_parameters(self):
        component = ClaudeGenerator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "ClaudeGenerator",
            "init_parameters": {
                "model_name": "claude-instant-1",
                "max_tokens_to_sample": 256,
                "use_async_client": False,
                "max_retries": 3,
                "timeout": 600,
                "streaming_callback": None,
                "api_base_url": "https://api.anthropic.com/v1/complete",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_parameters(self):
        component = ClaudeGenerator(
            api_key="test-api-key",
            model_name="claude-instant-1",
            max_tokens_to_sample=10,
            use_async_client=True,
            max_retries=2,
            timeout=20,
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
            temperature=0.5,
            top_p=0.1,
            top_k=5,
            stop_sequences=["###", "---"],
        )
        data = component.to_dict()
        assert data == {
            "type": "ClaudeGenerator",
            "init_parameters": {
                "model_name": "claude-instant-1",
                "max_tokens_to_sample": 10,
                "use_async_client": True,
                "max_retries": 2,
                "timeout": 20,
                "streaming_callback": "haystack.preview.components.generators.anthropic.claude.default_streaming_callback",
                "api_base_url": "test-base-url",
                "temperature": 0.5,
                "top_p": 0.1,
                "top_k": 5,
                "stop_sequences": ["###", "---"],
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_lambda_streaming_callback(self):
        component = ClaudeGenerator(
            api_key="test-api-key",
            model_name="claude-instant-1",
            max_tokens_to_sample=10,
            use_async_client=True,
            max_retries=2,
            timeout=20,
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
            temperature=0.5,
            top_p=0.1,
            top_k=5,
            stop_sequences=["###", "---"],
        )
        data = component.to_dict()
        assert data == {
            "type": "ClaudeGenerator",
            "init_parameters": {
                "model_name": "claude-instant-1",
                "max_tokens_to_sample": 10,
                "use_async_client": True,
                "max_retries": 2,
                "timeout": 20,
                "streaming_callback": "test_claude_generator.<lambda>",
                "api_base_url": "test-base-url",
                "temperature": 0.5,
                "top_p": 0.1,
                "top_k": 5,
                "stop_sequences": ["###", "---"],
            },
        }

    @pytest.mark.unit
    def test_from_dict_default_parameters(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        data = {
            "type": "ClaudeGenerator",
            "init_parameters": {
                "model_name": "claude-instant-1",
                "max_tokens_to_sample": 256,
                "use_async_client": False,
                "max_retries": 3,
                "timeout": 600,
                "streaming_callback": "haystack.preview.components.generators.anthropic.claude.default_streaming_callback",
                "api_base_url": "https://api.anthropic.com/v1/complete",
            },
        }
        component = ClaudeGenerator.from_dict(data)
        assert component.api_key == "test-api-key"
        assert component.model_name == "claude-instant-1"
        assert component.max_tokens_to_sample == 256
        assert component.use_async_client == False
        assert component.max_retries == 3
        assert component.timeout == 600
        assert component.streaming_callback == default_streaming_callback
        assert component.api_base_url == "https://api.anthropic.com/v1/complete"
        assert component.model_parameters == {}

    @pytest.mark.unit
    def test_from_dict_with_custom_parameters(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        data = {
            "type": "ClaudeGenerator",
            "init_parameters": {
                "model_name": "claude-instant-1",
                "max_tokens_to_sample": 10,
                "use_async_client": True,
                "max_retries": 2,
                "timeout": 20,
                "streaming_callback": "haystack.preview.components.generators.anthropic.claude.default_streaming_callback",
                "api_base_url": "test-base-url",
                "temperature": 0.5,
                "top_p": 0.1,
                "top_k": 5,
                "stop_sequences": ["###", "---"],
            },
        }
        component = ClaudeGenerator.from_dict(data)
        assert component.api_key == "test-api-key"
        assert component.model_name == "claude-instant-1"
        assert component.max_tokens_to_sample == 10
        assert component.use_async_client == True
        assert component.max_retries == 2
        assert component.timeout == 20
        assert component.streaming_callback == default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {
            "temperature": 0.5,
            "top_p": 0.1,
            "top_k": 5,
            "stop_sequences": ["###", "---"],
        }

    @pytest.mark.unit
    def test_run_no_async(self):
        with patch("haystack.preview.components.generators.anthropic.claude.Anthropic") as mock_anthropic_client:
            mock_anthropic_client.return_value.completions.create.return_value = mock_anthropic_response
            component = ClaudeGenerator(api_key="test-api-key")
            results = component.run(prompt="test-prompt-1")
            assert results == {
                "replies": ["Generated response for this prompt: test-prompt-1"],
                "metadata": [{"model": "claude-instant-1", "finish_reason": "stop_sequence"}],
            }
            mock_anthropic_client.create.assert_called_once_with(
                model="claude-instant-1", api_key="test-api-key", prompt="test-prompt-1", stream=False
            )
