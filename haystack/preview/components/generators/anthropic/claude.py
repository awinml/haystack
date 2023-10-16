import asyncio
import logging
from collections import defaultdict
import sys
from typing import Any, Callable, Dict, List, Optional

from haystack.lazy_imports import LazyImport
from haystack.preview import DeserializationError, component, default_from_dict, default_to_dict

with LazyImport(message="Run 'pip install anthropic'") as anthropic_import:
    from anthropic import Anthropic, AsyncAnthropic

logger = logging.getLogger(__name__)


API_BASE_URL = "https://api.anthropic.com/v1/complete"


def default_streaming_callback(chunk):
    """
    Default callback function for streaming responses from Anthropic API.
    Prints the tokens of the first completion to stdout as soon as they are received and returns the chunk unchanged.
    """
    print(chunk.completion, end="", flush=True)


@component
class ClaudeGenerator:
    """
    LLM Generator compatible with Claude (Anthropic) large language models.

    Queries the LLM using Anthropic's API. Invocations are made using Anthropic SDK ('anthropic' package)
    See [Anthropic Claude API](https://docs.anthropic.com/claude/reference/) for more details.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-instant-1",
        use_async_client: bool = False,
        max_retries: Optional[int] = 3,
        timeout: Optional[int] = 600,
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = API_BASE_URL,
        **kwargs,
    ):
        """
        Creates an instance of ClaudeGenerator for Anthropic's Claude models.

        :param api_key: The Anthropic API key.
        :param model_name: The name of the model to use.
        :param use_async_client: Boolean Flag to select the Async Client, defaults to `False`. It is recommended to use Async Client for applications with many concurrent calls.
        :param max_retries: Maximum number of retries for requests, defaults to `2`.
        :param timeout: Request timeout in seconds, defaults to `600`.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The Anthropic API Base url, defaults to `https://api.anthropic.com/v1/complete`.
        :param kwargs: Other parameters to use for the model. These parameters are all sent directly to the Anthropic
            endpoint. See Anthropic [documentation](https://docs.anthropic.com/claude/docs) for more details.
            Some of the supported parameters:
            - `max_tokens_to_sample`: The maximum number of tokens the output text can have.
            - `temperature`: The sampling temperature to use, controls the amount of randomness injected into the response. Ranges from 0 to 1. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications, and 0
                 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature,
            called nucleus sampling, where the model
                considers the results of the tokens with top_p
                 probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are
                considered. You should either alter temperature or
                 top_p, but not both.
            - `top_k`: Ensures only the top k most likely tokens are considered for generation at each step. Used to remove "long tail" low probability responses.
            - `stop_sequences`: One or more additional sequences
             after which the LLM should stop generating tokens.
                By default, Anthropic models stop on "\n\nHuman:",
                and may include additional built-in stop sequences
                 in the future.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.use_async_client = use_async_client
        self.max_retries = max_retries
        self.timeout = timeout
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.model_parameters = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        if self.streaming_callback:
            module = self.streaming_callback.__module__
            if module == "builtins":
                callback_name = self.streaming_callback.__name__
            else:
                callback_name = f"{module}.{self.streaming_callback.__name__}"
        else:
            callback_name = None

        return default_to_dict(
            self,
            api_key=self.api_key,
            model_name=self.model_name,
            use_async_client=self.use_async_client,
            max_retries=self.max_retries,
            timeout=self.timeout,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            **self.model_parameters,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaudeGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        streaming_callback = None
        if "streaming_callback" in init_params and init_params["streaming_callback"]:
            parts = init_params["streaming_callback"].split(".")
            module_name = ".".join(parts[:-1])
            function_name = parts[-1]
            module = sys.modules.get(module_name, None)
            if not module:
                raise DeserializationError(f"Could not locate the module of the streaming callback: {module_name}")
            streaming_callback = getattr(module, function_name, None)
            if not streaming_callback:
                raise DeserializationError(f"Could not locate the streaming callback: {function_name}")
            data["init_parameters"]["streaming_callback"] = streaming_callback
        return default_from_dict(cls, data)

    def _generate_completion(self, client, prompt):
        generated_completion = client.completions.create(
            model=self.model_name, prompt=prompt, stream=self.streaming_callback is not None, **self.model_parameters
        )
        return generated_completion

    async def _generate_completion_async(self, client, prompt):
        generated_completion = await client.completions.create(
            model=self.model_name, prompt=prompt, stream=self.streaming_callback is not None, **self.model_parameters
        )
        return generated_completion

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str):
        """
        Queries the LLM with the prompts to produce replies.

        :param prompts: The prompts to be sent to the generative model.
        """

        if self.use_async_client is True:
            client = AsyncAnthropic(api_key=self.api_key, max_retries=self.max_retries, timeout=self.timeout)
            completion = asyncio.run(self._generate_completion_async(client, prompt))
        else:
            client = Anthropic(api_key=self.api_key, max_retries=self.max_retries, timeout=self.timeout)
            completion = self._generate_completion(client, prompt)

        replies: List[str]
        metadata: List[Dict[str, Any]]
        if self.streaming_callback:
            replies_dict: Dict[str, str] = defaultdict(str)
            metadata_dict: Dict[str, Dict[str, Any]] = defaultdict(dict)
            for chunk in completion:
                chunk = self.streaming_callback(chunk)
                replies_dict[chunk] = chunk.completion
                metadata_dict[chunk] = {"model": chunk.model, "stop_reason": chunk.stop_reason}
            replies = list(replies_dict.values())
            metadata = list(metadata_dict.values())
            self._check_truncated_answers(metadata)
            return {"replies": replies, "metadata": metadata}

        metadata = [{"model": completion.model, "stop_reason": completion.stop_reason}]
        replies = [completion.completion]
        self._check_truncated_answers(metadata)
        return {"replies": replies, "metadata": metadata}

    def _check_truncated_answers(self, metadata: List[Dict[str, Any]]):
        """
        Check the `stop_reason` returned with the Anthropic completions.
        If the `stop_reason` is `max_tokens`, log a warning to the user.

        :param result: The result returned from the Anthropic API.
        :param payload: The payload sent to the Anthropic API.
        """
        truncated_completions = sum(1 for meta in metadata if meta.get("stop_reason") != "stop_sequence")
        if truncated_completions > 0:
            logger.warning(
                "%s out of the %s completions have been truncated before reaching a natural stopping point. "
                "Increase the max_tokens_to_sample parameter to allow for longer completions.",
                truncated_completions,
                len(metadata),
            )
