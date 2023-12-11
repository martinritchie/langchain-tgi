import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from pydantic import BaseModel, validator
from text_generation import AsyncClient
from text_generation.types import Response

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import SimpleChatModel
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk, BaseMessageChunk

logger = logging.getLogger(__name__)


class HFLlamaFormatter:
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    HF_BOS_TOKEN, HF_EOS_TOKEN = "<s>", "</s>"

    @classmethod
    def message2string(cls, message: BaseMessage) -> str:
        """Converts message to a string"""
        if isinstance(message, HumanMessage):
            return f"{cls.HF_BOS_TOKEN}{cls.B_INST}{message.content}{cls.E_INST}"
        elif isinstance(message, AIMessage):
            return f"{message.content}{cls.HF_EOS_TOKEN}"
        elif isinstance(message, SystemMessage):
            return f"{cls.HF_BOS_TOKEN}{cls.B_SYS}{message.content}{cls.E_SYS}{cls.HF_EOS_TOKEN}"
        else:
            raise ValueError(
                (
                    f"Received unsupported message type: {type(message)}.\n"
                    " Supported message types for the LLaMa-2 Foundation Model:"
                    f" {[SystemMessage, HumanMessage, AIMessage]}"
                )
            )

    @classmethod
    def dialog2string(cls, dialog: List[BaseMessage], join_char: str = "\n") -> str:
        string = join_char.join([cls.message2string(message) for message in dialog])
        return string


class ClientKwargs(BaseModel):
    base_url: str  # text-generation-inference instance base url
    headers: Optional[Dict[str, str]] = None  # Additional headers
    cookies: Optional[Dict[str, str]] = None  # Cookies to include in the requests
    timeout: int = 10  # Timeout in seconds


class HFChatTGI(SimpleChatModel):
    """"""

    client_kwargs: ClientKwargs
    tgi_client: AsyncClient

    content_formatter: Any = None
    """The content formatter that adds the special tokens to the prompt."""

    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    @validator("tgi_client", always=True, allow_reuse=True)
    def validate_client(cls, tgi_client, values: Dict) -> AsyncClient:
        """Validate that api key and python package exists in environment."""

        if tgi_client is None and values.get("tgi_client_parameters") is not None:
            logger.info("Creating new TGI client")
            tgi_client = AsyncClient(**values["tgi_client_parameters"].dict())
        if tgi_client is None and values.get("tgi_client_parameters") is None:
            tgi_client = cls._get_client()

        return tgi_client

    @classmethod
    def _get_client(cls) -> AsyncClient:
        """Get the client."""
        text_generation_service_url = os.getenv("TGI_URL", None)
        if text_generation_service_url is None:
            raise ValueError(
                "No TGI_URL environment variable found. Please set this to the URL of"
                " the TGI service, e.g., 'http://127.0.0.1:8080' or set the "
                " `client_kwargs` when initializing this model."
            )
        logger.info(f"Using TGI_URL: {text_generation_service_url}")

        return AsyncClient(text_generation_service_url)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface-llama2"

    def _call_base(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Response:
        """Call out to an AzureML Managed Online endpoint.
        Args:
            messages: The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = azureml_model("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        prompt = self.content_formatter._format_request_payload(messages, _model_kwargs)

        response = asyncio.run(self.tgi_client.generate(prompt=prompt, **kwargs))

        return response

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to an AzureML Managed Online endpoint.
        Args:
            messages: The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = azureml_model("Tell me a joke.")
        """
        response = self._call_base(messages, stop, run_manager, **kwargs)
        return response.generated_text

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the model.
        Args:
            messages: The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = model.generate("Tell me a joke.")
        """
        response = self._call_base(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )

        return self._create_chat_result(response)

    def _create_chat_result(self, response: Response) -> ChatResult:
        """Create a ChatResult from a response string."""
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response.generated_text),
                    generation_info=response.details.dict(),
                )
            ]
        )
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        _model_kwargs = self.model_kwargs or {}
        prompt = self.content_formatter._format_request_payload(messages, _model_kwargs)
        response = await self.tgi_client.generate(prompt=prompt, **kwargs)
        return self._create_chat_result(response)


    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        _model_kwargs = self.model_kwargs or {}
        prompt = self.content_formatter._format_request_payload(messages, _model_kwargs)

        response = asyncio.run(self.tgi_client.generate_stream(prompt=prompt, **kwargs))
        for chunk in response:
            yield self._create_chat_generation_chunk(chunk)


    def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        
