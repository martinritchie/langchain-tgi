from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel, BaseMessage, ChatResult
from langchain.llms import HuggingFaceTextGenInference
from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


class BaseHuggingFaceTextGenChat(BaseChatModel, ABC):
    llm: HuggingFaceTextGenInference

    @abstractmethod
    def _preprocess(self, messages: List[BaseMessage]) -> str:
        """Format the messages into a string suitable for the LLM"""
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, output: str) -> ChatResult:
        """Format the output from the LLM into a ChatResult"""
        raise NotImplementedError

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        input_text = self._preprocess(messages)
        stop_tokens = stop or [self.HF_EOS_TOKEN]
        model_output = self.llm(
            prompt=input_text, stop=stop_tokens, run_manager=run_manager, **kwargs
        )
        return self._postprocess(model_output)


class Llama2Chat(BaseHuggingFaceTextGenChat):
    """Llama2Chat model"""

    # Llama2 specific tokens, please see:
    #  https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
    B_INST: str = "[INST]"
    E_INST: str = "[/INST]"
    B_SYS: str = "<<SYS>>\n"
    E_SYS: str = "\n<</SYS>>\n\n"

    # HuggingFace specific tokens
    HF_BOS_TOKE: str = "<s>"
    HF_EOS_TOKEN: str = "</s>"

    join_char: str = "\n"

    def message2string(self, message: BaseMessage) -> str:
        """Converts message to a string"""
        if isinstance(message, HumanMessage):
            return f"{self.HF_BOS_TOKEN}{self.B_INST}{message.content}{self.E_INST}"
        elif isinstance(message, AIMessage):
            return f"{message.content}{self.HF_EOS_TOKEN}"
        elif isinstance(message, SystemMessage):
            return (
                f"{self.HF_BOS_TOKEN}{self.B_SYS}{message.content}{self.E_SYS}"
                f"{self.HF_EOS_TOKEN}"
            )
        else:
            raise ValueError(
                (
                    f"Received unsupported message type: {type(message)}.\n"
                    " Supported message types for the LLaMa-2 Foundation Model:"
                    f" {[SystemMessage, HumanMessage, AIMessage]}"
                )
            )

    def _preprocess(self, messages: List[BaseMessage]) -> str:
        """Format the messages into a string suitable for the LLM"""
        return self.join_char.join(
            [self.message2string(message) for message in messages]
        )
