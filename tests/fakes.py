"""Test doubles for LLM-based code."""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


class FakeStructuredChatModel(BaseChatModel):
    """Returns pre-programmed structured responses, picked by schema type.

    `responses` is consumed in order per schema; `prompts` records what the model was sent.
    """

    responses: list[BaseModel] = Field(default_factory=list)
    prompts: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        def respond(prompt: LanguageModelInput) -> BaseModel:
            self.prompts.append(prompt)
            for i, response in enumerate(self.responses):
                if isinstance(schema, type) and isinstance(response, schema):
                    return self.responses.pop(i)
            raise AssertionError(f"No fake response left for schema {schema!r}")

        return RunnableLambda(respond)


def prompt_text(prompt: Any) -> str:
    """Flatten a recorded prompt (a PromptValue) to plain text for assertions."""
    messages: Sequence[BaseMessage] = prompt.to_messages()
    return "\n".join(str(m.content) for m in messages)
