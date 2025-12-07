import re
from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("apriel")
class AprielReasoningParser(ReasoningParser):
    """
    Reasoning parser for Apriel

    AprielReasoningParser

    This class implements a reasoning parser specifically designed
    for the Apriel Model. It is responsible for parsing and
    extracting structured reasoning and answer segments from model
    outputs that follow a specific pattern.

    Key Features:
        - For non-stream output , Recognizes and extracts reasoning ("think")
         and answer ("answer") sections from text using regular expressions.
        - For stream process, it requires a token id sequences to change the
          reasoning state and other state so it maintains internal state to
          manage parsing across multiple token.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        self.think_start_expr = "Here are my reasoning steps:"
        self.think_end_expr = "[BEGIN FINAL RESPONSE]"

        self.response_start_expr = "[BEGIN FINAL RESPONSE]"
        self.response_end_expr = "<|end|>"
        self.secondary_end_exp = "[END FINAL RESPONSE]"

        self.think_start_ids = [11745, 1584, 2036, 38528, 9578, 1058]
        self.response_start_ids = [998]
        self.response_end_ids = [999]


        # when state change, send out all the buffered text in last state
        self.buffered_text = []
        self.buffered_ids = []

        self.current_state = "reasoning"
        self.all_states = ["reasoning", "response"]

        self.current_state = "think"
        self.expected_sequence = self.think_start_ids
        self.sequence_index = 0
        self.token_buffer = []
        self.text_buffer = ""

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.current_state == "response"

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # for hunyuan streaming reason parsing, the stream parse
        # will call first, and the same token will be called in
        # is_reasoning_end and extract_content_ids
        # this id is not part of content, so just return [] here.
        return []

    def extract_reasoning_content(
            self, model_output: str, request: "ChatCompletionRequest"
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract reasoning and final response from model output.

        Args:
            model_output (str): Full raw output string from the model.
            request (ChatCompletionRequest): Request being processed.

        Returns:
            tuple[Optional[str], Optional[str]]:
                (reasoning_content, final_message).
                If no response markers are found, returns (None, model_output).
        """
        start = model_output.find(self.response_start_expr)
        end = model_output.find(self.response_end_expr)
        if end == -1:
            end = model_output.find(self.secondary_end_exp)

        if start != -1:  # Found "[BEGIN FINAL RESPONSE]"
            # Everything before start is reasoning
            reasoning_section = model_output[:start].strip()
            if self.think_start_expr in reasoning_section:
                reasoning_content = reasoning_section.replace(self.think_start_expr, "").strip()
            else:
                reasoning_content = reasoning_section

            # Take everything after start marker
            if end != -1 and end > start:
                final_message = model_output[start + len(self.response_start_expr):end].strip()
            else:
                # No <|end|> → take until end of string
                final_message = model_output[start + len(self.response_start_expr):].strip()

            return reasoning_content, final_message

        # Fallback: no markers → treat whole thing as final message
        return None, model_output.strip()

    def extract_reasoning_content_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """Extract content using token ID sequence state machine"""
        # Define sequences

        if self.response_start_expr not in previous_text:
            return DeltaMessage(reasoning_content=delta_text)
        else:
            return DeltaMessage(content=delta_text)
