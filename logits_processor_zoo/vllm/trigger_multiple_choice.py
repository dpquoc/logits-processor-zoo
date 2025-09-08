from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List, Union
import torch
from logits_processor_zoo.utils import text_to_token, enforce_tokens

class TriggeredMultipleChoiceLogitsProcessor:
    """
    A logits processor that detects a trigger phrase in the generated output (or prompt) and then applies multiple choice enforcement for the immediate next token.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    choices (List[str]): List of choice strings (assumed to be single tokens, e.g., 'Yes', 'No').
    trigger_phrase (str): The phrase to detect in the output. When detected at the end of the context, the next token is enforced to be one of the choices.
    """
    def __init__(self, tokenizer: Union[PreTrainedTokenizer, str], choices: List[str], trigger_phrase: str):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.choices = choices
        self.trigger_phrase = trigger_phrase

        self.trigger_tokens = self.tokenizer.encode(self.trigger_phrase, add_special_tokens=False)
        self.choice_tokens = [text_to_token(self.tokenizer, choice, last=False) for choice in self.choices]
        self._reset()

    def clone(self):
        return TriggeredMultipleChoiceLogitsProcessor(self.tokenizer, self.choices, self.trigger_phrase)

    def _reset(self):
        self.triggered = False

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        if not past_token_ids:  # new generation
            self._reset()

        if self.triggered:
            return scores

        full_context = prompt_tokens_ids + past_token_ids
        trigger_len = len(self.trigger_tokens)
        if len(full_context) >= trigger_len and full_context[-trigger_len:] == self.trigger_tokens:
            self.triggered = True
            scores = enforce_tokens(scores, self.choice_tokens)

        return scores