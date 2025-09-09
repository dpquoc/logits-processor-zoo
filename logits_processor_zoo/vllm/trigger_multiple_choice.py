from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List, Union
import torch
from logits_processor_zoo.utils import text_to_token, enforce_tokens

class TriggeredMultipleChoiceLogitsProcessor:
    """
    A logits processor that detects a trigger phrase in the generated output and then 
    applies multiple choice enforcement for the immediate next token.
    This version is stateless.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    choices (List[str]): List of choice strings (assumed to be single tokens, e.g., 'yes', 'no').
    trigger_phrase (str): The phrase to detect in the output. When detected at the end of the context, 
                          the next token is enforced to be one of the choices.
    """
    def __init__(self, tokenizer: Union[PreTrainedTokenizer, str], choices: List[str], trigger_phrase: str):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.choices = choices
        self.trigger_phrase = trigger_phrase

        # Store the token IDs for the trigger phrase
        self.trigger_tokens = self.tokenizer.encode(self.trigger_phrase, add_special_tokens=False)
        self.trigger_len = len(self.trigger_tokens)
        
        # Store the token IDs for the choices
        self.choice_tokens = [text_to_token(self.tokenizer, choice, last=False) for choice in self.choices]
        
        # This processor is stateless, so clone can just return a new instance
    def clone(self):
        return TriggeredMultipleChoiceLogitsProcessor(self.tokenizer, self.choices, self.trigger_phrase)

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        # Combine the prompt and the generated tokens to get the full context
        full_context = prompt_tokens_ids + past_token_ids

        # If the context is too short to contain the trigger, do nothing
        if len(full_context) < self.trigger_len:
            return scores

        # Check if the last N tokens of the context match our trigger phrase
        if full_context[-self.trigger_len:] == self.trigger_tokens:
            # If they match, it means the trigger was just completed.
            # Enforce the choices for the *current* token being generated.
            scores = enforce_tokens(scores, self.choice_tokens)

        return scores