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
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.choices = choices
        self.trigger_phrase = trigger_phrase

        # This is the sequence of token IDs we are looking for.
        self.trigger_tokens = self.tokenizer.encode(self.trigger_phrase, add_special_tokens=False)
        if not self.trigger_tokens:
            raise ValueError(f"Trigger phrase '{self.trigger_phrase}' tokenized to an empty list.")
            
        # These are the token IDs we will force the model to choose from.
        self.choice_tokens = [text_to_token(self.tokenizer, choice, last=False) for choice in self.choices]
        
        self._reset()

    def clone(self):
        # vLLM requires a clone method for stateful processors to handle batching.
        return TriggeredMultipleChoiceLogitsProcessor(self.tokenizer, self.choices, self.trigger_phrase)

    def _reset(self):
        """Resets the state of the processor."""
        self._enforce_next_token = False

    def __call__(self, prompt_token_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        # This method signature is for demonstration; vLLM uses a slightly different one.
        # The logic inside is what matters. vLLM provides the full sequence in `input_ids`.
        # For compatibility with your script, we'll assume this structure.
        
        # On the first token of a new generation, reset the state.
        if not past_token_ids:
            self._reset()

        # --- Step 1: Enforce if the flag was set in the previous step ---
        if self._enforce_next_token:
            # Apply the enforcement
            scores = enforce_tokens(scores, self.choice_tokens)
            # Reset the flag immediately so we only enforce for this single token.
            self._enforce_next_token = False
            return scores

        # --- Step 2: Search for the trigger phrase at the end of the current sequence ---
        full_context = prompt_token_ids + past_token_ids
        trigger_len = len(self.trigger_tokens)

        # We only need to check if we have enough tokens to potentially match the trigger
        if len(full_context) >= trigger_len:
            # Check if the last N tokens match our trigger sequence
            if full_context[-trigger_len:] == self.trigger_tokens:
                # If they match, set the flag. The enforcement will happen on the *next* call.
                self._enforce_next_token = True
        
        return scores