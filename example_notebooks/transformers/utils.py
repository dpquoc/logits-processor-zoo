import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList


class LLMRunner:
    def __init__(self, model_name='google/gemma-1.1-2b-it'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response(self, prompts, logits_processor_list=None, max_tokens=1000):
        if logits_processor_list is None:
            logits_processor_list = []

        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True)["input_ids"]

        out_ids = self.model.generate(input_ids.cuda(), max_new_tokens=max_tokens, min_new_tokens=1,
                                      logits_processor=LogitsProcessorList(logits_processor_list))

        gen_output = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for prompt, out in zip(prompts, gen_output):
            print(f"Prompt: {prompt}")
            print()
            print(f"LLM response:\n{out[len(prompt):].strip()}")
            print("-----END-----")
            print()
            print()
