import vllm


class vLLMRunner:
    def __init__(self, model_name='google/gemma-1.1-2b-it'):
        self.model = vllm.LLM(
            model_name,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True
        )
        self.tokenizer = self.model.get_tokenizer()

    def generate_response(self, prompts, logits_processor_list=None, max_tokens=1000):
        if logits_processor_list is None:
            logits_processor_list = []

        gen_output = self.model.generate(
            prompts,
            vllm.SamplingParams(
                n=1,
                temperature=0,
                seed=0,
                skip_special_tokens=True,
                max_tokens=max_tokens,
                logits_processors=logits_processor_list
            ),
            use_tqdm=False
        )

        for prompt, out in zip(prompts, gen_output):
            out = out.outputs[0].text
            print(f"Prompt: {prompt}")
            print(out)
            print("-----END-----")
            print()
