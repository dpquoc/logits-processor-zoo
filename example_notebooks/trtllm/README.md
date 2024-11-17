# Test TensorRT-LLM logits processors

## Quick Start

Follow this guide to create an engine:
https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html

## Examples

```
python example_notebooks/trtllm/gen_length_logits_processor.py --engine_path ../TensorRT-LLM/examples/llama/llama-engine/ --tokenizer_path ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/x/
python example_notebooks/trtllm/multiple_choice_logits_processor.py --engine_path ../TensorRT-LLM/examples/llama/llama-engine/ --tokenizer_path ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/x/ --prompt "Which one is heavier?\n1. 1 kg\n2. 100 kg\n3. 10 kg\nAnswer:"
```