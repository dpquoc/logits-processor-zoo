#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from logits_processor_zoo.transformers import MultipleChoiceLogitsProcessor


def test_cite_from_prompt_logits_processor(llm_runner):
    example_prompts = [
        """
I am getting a lot of calls during the day. What is more important for me to consider when I buy a new phone?
a) Camera
b) Screen resolution
c) Operating System
d) Battery

Answer:
        """,

        """
Which user review doesn't belong to a summer dress?
a) Looks good
b) Keeps warm
c) Too long
d) Liked the color

Answer:
        """
    ]

    choices = ["a", "b", "c", "d"]
    logits_processors = [MultipleChoiceLogitsProcessor(llm_runner.tokenizer, choices=choices, delimiter=")")]
    processed_gen_output = llm_runner.generate_response(example_prompts, logits_processors,  max_new_tokens=1)

    assert all((out in choices) for out in processed_gen_output)

    example_prompts = [prompt.replace("a)", "1.").replace("b)", "2.").replace("c)", "3.").replace("d)", "4.")
                       for prompt in example_prompts]

    choices = ["1", "2", "3", "4"]
    logits_processors = [MultipleChoiceLogitsProcessor(llm_runner.tokenizer, choices=choices, delimiter=".",
                                                       boost_first_words=1.0)]
    processed_gen_output = llm_runner.generate_response(example_prompts, logits_processors,  max_new_tokens=1)

    assert all((out in choices) for out in processed_gen_output)
