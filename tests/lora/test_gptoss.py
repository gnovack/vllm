# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "openai/gpt-oss-20b"

# PROMPT_TEMPLATE = "<｜begin▁of▁sentence｜>You are a helpful assistant.\n\nUser: {context}\n\nAssistant:"  # noqa: E501
PROMPT_TEMPLATE = "Question: {question}\nAnswer: "  # noqa: E501


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    question = (
        "Mr. Sanchez found out that 40% of his Grade 5 students "
        "got a final grade below B. How many of his students got "
        "a final grade of B and above if he has 60 students in Grade 5?"
    )
    prompts = [
        PROMPT_TEMPLATE.format(question=question),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None,
    )
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


# FIXME: Load gpt-oss adapter
def test_gptoss20b_lora(gptoss20b_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_loras=4,
        trust_remote_code=True,
        # enforce_eager=True
    )

    expected_lora_output = [
        "40% of 60 students = 0.4 x 60 = 24 students\n60 students - 24 students = 36 "
        "students\nTherefore, 36 students got a final grade of B and above."
    ]

    output1 = do_sample(llm, gptoss20b_lora_files, lora_id=1)
    print(output1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
