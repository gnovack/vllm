# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "openai/gpt-oss-20b"

PROMPT_TEMPLATE = "Question: {question}\n\nQuery: "  # noqa: E501


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    prompts = [
        PROMPT_TEMPLATE.format(question="Count the number of farms."),
        PROMPT_TEMPLATE.format(
            question=(
                "What is the average number of employees of the departments "
                "whose rank is between 10 and 15?"
            )
        ),
        PROMPT_TEMPLATE.format(
            question=(
                "What is the status of the city that has hosted the most competitions?"
            )
        ),
        PROMPT_TEMPLATE.format(
            question="which course has most number of registered students?"
        ),
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
        "SELECT COUNT(*) FROM farms",
        "SELECT AVG(Num_Employees) FROM Department WHERE Rank BETWEEN 10 AND 15",
        (
            "SELECT status FROM city WHERE city_id IN (SELECT city_id FROM competition "
            "GROUP BY city_id ORDER BY COUNT(*) DESC LIMIT 1);"
        ),
        (
            "SELECT course_id, COUNT(*) AS num_of_students FROM register GROUP BY "
            "course_id ORDER BY num_of_students DESC LIMIT 1;"
        ),
    ]

    output1 = do_sample(llm, gptoss20b_lora_files, lora_id=1)
    print(output1)
    for i in range(len(expected_lora_output)):
        cleaned_output = re.sub(r"\s+", " ", output1[i])
        assert cleaned_output.startswith(expected_lora_output[i])
