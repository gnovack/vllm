# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm
from vllm.lora.request import LoRARequest
from vllm.distributed import cleanup_dist_env_and_memory

from ..utils import multi_gpu_test

MODEL_PATH = "openai/gpt-oss-20b"

PROMPT_TEMPLATE = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-10-29

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.
"
##Instruction:
farm contains tables such as city, farm, farm_competition, competition_record. Table city has columns such as City_ID, Official_Name, Status, Area_km_2, Population, Census_Ranking. City_ID is the primary key.
Table farm has columns such as Farm_ID, Year, Total_Horses, Working_Horses, Total_Cattle, Oxen, Bulls, Cows, Pigs, Sheep_and_Goats. Farm_ID is the primary key.
Table farm_competition has columns such as Competition_ID, Year, Theme, Host_city_ID, Hosts. Competition_ID is the primary key.
Table competition_record has columns such as Competition_ID, Farm_ID, Rank. Competition_ID is the primary key.
The Host_city_ID of farm_competition is the foreign key of City_ID of city.
The Farm_ID of competition_record is the foreign key of Farm_ID of farm.
The Competition_ID of competition_record is the foreign key of Competition_ID of farm_competition.


###Input:
{context}

###Response:<|end|><|start|>assistant<|channel|>final<|message|>"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT AVG(Working_Horses) FROM farm WHERE Total_Horses > 5000;",
    "SELECT MAX(Cows) AS Max_Cows, MIN(Cows) AS Min_Cows FROM farm;",
    "SELECT MAX(Cows) AS Max_Cows, MIN(Cows) AS Min_Cows FROM farm;",
]

from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel

@pytest.fixture
def expected_output(gptoss20b_lora_files):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    ).eval()
    
    model = PeftModel.from_pretrained(model, gptoss20b_lora_files).merge_and_unload()

    prompts = [
        PROMPT_TEMPLATE.format(
            context="Give the average number of working horses on farms with more than 5000 total horses."  # noqa: E501
        ),  # noqa: E501
        PROMPT_TEMPLATE.format(
            context="What are the maximum and minimum number of cows across all farms."
        ),
        PROMPT_TEMPLATE.format(
            context="Return the maximum and minimum number of cows across all farms."
        ),
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_lengths = inputs['attention_mask'].sum(dim=1)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        num_return_sequences=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract only the newly generated tokens
    generated_texts = []
    for i, output in enumerate(outputs):
        new_tokens = output[input_lengths[i]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text.strip())

    del model
    cleanup_dist_env_and_memory()
    
    return generated_texts




def generate_and_test(llm: vllm.LLM, lora_path: str, lora_id: int, expected_outputs: list[str]) -> None:
    prompts = [
        PROMPT_TEMPLATE.format(
            context="Give the average number of working horses on farms with more than 5000 total horses."  # noqa: E501
        ),  # noqa: E501
        PROMPT_TEMPLATE.format(
            context="What are the maximum and minimum number of cows across all farms."
        ),
        PROMPT_TEMPLATE.format(
            context="Return the maximum and minimum number of cows across all farms."
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
    for i in range(len(expected_outputs)):
        assert generated_texts[i].startswith(expected_outputs[i])


def test_gpt_oss_lora(gptoss20b_lora_files, expected_output):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=8,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        compilation_config=vllm.config.CompilationConfig(  # Avoid OOM
            cudagraph_specialize_lora=False,
        ),
    )

    generate_and_test(llm, gptoss20b_lora_files, lora_id=1, expected_outputs=expected_output)
    generate_and_test(llm, gptoss20b_lora_files, lora_id=2, expected_outputs=expected_output)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
def test_gpt_oss_lora_tp2(gptoss20b_lora_files, fully_sharded_loras):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        fully_sharded_loras=fully_sharded_loras,
        compilation_config=vllm.config.CompilationConfig(  # Avoid OOM
            cudagraph_specialize_lora=False,
        ),
    )

    generate_and_test(llm, gptoss20b_lora_files, lora_id=1)
    generate_and_test(llm, gptoss20b_lora_files, lora_id=2)
