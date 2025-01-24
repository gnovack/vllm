export NEURON_CC_FLAGS="--verbose=debug --logfile=neuron-compiler.log --internal-compiler-debug-mode=penguin --compile_workdir=/root/workspace/gnovack/vllm/compiler-workdir --logical-nc-config=2 -O1"
VLLM_USE_V1=1 PYTHONPATH=/root/workspace/gnovack/vllm python vllm/entrypoints/openai/api_server.py \
    --model /root/workspace/gnovack/models/llama-3.1-8b-instruct \
    --max-num-seqs 8 \
    --max-model-len 4096 \
    --max-num-batched-tokens 128 \
    --enable-chunked-prefill \
    --block-size 128 \
    --device neuron \
    -tp 4 \
    --worker-cls="vllm.v1.worker.neuron_worker.NeuronWorker"