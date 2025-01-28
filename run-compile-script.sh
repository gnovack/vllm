# rm -rf /var/tmp/neuron-compile-cache/*

# export TORCHDYNAMO_VERBOSE=1
export PYTHONPATH=/root/workspace/gnovack/vllm
# export TORCH_LOGS=+dynamo,graph
export NEURON_RT_NUM_CORES=16
# export XLA_DISABLE_FUNCTIONALIZATION=0
export NEURON_CC_FLAGS="-O1 --verbose=debug --logical-nc-config=1 --logfile=neuron-compiler.log --internal-compiler-debug-mode=all --compile_workdir=/root/workspace/gnovack/vllm/compiler-workdir"
# export NEURON_CC_FLAGS="-O1"

python examples/offline_model_neuron.py > compile-script-output 2>&1
# python examples/offline_inference_neuron.py > inference-script-output 2>&1
