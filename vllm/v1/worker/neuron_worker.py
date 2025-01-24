"""A GPU worker class."""
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed
import torch_xla.core.xla_model as xm

from vllm.config import ParallelConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.neuron_model_runner import NeuronModelRunner

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput


class NeuronWorker(Worker):

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # TODO(gnovack) - get number of blocks based on available mem
        return (1_000, 0)

    def initialize(self):
        if self.device_config.device.type == "cpu":
            
            # TODO(gnovack) - support logical nc configs here too
            # os.environ["NEURON_RT_NUM_CORES"] = "1"
            
            # Initialize the distributed environment.
            init_worker_distributed_environment(self.parallel_config, self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)
            
            self.device = xm.xla_device()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        with torch.inference_mode():
            self.model_runner = NeuronModelRunner(self.vllm_config, self.device)

        # TODO(gnovack) - get number of blocks based on available mem
        self.cache_config.num_gpu_blocks = 1_000
        self.cache_config.num_cpu_blocks = 0

    def compile_or_warm_up_model(self):
        # TODO: Implement AOT compilation logic here...
        self.model_runner.capture_model()
        ...
    
    def initialize_cache(self, num_device_blocks: int) -> None:
        # TODO(gnovack) - validate num_device_blocks
        self.model_runner.initialize_kv_cache(num_device_blocks)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend="gloo")

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size, backend="gloo")