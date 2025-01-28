"""A GPU worker class."""
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed
import torch_xla.core.xla_model as xm
from torch_xla._internal.pjrt import initialize_multiprocess

from vllm.config import ParallelConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.neuron_model_runner import NeuronModelRunner

logger = init_logger(__name__)

# if TYPE_CHECKING:
#     from vllm.v1.core.scheduler import SchedulerOutput


class NeuronWorker(Worker):

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        return 6e9

    def init_device(self):
        if self.device_config.device.type == "cpu":
            
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

    def compile_or_warm_up_model(self):
        # TODO: Implement AOT compilation logic here...
        self.model_runner.capture_model()
    
    def initialize_cache(self, kv_cache_config: KVCacheConfig) -> None:
        # TODO(gnovack) - validate num_device_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    initialize_multiprocess(rank, parallel_config.tensor_parallel_size)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend="xla")

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)