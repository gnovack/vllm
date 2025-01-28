from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger
from .interface import _Backend, Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON
    device_name: str = "neuron"
    device_type: str = "neuron"
    ray_device_key: str = "neuron_cores"
    dispatch_key: str = "XLA"
    supported_quantization: list[str] = ["neuron_quant"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.NEURON:
            logger.info("Cannot use %s backend on Neuron.", selected_backend)
        return _Backend.NEURON

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool) -> str:
        if not use_v1:
            logger.info("Neuron backend is only supported in V1")
        logger.info("Using Pallas backend.")
        return "vllm.v1.attention.backends.neuron_attn.NeuronAttentionBackend"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm.worker.neuron_worker.NeuronWorker"


        assert (vllm_config.lora_config
                is None), "LoRA is not supported for Neuron backend."
        assert (not vllm_config.speculative_config
                ), "Speculative decoding not yet supported for Neuron backend."


    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False
