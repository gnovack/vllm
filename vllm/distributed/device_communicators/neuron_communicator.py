
import torch
from torch.distributed import ProcessGroup
from vllm.platforms import current_platform

if current_platform.is_neuron():
    import torch_xla.core.xla_model as xm


class NeuronCommunicator:

    def __init__(self, group: ProcessGroup):
        if not current_platform.is_neuron():
            self.disabled = True
            return
        self.disabled = False

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return xm.all_reduce(xm.REDUCE_SUM, x)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "TPUs only support dim=-1 for all-gather."
        return xm.all_gather(x, dim=dim)
