# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional
import torch
from torch.distributed import ProcessGroup
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from .base_device_communicator import DeviceCommunicatorBase
import ray
# Check if we're using Ray for distributed execution
USE_RAY = get_current_vllm_config().parallel_config.distributed_executor_backend == "ray"
logger = init_logger(__name__)

if current_platform.is_tpu():
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch_xla._internal import pjrt
    from torch_xla.distributed.xla_multiprocessing import (
        create_optimized_replica_groups)
    # Import ray_utils regardless of execution mode to maintain your preferred approach
    from vllm.executor import ray_utils

class TpuCommunicator(DeviceCommunicatorBase):
    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        # NOTE(woosuk): When using TP > 1 on TPUs, every TPU on the same node
        # must be used together. Therefore, the local rank and world size can
        # be simply calculated as follows.
        global_rank = self.global_rank
        global_world_size = self.global_world_size

        if ray.is_initialized():
            logger.info("Ray is already initialized. Skipping Ray initialization.")
        elif current_platform.is_rocm() or current_platform.is_xpu():
            # Try to connect existing ray instance and create a new one if not found
            try:
                ray.init("auto")
            except ConnectionError:
                logger.warning(
                    "No existing RAY instance detected. "
                    "A new instance will be launched with current node resources.")
        
        # Always use Ray approach, removing the MP branch entirely
        logger.info("TpuCommunicator initialized with RAY")
        # Calculate how many TPU nodes are in the current deployment. This
        # is the Ray placement group if it is deployed with Ray. Default
        # to the number of TPU nodes in the Ray cluster. The number of TPU
        # nodes is computed by the total number of TPUs divided by the
        # number of TPU accelerators per node, to account for clusters
        # with both CPUs and TPUs.
        num_nodes = ray_utils.get_num_tpu_nodes()
        num_nodes_in_pg = ray_utils.get_num_nodes_in_placement_group()
        if num_nodes_in_pg > 0:
            num_nodes = num_nodes_in_pg
        local_world_size = global_world_size // num_nodes
        local_rank = global_rank % local_world_size

        # Ensure environment variables are set for multihost deployments.
        # On GKE, this is needed for libtpu and TPU driver to know which TPU
        # chip is actually visible. Otherwise the TPU driver will fail to
        # initialize because the number of devices would be different from
        # the number of visible worker addresses.
        os.environ["CLOUD_TPU_TASK_ID"] = str(global_rank)
        os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
        pjrt.initialize_multiprocess(local_rank, local_world_size)
        xr._init_world_size_ordinal()
        

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Keep the groups specification from latest code for optimized communication
        return xm.all_reduce(xm.REDUCE_SUM, input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "TPUs only support dim=-1 for all-gather."
        return xm.all_gather(input_, dim=dim)
