import os
from typing import Optional
import torch
from torch.distributed import ProcessGroup
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from .base_device_communicator import DeviceCommunicatorBase

# Define this variable based on configuration
USE_RAY = get_current_vllm_config().parallel_config.distributed_executor_backend == "ray"
logger = init_logger(__name__)

if current_platform.is_tpu():
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch_xla._internal import pjrt
    
    # Import conditionally to prevent errors
    if USE_RAY:
        from vllm.executor import ray_utils
    
    # Only import if we need it
    try:
        from torch_xla.distributed.xla_multiprocessing import create_optimized_replica_groups
    except ImportError:
        # Provide a fallback if this isn't available
        def create_optimized_replica_groups():
            return None

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
        
        if USE_RAY:
            logger.info("TpuCommunicator initialized with RAY")
            # Calculate how many TPU nodes are in the current deployment
            num_nodes = ray_utils.get_num_tpu_nodes()
            num_nodes_in_pg = ray_utils.get_num_nodes_in_placement_group()
            if num_nodes_in_pg > 0:
                num_nodes = num_nodes_in_pg
            local_world_size = global_world_size // num_nodes
            local_rank = global_rank % local_world_size
        else:
            logger.info("TpuCommunicator initialized with MP")
            # For non-Ray mode, use simpler approach from second version
            local_world_size = global_world_size  # Or handle differently as needed
            local_rank = global_rank % local_world_size
            
            # Add this check only if needed and available in your environment
            try:
                num_hosts = torch_xla.tpu.num_tpu_workers()
                if num_hosts > 1:
                    logger.warning(f"Running on {num_hosts} hosts, expected 1")
            except:
                logger.info("Could not check number of TPU workers")
        
        # Ensure environment variables are set for multihost deployments
        os.environ["CLOUD_TPU_TASK_ID"] = str(global_rank)
        os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
        
        pjrt.initialize_multiprocess(local_rank, local_world_size)
        xr._init_world_size_ordinal()
        
        # Only create groups if the function exists
        try:
            self.groups = create_optimized_replica_groups()
        except:
            self.groups = None
            logger.warning("Could not create optimized replica groups")

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Use groups parameter only if available
        if hasattr(self, 'groups') and self.groups is not None:
            return xm.all_reduce(xm.REDUCE_SUM, input_, groups=self.groups)
        else:
            return xm.all_reduce(xm.REDUCE_SUM, input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "TPUs only support dim=-1 for all-gather."
        return xm.all_gather(input_, dim=dim)
