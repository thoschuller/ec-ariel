import torch
from typing_extensions import Protocol

class _Checkpointable(Protocol):
    """
    Interface for checkpointable objects.
    Implemented as a protocol, implicit subtyping is supported so subclasses do not need to inherit this explicitly.
    This is to allow arbitrary objects/tensor subclasses to hook into DCP seamlessly through implementing the interface.
    """
    def __create_write_items__(self, fqn: str, object: object) -> list[object]:
        """
        Return a list of WriteItems based on object's contents.
        """
    def __create_chunk_list__(self) -> list[object]:
        """
        Return a list of `ChunkStorageMetadata` based on object's contents.
        """
    def __get_tensor_shard__(self, index: int) -> torch.Tensor:
        """
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
