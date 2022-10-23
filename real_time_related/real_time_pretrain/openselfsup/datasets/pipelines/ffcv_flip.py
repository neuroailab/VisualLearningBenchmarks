from numpy import dtype
from numpy.random import rand
import numpy as np
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from numba import njit
from dataclasses import replace

class RandomVerticalFlip(Operation):
    """Flip the image vertically with probability flip_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    flip_prob : float
        The probability with which to flip each image in the batch
        vertically.
    """

    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def flip(images, dst):
            should_flip = rand(images.shape[0]) < flip_prob
            for i in my_range(images.shape[0]):
                if should_flip[i]:
                    dst[i] = images[i, ::-1]
                else:
                    dst[i] = images[i]

            return dst

        flip.is_parallel = True
        return flip

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), 
                AllocationQuery(previous_state.shape, previous_state.dtype))
