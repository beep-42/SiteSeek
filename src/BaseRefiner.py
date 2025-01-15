from abc import ABC, abstractmethod
from typing import Dict, List

from numpy.typing import NDArray


class BaseRefiner(ABC):

    @abstractmethod
    def refine(self, possible_matches: Dict[int, List[int]], positions: NDArray,
                               template_positions: NDArray, rot: NDArray, trans: NDArray) -> (
            Dict[int, int], float, NDArray, NDArray
    ):
        pass
