from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Dict


class BaseClustering(ABC):

    @abstractmethod
    def cluster(self, candidate_ids: list[int], positions_3d: NDArray, sequences: List[str], all_possible_kmers: List[str],
                rated_kmer_dict: Optional[Dict[str, dict[str, int]]] = None) -> List[Dict[int, List[int]]]:
        pass


    @abstractmethod
    def set_template(self, template_positions_3d: NDArray, site_indices: List[int]) -> None:
        pass
