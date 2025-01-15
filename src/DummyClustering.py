from typing import List, Optional, Dict

from numpy._typing import NDArray

from BaseClustering import BaseClustering
from KmerGen import generate_kmers
import numpy as np


class DummyClustering(BaseClustering):


    def __init__(self, top_k: int):

        self.top_k = top_k

    def set_template(self, template_positions_3d: NDArray, site_indices: List[int]) -> None:
        pass


    def cluster(self, candidate_ids: list[int], positions_3d: np.ndarray, sequences: list[str],
                all_possible_kmers: list[str], rated_kmer_dict: dict[str, dict[str, int]] = None)\
            -> dict[int: list[int]]:

        # test - one giant cluster
        accepted_clusters: list[dict[int, list[int]]] = []
        for candidate in candidate_ids:
            generated = generate_kmers(sequences[candidate])
            possible_positions = []
            for km in rated_kmer_dict.keys():
                hits = []
                for i, one in enumerate(generated):
                    if one in rated_kmer_dict[km].keys():
                        hits.append((i, rated_kmer_dict[km][one]))
                for o in sorted(hits, key=lambda x: x[1], reverse=True)[:self.top_k]:
                    possible_positions.append(o[0])

            possible_positions = list(set(possible_positions))
            if len(possible_positions) >= 3:
                accepted_clusters.append({candidate: possible_positions})

        return accepted_clusters