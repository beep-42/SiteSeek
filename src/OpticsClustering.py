import numpy as np

from BaseClustering import BaseClustering
from KmerGen import generate_kmers
from sklearn.cluster import OPTICS
from numpy.typing import NDArray
import warnings


class ClusteringOptics(BaseClustering):
    """
    This class contains the OPTICS-based clustering algorithm for clustering the K-mer centers of the similar K-mers
    in the sequence.

    Only clusters of max size of the original cavity times the cavity_scale_error are considered. If
    min_kmer_in_cavity_fraction is set, only clusters of at least such size are considered.

    Attributes:
        cavity_scale_error (float):
        min_kmer_in_cavity_fraction (float):
        top_k (int):
    """

    def __init__(self, cavity_scale_error: float, min_kmer_in_cavity_fraction: float, top_k: int = None) -> None:
        """
        Initializes the ClusteringOptics class.

        Args:
            cavity_scale_error (float): The maximum accepted distance between two K-mers in the hit cavity with respect
            to the maximum distance between residues of the original cavity.
            min_kmer_in_cavity_fraction (float): The minimum required relative number of K-mers in the hit cavity
            compared to the number of residues in the original cavity.
            top_k (int): if not None only the top_k most similar K-mers in the target sequences for each K-mer are
            considered during the clustering - significantly improves the runtime.
        """

        self.cavity_scale_error = cavity_scale_error
        self.min_kmer_in_cavity_fraction = min_kmer_in_cavity_fraction
        self.top_k = top_k

        self.optics = None
        self.site_size = None

        # supress max reachability warnings and divide with 0
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def set_template(self, template_positions_3d: NDArray, site_indices: list[int]) -> None:
        """
        Initializes the OPTICS-based clustering algorithm and calculates the maximum accepted distance between
        residues of the cluster.

        Args:
            template_positions_3d (NDArray): The original 3d positions of the residues.
            site_indices (list[int]): The indices of the binding site residues in the sequence.
        """

        # calculate the max pairwise distance
        max_distance = 0
        for a in site_indices:
            for b in site_indices:
                max_distance = max(max_distance, np.linalg.norm(template_positions_3d[a] - template_positions_3d[b]))
        # scale by the accepted cavity error (usually >= 1)
        max_distance *= self.cavity_scale_error

        self.optics = OPTICS(min_samples=3, max_eps=max_distance)
        self.site_size = len(site_indices)

    def cluster(self, candidate_ids: list[int], positions_3d: np.ndarray, sequences: list[str],
                all_possible_kmers: list[str], rated_kmer_dict: dict[str, dict[str, int]] = None) -> dict[int: list[int]]:
        """
        Cluster the given
        """

        if self.optics is None:
            raise Exception("Run the initiate_optics method first.")

        # prepare a list of accepted clusters, each element in the list is a dict mapping a protein id in the db
        # to the list of positions in the given cluster
        # TODO: Change the dict to a tuple?
        accepted_clusters: list[dict[int, list[int]]] = []

        for candidate in candidate_ids: # tqdm(candidate_ids, desc='Clustering'):

            if self.top_k is None:
                # find valid positions
                possible_positions = []
                for pos, kmer in enumerate(generate_kmers(sequences[candidate])):
                    if kmer in all_possible_kmers:
                        possible_positions.append(pos)
            else:
                # best k version - for every kmer in site, select K most similar (avoid "rich" hits with many similarities)
                generated = generate_kmers(sequences[candidate])
                possible_positions = []
                for km in rated_kmer_dict.keys():
                    hits = []
                    for i, one in enumerate(generated):
                        if one in rated_kmer_dict[km].keys():
                            hits.append((i, rated_kmer_dict[km][one]))
                    for o in sorted(hits, key=lambda x: x[1], reverse=True)[:self.top_k]:
                        possible_positions.append(o[0])

            if len(possible_positions) <= max(3, int(self.site_size * self.min_kmer_in_cavity_fraction)):
                continue

            labels = self.optics.fit_predict(positions_3d[candidate][possible_positions])


            clusters = {}
            for i in range(len(possible_positions)):
                # skip unclustered
                if labels[i] == -1:
                    continue

                if labels[i] in clusters:
                    clusters[labels[i]].append(possible_positions[i])
                else:
                    clusters[labels[i]] = [possible_positions[i]]

            for cluster_i in clusters:

                if len(clusters[cluster_i]) >= self.site_size * self.min_kmer_in_cavity_fraction:

                    accepted_clusters.append({candidate: clusters[cluster_i]})

        return accepted_clusters