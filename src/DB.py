import RandomConsensusAligner
import TetraHedronAligner
from Clustering import Clustering
from ICP import ICP
from KMerFilter import KMerFilter
from KmerGen import *
from Bio import PDB
import numpy as np
from Bio.PDB import Structure
from Helpers import d3to1
from Bio.PDB.Superimposer import SVDSuperimposer
from collections import defaultdict
from scipy.spatial import KDTree
import random

from tqdm import tqdm

from MostCommonMappingAligner import MostCommonMappingAligner
from Scoring import score_hit


class DB:
    """
    Database class for storing and searching similar sites.
    """

    def __init__(self) -> None:
        self.db = {}
        self.ids = []
        self.pdb_code_to_index = {}
        self.seqs = []
        self.pos = []
        self.kdtrees = []

    def add(self, name: str, seq: str, struct: PDB.Structure):
        """
        Add the given entry to the database.
        :param name: ID of the structures
        :param seq: Sequence of the structure
        :param struct: Bio.PDB.Structure containing the given structure
        """

        index = len(self.seqs)
        self.ids.append(name)
        self.seqs.append(seq)

        kmers = generate_kmers(seq)
        for km in kmers:

            if km in self.db.keys():

                self.db[km].append(index)

            else:

                self.db[km] = [index]

        self.pos.append(self._structure_to_pos(struct))
        self.pdb_code_to_index[name] = len(self.ids) - 1

        # build and index kd-tree
        tree = KDTree(self.pos[-1])
        self.kdtrees.append(tree)

    @staticmethod
    def _structure_to_pos(structure):
        """Converts pdb structure to array of positions"""
        positions = []

        for res in structure.get_residues():
            if res.resname in d3to1:
                positions.append(res.center_of_mass())

        positions = np.array(positions)

        return positions

    def get_all_similar_kmers(self, all_seqs, template_positions, site, all_kmers, cavity_scale_error,
                              min_kmer_in_cavity_fraction) -> dict[int: list[int]]:
        accepted_clusters = []  # dict: seq_id => list[pos: int]
        for hit in all_seqs:
            c = []
            for pos, kmer in enumerate(generate_kmers(self.seqs[hit])):
                if kmer in all_kmers:
                    c.append(pos)
            accepted_clusters.append({hit: c})

        return accepted_clusters

    @staticmethod
    def _find_potential_matches(positions: list[int],
                                seq1: str, seq2: str,
                                k_mer_similarity_threshold: int,
                                kmers: list[str],
                                add_all_points=True) -> defaultdict[int, list[int]]:

        """
        Generates non-redundant dictionary, that maps every position from positions on the first sequence, to positions
        on the second, provided the kmers around these positions are at least as similar as k_mer_similarity_threshold.
        :param positions: Positions on the first sequence to match
        :param seq1:
        :param seq2:
        :param k_mer_similarity_threshold: Minimal required similarity
        :param kmers: All kmers generated from the second sequence
        :param add_all_points: Whether to match only center points, or all points from the kmer (this trimer
        implementation matches all three)

        :return: Mapping of the points.
        """

        potential_matches = defaultdict(list)
        for res_pos in positions:

            kmer = get_one_trimer_slice(seq1, res_pos)  # kmer around the position

            for ref_pos, ref_kmer in enumerate(kmers):
                if score_seq_similarity(kmer, ref_kmer) >= k_mer_similarity_threshold:

                    if add_all_points:

                        for shift in [-1, 0, 1]:

                            if 0 <= res_pos + shift < len(seq1) and 0 <= ref_pos + shift < len(seq2):
                                potential_matches[res_pos + shift].append(ref_pos + shift)

                    else:

                        potential_matches[res_pos + shift].append(ref_pos + shift)

        # remove redundancies
        for m in list(potential_matches.keys()):
            potential_matches[m] = list(set(potential_matches[m]))

        return potential_matches

    def kmer_cluster_svd_search(self, template_positions: list[np.ndarray],
                                seq: str,
                                site: list[int],
                                k_mer_min_found_fraction: float,
                                k_mer_similarity_threshold: float,
                                cavity_scale_error: float,
                                min_kmer_in_cavity_fraction: float,
                                icp_rounds: int):

        kmers_dict = generate_similar_kmers_around(seq, site, k_mer_similarity_threshold)
        all_kmers = []
        for kmer_key in kmers_dict:
            all_kmers += kmers_dict[kmer_key]

        # PART 1: K-mer filtering
        all_seq, kmer_found = KMerFilter.search_kmers(kmers_dict, k_mer_min_found_fraction)

        # PART 2: CLUSTERING
        accepted_clusters = Clustering.optics(all_seq, self.pos, self.seqs, self.ids, template_positions, site,
                                              all_kmers, cavity_scale_error, min_kmer_in_cavity_fraction,
                                              graphics=False)

        # PART 3: MAPPING
        kmers = generate_kmers(seq)
        successful_hits = []

        RCAligner = RandomConsensusAligner.RandomConsensusAligner(template_positions)

        for cluster in tqdm(accepted_clusters, desc='Evaluating clusters'):

            hit = list(cluster.keys())[0]  # seq. index

            positions = cluster[hit]

            potential_matches = DB._find_potential_matches(positions, self.seqs[hit], seq, k_mer_similarity_threshold,
                                                           kmers, add_all_points=True)

            # filter to have at least min_kmer_in_cavity_fraction
            unique_found_positions = set([item for sublist in potential_matches.values() for item in sublist])
            if len(unique_found_positions) <= len(site) * min_kmer_in_cavity_fraction:
                continue

            # rmsd, rot, trans, mapping, rounds, votes, avg_dev, rms_dev, med_dev = SuitableTriangleExpansion.get_rottrans_and_mapping(potential_matches,
            #                                                                                                             self.pos[hit],
            #                                                                                                             template_positions,
            #                                                                                                             rmsd_cutoff = 1.8,
            #                                                                                                             max_dev = 0.05,
            #                                                                                                             rounds = 1500,
            #                                                                                                             min_votes=10000
            #                                                                                                             )

            rmsd, rot, trans, mapping, rounds, votes, avg_dev, rms_dev, med_dev = RCAligner.match_to_source(
                potential_matches, self.pos[hit], allowed_error=.15, rounds=100, stop_at=5)

            if rot is None:  # no hit
                continue

            for _ in range(icp_rounds):
                if mapping is not None:
                    mapping, rmsd, rot, trans = ICP.get_closest_mappings(potential_matches,
                                                                         self.pos[hit],
                                                                         template_positions,
                                                                         rot, trans,
                                                                         cutoff=10)
            if mapping is None:
                continue

            score = self.compute_mapping_similarity_score(mapping, self.seqs[hit], seq)

            successful_hits.append({
                'hit_id': self.ids[hit],
                'found kmers': kmer_found[hit],
                'found kmer frac': kmer_found[hit] / len(site),
                'similarity score': score,
                'rmsd': rmsd,
                'coverage': len(mapping) / len(site),
                'mapping': mapping,
                'cluster': cluster[hit],
                'translation': trans,
                'rotation': rot,
                'triangle rounds': rounds,
                'triangle votes': votes,
                'avg dev': avg_dev,
                'rms dev': rms_dev,
                'med dev': med_dev
            })

        return sorted(sorted(successful_hits, key=lambda x: x['rmsd']), key=lambda x: x['similarity score'],
                      reverse=True)

    def score_list(self, template_positions: list[np.ndarray],
                   seq: str,
                   site: list[int],
                   k_mer_min_found_fraction: float,
                   k_mer_similarity_threshold: float,
                   cavity_scale_error: float,
                   min_kmer_in_cavity_fraction: float,
                   align_rounds: int,
                   align_sufficient_samples: int,
                   align_delta: float,
                   icp_rounds: int,
                   icp_cutoff: float,
                   ids_to_score: list[str]) -> defaultdict[dict]:

        results = defaultdict(dict)
        for sid in ids_to_score:
            results[sid]['score'] = - 1

        kmers_dict = generate_similar_kmers_around(seq, site, k_mer_similarity_threshold)
        all_kmers = []
        for kmer_key in kmers_dict:
            all_kmers += kmers_dict[kmer_key]

        # PART 1: K-mer filtering
        all_seq, kmer_found = KMerFilter.search_kmers(self, kmers_dict, k_mer_min_found_fraction,
                                                      [self.pdb_code_to_index[x] for x in ids_to_score])

        # PART 2: CLUSTERING
        accepted_clusters = Clustering.optics(all_seq, self.pos, self.seqs, self.ids, template_positions, site,
                                              all_kmers, cavity_scale_error, min_kmer_in_cavity_fraction,
                                              graphics=False)

        # PART 3: MAPPING
        kmers = generate_kmers(seq)

        RCAligner = RandomConsensusAligner.RandomConsensusAligner(template_positions)

        for cluster in tqdm(accepted_clusters, desc='Evaluating clusters'):

            hit = list(cluster.keys())[0]  # seq. index

            positions = cluster[hit]

            potential_matches = DB._find_potential_matches(positions, self.seqs[hit], seq, k_mer_similarity_threshold,
                                                           kmers, add_all_points=True)

            # filter to have at least min_kmer_in_cavity_fraction
            unique_found_positions = set([item for sublist in potential_matches.values() for item in sublist])
            if len(unique_found_positions) <= len(site) * min_kmer_in_cavity_fraction:
                continue

            rmsd, rot, trans, mapping, rounds, votes, avg_dev, rms_dev, med_dev = RCAligner.match_to_source(
                potential_matches, self.pos[hit], allowed_error=align_delta, rounds=align_rounds, stop_at=align_sufficient_samples)

            # rmsd, rot, trans, mapping, rounds, votes, avg_dev, rms_dev, med_dev = MostCommonMappingAligner.get_rottrans_and_mapping(potential_matches,
            #                                                                                                             self.pos[hit],
            #                                                                                                             template_positions,
            #                                                                                                             rmsd_cutoff = 1.5,
            #                                                                                                             max_dev = 0.15,
            #                                                                                                             rounds = 1000,
            #                                                                                                             min_votes=15
            #                                                                                                             )

            if rot is None:  # no hit
                continue

            for _ in range(icp_rounds):
                if mapping is not None:
                    mapping, rmsd, rot, trans = ICP.align_closest_mappings(potential_matches,
                                                                           self.pos[hit],
                                                                           template_positions,
                                                                           rot, trans,
                                                                           cutoff=icp_cutoff)
            if mapping is None:
                continue

            found_kmer_fraction = kmer_found[hit] / len(site)
            mapping_coverage = len(mapping) / len(site)
            votes_ratio = votes / len(site)

            s = score_hit(found_kmer_fraction, mapping_coverage, votes_ratio, self.kdtrees[hit],
                          self.pos[hit], template_positions, self.seqs[hit], seq, site, mapping,
                          rot, trans)

            if results[self.ids[hit]]['score'] < s:
                results[self.ids[hit]]['score'] = s
                results[self.ids[hit]]['rot'] = rot
                results[self.ids[hit]]['trans'] = trans
                results[self.ids[hit]]['mapping'] = mapping

        return results

    def get_sequence(self, db_id):
        return self.seqs[db_id]
