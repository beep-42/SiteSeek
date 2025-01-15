from typing import List, Dict, Tuple

import numpy as np

from BaseKMerFilter import BaseKMerFilter


class HardKMerFilter(BaseKMerFilter):
    """
    A class to filter all the database sequences based on occurrences of K-mers similar to the given K-mers from the
    binding site. Sequences with more than k_mer_min_found_fraction are considered positive - hard filtering.

    This implementation requires a database class with a precalculated map of K-mer -> sequences containing the given
    K-mer. The filtering is done in O(nk) where n is the number of searched K-mers and k is upper bound of
    the number of sequences containing one of the search K-mers.

    Attributes:
        k_mer_min_found_fraction (float): the minimum fraction of similar K-mers from the searched cavity to contain
        to be considered a positive hit. Applied independently across all K-mers. The intersection of all searches
        is subsequently considered positive.

    Methods:
        search_kmers(): Given the Database return all ids whose sequences contain at least one similar K-mer to
        at least k_mer_min_found_fraction of key K-mers in the k_mers dictionary.
    """

    def __init__(self, k_mer_min_found_fraction) -> None:
        """
        Initialize the KMerFilter object.

        Args:
            k_mer_min_found_fraction(float): the minimum fraction of similar K-mers in the target sequences to consider
            it as a positive hit.
        """
        self.k_mer_min_found_fraction = k_mer_min_found_fraction


    def search_kmers(self, db: object, kmer_tuple: Tuple[str], kmers_dict: Dict[str, List[str]], rated_kmers_dict: Dict[str, Dict[str, int]], use_subset: List[int] = None)\
            -> (List[int], Dict[int, int]):
        """
        Searches all sequences in the given Database. Each one needs to contain at least one similar K-mer to every key
        k-mer in the kmers dictionary to be considered a positive hit.

        Args:
            db (Database): the Database object.
            :param kmer_tuple (Tuple[str]): the Tuple of K-mer tuples. Not used.
            kmers_dict (dict[str, list[str]]): the dictionary mapping cavity K-mers to all similar alternatives.
            :param rated_kmers_dict: Not used, for API reasons.
            use_subset (list[int]): if supplied it specifies only the subset of the database to be searched.
        """

        all_seqs = {x: 0 for x in range(len(db.sequences))} if use_subset is None else {x: 0 for x in use_subset}

        max_incorrect = len(kmers_dict) - int(self.k_mer_min_found_fraction * len(kmers_dict))

        for i, km in enumerate(kmers_dict):  # find all that have at least threshold% of k-mers

            found = set()
            for one in kmers_dict[km]:
                #if one in db.kmer_db:

                    for s in all_seqs:

                        if s in db.kmer_db[one] and s not in found:
                        # if np.sum(np.equal(db.kmer_db[one], s)) == 1 and s not in found:
                            all_seqs[s] += 1
                            found.add(s)

            if i >= max_incorrect:

                new = {}
                for one in all_seqs:
                    if i - all_seqs[one] <= max_incorrect:
                        new[one] = all_seqs[one]

                all_seqs = new

        all_seq_keys = list(all_seqs.keys())

        return all_seq_keys, all_seqs
