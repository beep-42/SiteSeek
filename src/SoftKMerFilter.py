from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from KmerGen import score_seq_similarity

import numpy as np

from BaseKMerFilter import BaseKMerFilter


class SoftKMerFilter(BaseKMerFilter):

    def __init__(self, overall_required_similarity: float = 0.9):

        self.overall_required_similarity = overall_required_similarity

    def search_kmers(self, db: object, kmer_tuple: Tuple[str], kmers_dict: Dict[str, List[str]],
                     rated_kmers_dict: Dict[str, Dict[str, int]],
                     use_subset: Optional[List[int]] = None) -> (List[int], Dict[int, int]):

        # Pipeline: Create a np.uint16 array of feasible size, for each K-mer, find hits, add their scores
        # measure self mapping score
        # create a bool map of successful hits, filter the IDs

        # calculate the score of the exact Kmers matching themselves (self-match score)
        self_score: int = sum([score_seq_similarity(kmer, kmer) for kmer in kmer_tuple])# sum([rated_kmers_dict[kmer][kmer] for kmer in kmer_tuple])
        required_score: float = self_score * self.overall_required_similarity

        # initialize the overall score array
        size = len(db.ids) # if use_subset is None else len(use_subset)
        scores = np.zeros(shape=size, dtype=np.uint16)

        # search the entire database
        if use_subset is None:
            for kmer in kmers_dict:
                seen: Dict[int, int] = defaultdict(int)
                for alternative_kmer in kmers_dict[kmer]:
                    if len(db.kmer_db[alternative_kmer]): # db.kmer_db.has_any_ids(alternative_kmer):
                        score = rated_kmers_dict[kmer][alternative_kmer]
                        # if alternative_kmer in db.kmer_db:
                        for id in db.kmer_db[alternative_kmer]:
                            seen[id] = max(seen[id], score)

                # correct for multiple occurrences of the kmer (not reflected in the dictionary)
                count = kmer_tuple.count(kmer)
                # if count > 1:
                #     for id in seen:
                #         seen[id] *= count

                # accumulate the scores
                for id in seen:
                    scores[id] += seen[id] * count

        else:
            reverse_subset: dict[int, int] = {}
            for i in range(len(use_subset)):
                reverse_subset[use_subset[i]] = i

            for kmer in kmers_dict:
                seen: Dict[int, int] = defaultdict(int)
                for alternative_kmer in kmers_dict[kmer]:
                    if len(db.kmer_db[alternative_kmer]): #  db.kmer_db.has_any_ids(alternative_kmer):
                        score = rated_kmers_dict[kmer][alternative_kmer]
                        for id in db.kmer_db[alternative_kmer]:
                            if id in use_subset:
                                seen[id] = max(seen[id], score)

                # correct for multiple occurrences of the kmer (not reflected in the dictionary)
                count = kmer_tuple.count(kmer)
                # if count > 1:
                #     for id in seen:
                #         seen[id] *= count

                # accumulate the scores
                for id in seen:
                    scores[reverse_subset[id]] += seen[id] * count


        # the same for both unconstrained and contrained filtering, later an intersection with the search subset is
        # performed

        # for kmer in kmers_dict:
        #     seen: np.ndarray = np.zeros(len(db), dtype=np.uint16)
        #     for alternative_kmer in kmers_dict[kmer]:
        #         score = rated_kmers_dict[kmer][alternative_kmer]
        #         seen[db.kmer_db[alternative_kmer]] = np.maximum(seen[db.kmer_db[alternative_kmer]], score)
        #     count = kmer_tuple.count(kmer)
        #     seen *= count
        #     scores += seen
        #
        # if use_subset:
        #     # filter out everything except for the ids in use_subset
        #     result = np.zeros_like(scores)
        #     result[use_subset] = scores[use_subset]
        #     scores = result

        hits = scores >= required_score

        hit_list: List[int] = list()
        rated: Dict[int, float] = dict()
        for id in range(len(hits)):
            if hits[id]:
                c_id = id if use_subset is None else use_subset[id]
                hit_list.append(c_id)
                rated[c_id] = scores[id] / self_score

        return hit_list, rated
