from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BaseKMerFilter (ABC):

    @abstractmethod
    def search_kmers(self, db: object, kmer_tuple: Tuple[str], kmers_dict: Dict[str, List[str]], rated_kmers_dict: Dict[str, Dict[str, int]], \
                     use_subset: Optional[List[int]] = None) \
        -> (List[int], Dict[int, int]):
        """
        Filters the sequences in the database.

        :param db: The database object.
        :param kmer_tuple: Tuple containing all the site K-mers.
        :param kmers_dict: The dictionary containing the searched kmers alternatives
        :param rated_kmers_dict: The dictionary mapping Kmers to their all their allowed alternatives and their scores.
        :param use_subset: The subset of the Database ids to consider. None means all ids are considered.

        Return:
            List[int]: all sequence IDs that passed the filter.
            Dict[int, int]: contains the number of parent K-mers from the kmers_dict that are
            in the sequence for each passed sequence
        """
        pass
