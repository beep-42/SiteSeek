import array
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from typing import List
from functools import lru_cache

class KmerMapperNumpy(object):
    """
    This class implements a direct indexing mapping Kmers to identifiers of structures (uint16).
    The ids can be accessed via a str representation of the Kmer (subsequently converted to index).
    """



    def __init__(self, k: int = 3, index_dtype=np.uint32):

        self.k = k
        self.index_type = index_dtype
        self.amino_acid_codes = "ACDEFGHIKLMNPQRSTVWY"
        self.total_amino_acids = len(self.amino_acid_codes)
        self.amino_acid_mappings = {self.amino_acid_codes[i]: i for i in range(self.total_amino_acids)}

        # self.pre_alloc_size = 1024 # replace with doubling
        self.table: List[NDArray] = [np.empty(shape=0, dtype=self.index_type) for _ in range(self.total_amino_acids ** self.k)]

        # remember the length of each pre-alloc array
        self.lengths: np.ndarray = np.full(shape=self.total_amino_acids ** self.k, fill_value=0, dtype=self.index_type)
        # remember the position of last value in each array
        self.next_indices: np.ndarray = np.full(shape=self.total_amino_acids ** self.k, fill_value=0, dtype=self.index_type)

    @lru_cache(maxsize=1024)
    def valid_kmer(self, kmer: str):
        for i in range(len(kmer)):
            if kmer[i] not in self.amino_acid_mappings: return False

        return len(kmer) == self.k

    @lru_cache(maxsize=1024)
    def _kmer2index(self, kmer: str) -> int:

        index = 0
        for i in range(len(kmer)):

            index += self.total_amino_acids ** i + self.amino_acid_mappings[kmer[i]]

        return index

    def __getitem__(self, kmer: str) -> NDArray:
        idx = self._kmer2index(kmer)
        return self.table[idx][:self.next_indices[idx]]

    # Should not be supported
    # def __setitem__(self, kmer: str, value: NDArray):
    #     self.table[self._kmer2index(kmer)] = value
    #     self.lengths[self._kmer2index(kmer)] = len(value)
    #     self.next_indices[self._kmer2index(kmer)] = len(value)

    def append(self, kmer: str, identifier: int) -> None:
        idx = self._kmer2index(kmer)
        # if identifier not in self.table[idx]:
        if np.sum(np.equal(self.table[idx], identifier)) < 1:

            # if we don't have the space for another value - resize (add 0)
            if self.lengths[idx] - self.next_indices[idx] < 1:
                self.table[idx].resize(2 * self.lengths[idx] + 1) # self.pre_alloc_size)
                self.lengths[idx] = 2 * self.lengths[idx] + 1 # self.pre_alloc_size

            # add the identifier
            self.table[idx][self.next_indices[idx]] = identifier
            self.next_indices[idx] += 1

            # self.table[idx] = np.append(self.table[idx], identifier)

    def drop_empty(self) -> None:

        for idx in range(len(self.table)):

            # self.table[idx] = self.table[idx][:self.last_indices[idx]]
            self.table[idx].resize(self.next_indices[idx])
            self.lengths[idx] = self.next_indices[idx]

    def bypass_pickle_view(self) -> None:

        for idx in range(len(self.table)):
            self.table[idx] = self.table[idx].copy()

    def __iter__(self):
        self.__curr = 0
        return self

    def __next__(self) -> np.ndarray:
        # Terminate if range over, otherwise return current, calculate next.

        if self.__curr >= len(self.table):
            raise StopIteration()

        self.__curr += 1

        return self.table[self.__curr][:self.next_indices[self.__curr]]

    def has_any_ids(self, kmer: str) -> bool:

        return self.next_indices[self._kmer2index(kmer)] > 0

class KmerMapperArray(object):

    def __init__(self, dtype = 'I'):
        self.map = defaultdict(lambda: array.array(dtype))
        self.amino_acid_codes = "ACDEFGHIKLMNPQRSTVWY"
        self.amino_acid_mappings = {self.amino_acid_codes[i]: i for i in range(len(self.amino_acid_codes))}

    def valid_kmer(self, kmer: str) -> bool:
        for letter in kmer:
            if letter not in self.amino_acid_mappings: return False
        return True

    def append(self, kmer, index):
        # if index not in self.map[kmer]:
        self.map[kmer].append(index)

    def __getitem__(self, kmer):
        return self.map[kmer]

    # def drop_empty(self):
    #     # for key in self.map:
    #     #     if len(set(self.map[key])) != len(self.map[key]):
    #     #         print("Found a duplicate!!")
    #     # print("Duplication check complete.")
    #     pass

    def has_any_ids(self, kmer: str) -> bool:
        return len(self.map[kmer]) > 0