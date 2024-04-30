import numpy as np
from KmerGen import score_seq_similarity, get_one_trimer_slice, generate_kmers_around, generate_similar_kmers


class Vectorizer:

    def __init__(self, binding_site: list[int], sequence: str, kmer_sim: int):

        self.kmers = []
        # add only different kmers
        for kmer in generate_kmers_around(sequence, binding_site):

            found_similar = False
            for another_kmer in self.kmers:
                if score_seq_similarity(another_kmer, kmer) >= kmer_sim:
                    found_similar = True
                    break

            if not found_similar:
                self.kmers.append(kmer)

        # expand kmers to all similar
        self.similar_kmers: dict[str, int] = {} # each str points to position in vector
        for i, kmer in enumerate(self.kmers):
            for similar in generate_similar_kmers(kmer, kmer_sim):
                self.similar_kmers[similar] = i

        self.vector_length = len(self.kmers)

    def vectorize(self, cluster: list[int], sequence: str) -> np.ndarray:

        vector = np.zeros(self.vector_length)
        for position in cluster:
            slice = get_one_trimer_slice(sequence, position)
            if slice not in self.similar_kmers:
                print(f"Unexpected trimer missing: {slice}")
            else:
                vector[self.similar_kmers[slice]] += 1

        return vector
