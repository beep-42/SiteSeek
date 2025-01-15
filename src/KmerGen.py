from Bio.Align import substitution_matrices
from Bio.Data.IUPACData import protein_letters
from functools import lru_cache

BLOSUM62 = substitution_matrices.load("BLOSUM62")


def generate_kmers(sequence: str, k: int = 3):

    kmers = []
    for i in range(0, len(sequence)):
        kmers.append(get_one_trimer_slice(sequence, i))
    return kmers

    """
    kmers = []
    for i in range(0, len(sequence) - k + 1):

        kmers.append(sequence[i:i+k])

    return kmers
    """


def get_one_trimer(sequence: str, pos: int):
    # ABCDEF
    #   |
    #  BCD
    # 1:  :4
    return sequence[max(0, pos - 2):max(0, pos - 2) + 3]


def get_one_trimer_slice(arr, pos: int):
    """
    Returns slice from arr of length k taking care of beginnings and ends.
    :param arr:
    :param pos:
    :param k:
    :return:
    """
    return arr[max(0, min(pos + 1, len(arr)) - 3):min(max(0, pos - 2) + 3, len(arr))]

def get_kmer_slice(arr, pos: int, k: int):

    l = (k - 1) // 2
    if pos - l < 0:
        return arr[0:k]
    elif pos + l > len(arr):
        return arr[-k:]
    else:
        return arr[pos - l:pos + l + 1]


def generate_kmers_around(sequence: str, pivots: list[int], k: int = 3) -> list[str]:
    """
    Generates kmers from string of AAs,
    :param sequence:
    :param k: length of each kmer, must be odd and greater than one
    :param pivots:
    :return:
    """

    assert k % 2 == 1   # k must be odd

    kmers = []

    # length = int((k - 1) / 2)
    #
    # for pivot in pivots:
    #
    #     if pivot + length >= len(sequence):
    #         kmers.append(sequence[
    #             len(sequence) - k:len(sequence)
    #         ])
    #     elif pivot - length < 0:
    #         kmers.append(sequence[0:k])
    #     else:
    #         kmers.append(sequence[pivot - length:pivot + length + 1])

    for pivot in pivots:
        kmers.append(get_one_trimer_slice(sequence, pivot))

    return kmers


def k_mer_around(sequence: str, pivot: int, k: int = 3) -> str:
    assert k % 2 == 1

    if pivot + (k - 1)//2 >= len(sequence):
        return sequence[-k:-1]

    elif pivot - k < 0:
        return sequence[0:k]

    else:
        return sequence[pivot - (k - 1)//2:pivot + (k - 1)//2]


def generate_similar_kmers(kmer: str, threshold: float, scoring_matrix=BLOSUM62) -> list[str]:

    def score(seq: str) -> float:
        nonlocal scoring_matrix, kmer

        s = 0
        for i, let in enumerate(seq):
            s += scoring_matrix[let, kmer[i]]

        return s

    def generate(kmers: list[str], pos: int = 0) -> list[str]:

        nonlocal score, threshold

        if len(kmers) == 0:
            return []

        next_round = []
        for one in kmers:

            for res in protein_letters:

                new = one[:pos] + res + one[pos + 1:]

                if score(new) >= threshold:
                    next_round.append(new)

        if pos >= len(kmers[0]) - 1:
            return next_round
        return generate(next_round, pos+1)

    return generate([kmer])


@lru_cache(maxsize=None)
def generate_similar_kmers_around(sequence: str, pivots: list[int], threshold: float, k: int = 3,
                                  scoring_matrix=BLOSUM62) -> (tuple[str], dict[str, list[str]], dict[str, dict[str, int]]):

    kmer_list = list()
    all_kmers = {}
    all_kmers_rated = {}
    for kmer in generate_kmers_around(sequence, pivots, k=k):
        all_kmers[kmer] = set(generate_similar_kmers(kmer, threshold=threshold, scoring_matrix=scoring_matrix))
        kmer_list.append(kmer)

        all_kmers_rated[kmer] = dict()
        for one in all_kmers[kmer]:
            all_kmers_rated[kmer][one] = score_seq_similarity(one, kmer)

    return tuple(kmer_list), all_kmers, all_kmers_rated


@lru_cache(maxsize=None)
def score_seq_similarity(seq1: str, seq2: str, scoring_matrix=BLOSUM62) -> float:
    s = 0
    for i in range(min(len(seq1), len(seq2))):
        s += scoring_matrix[seq1[i], seq2[i]]

    return s


def rotate_kmer(seq: str, rotation: int) -> str:

    return seq[rotation:] + seq[:rotation]


def compute_mapping_similarity_score(mapping: dict[int, int], seq1: str, seq2: str):
        """
        Computes similarity of the two sequences between position specified by the mapping using BLOSUM62 scoring table.
        :param mapping: mapping between the two sequences.
        :param seq1:
        :param seq2:
        :return:
        """
        score = 0
        for kmer1, kmer2 in mapping.items():
            score += score_seq_similarity(get_one_trimer_slice(seq1, kmer1), get_one_trimer_slice(seq2, kmer2))

        return score