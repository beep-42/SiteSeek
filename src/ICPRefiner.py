import numpy as np
from Bio.PDB.Superimposer import SVDSuperimposer
from BaseRefiner import BaseRefiner


class ICP(BaseRefiner):
    """
    Iteratively aligns the given points for the given number of rounds. Additionally, the similar K-mers closer than
    the cutoff are also mapped on each other.

    Attributes:
        cutoff (float): All similar K-mers closer than cutoff (in Euclidean distance) are mapped on each other.
        rounds (int): How many rounds to perform (stops if no K-mers are mapped).

    Methods:
        align_closest_mappings(): Run the ICP algorith with the close similar K-mers extension.

    """


    def __init__(self, cutoff: float or None, rounds: int) -> None:
        """
        Initializes the ICP object.

        Args:
              cutoff(float): The cutoff distance in Angstroms.
              round(int): The number of required ICP rounds.
        """
        self.cutoff = cutoff
        self.rounds = rounds
        self.svd_ = SVDSuperimposer()


    def refine(self, possible_matches: dict[int, list[int]], positions: np.ndarray,
                               template_positions: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> (
            dict[int, int], float, np.ndarray, np.ndarray
    ):
        """
        Superimposes the structures using the provided rot & translation and returns the closest mapping from
        possible_matches for the predefined number of rounds.

        :param possible_matches: A dict of all template K-mer sequential positions and sequential positions of
         K-mers in the target sequence with sufficient similarity (allowed mappings).
        :param positions: Numpy array containing the 3D positions of each amino acid in the target in a sequential
        order.
        :param template_positions: Numpy array containing the 3D positions of each amino acid in the query protein in
        a sequential order.
        :param rot: A numpy rotation matrix (right multiplying).
        :param trans: A numpy translational vector.
        :return: A tuple containing: the dictionary of mapped positions from the query to the target, RMSD resulting
        from the superposition based on the mapping (superposing only the mapped points), right multiplying rotation
        matrix and translational vector.
        """
        for _ in range(self.rounds):
            if rot is not None:
                mapping, rmsd, rot, trans = self.align_closest_mappings(possible_matches, positions, template_positions,
                                                                        rot, trans)
            else: return None, None, None, None

        return mapping, rmsd, rot, trans


    def align_closest_mappings(self, possible_matches: dict[int, list[int]], positions: np.ndarray,
                               template_positions: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> (
        dict[int, int], float, np.ndarray, np.ndarray
    ):
        """
        Superimposes the structures using the provided rot & translation and returns the closest mapping from
        possible_matches.
        :param possible_matches: A dict of all template K-mer sequential positions and sequential positions of
         K-mers in the target sequence with sufficient similarity (allowed mappings).
        :param positions: Numpy array containing the 3D positions of each amino acid in the target in a sequential
        order.
        :param template_positions: Numpy array containing the 3D positions of each amino acid in the query protein in
        a sequential order.
        :param rot: A numpy rotation matrix (right multiplying).
        :param trans: A numpy translational vector.
        :return: A tuple containing: the dictionary of mapped positions from the query to the target, RMSD resulting
        from the superposition based on the mapping (superposing only the mapped points), right multiplying rotation
        matrix and translational vector.
        """

        transformed = np.dot(positions, rot) + trans
        mapped: dict[int, int] = {}

        for kmer in possible_matches:

            # with repeating
            dists = [np.linalg.norm(transformed[kmer] - template_positions[x]) for x in possible_matches[kmer]]
            mapping = possible_matches[kmer][np.argmin(dists)]

            # # without repeating
            # dists = [np.linalg.norm(transformed[kmer] - template_positions[x]) if kmer not in mapped else 99999999 for x
            #          in possible_matches[kmer]]
            # mapping = possible_matches[kmer][np.argmin(dists)]

            if self.cutoff is None or min(dists) <= self.cutoff:
                mapped[kmer] = mapping

        x = []
        y = []

        for kmer in mapped:
            y.append(positions[kmer])
            x.append(template_positions[mapped[kmer]])

        if len(x) < 3:
            return None, None, None, None

        self.svd_.set(np.array(x).astype(np.float32), np.array(y).astype(np.float32))   # the superimposer does not work with float16
        self.svd_.run()

        return mapped, self.svd_.get_rms(), *self.svd_.get_rotran()