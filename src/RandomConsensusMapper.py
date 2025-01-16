import random
import numpy as np
from collections import defaultdict
from Bio.PDB.Superimposer import SVDSuperimposer
from abc import ABC, abstractmethod

import line_profiler


class BaseMapper(ABC):

    @abstractmethod
    def set_source(self, source_to_match_positions):
        pass

    @abstractmethod
    def match_to_source(self, possible_matches: dict[int, list[int]], hit_positions: np.ndarray) -> (
        float, np.ndarray, np.ndarray, dict[int, int], int
    ):
        pass


class DistanceMatrix:

    """
    Distance matrix class that computes only requested distances and stores them.

    Attributes:
        source_positions: a numpy array of the source positions.
        source_dist_matrix: a numpy matrix with zeros on positions without the calculated distances and the distances
        otherwise (always upper triangular).
    """

    def __init__(self, source_positions: np.ndarray, max_size: int = 2000) -> None:
        """
        Initializes the distance matrix (does not calculate any positions, only allocates the distance matrix).

        Args:
            source_positions: a numpy array of the source positions.
            max_size: the maximum length of the source_positions to consider caching. For longer structures
            all distances are computed.
        """

        self.max_size = max_size
        self.source_positions = source_positions
        if max(source_positions.shape) <= self.max_size:
            self.source_dist_matrix = np.zeros((len(source_positions), len(source_positions)), dtype=np.float64)
            self.direct_compute = False
        else:
            self.direct_compute = True

    def get_dist(self, i, j) -> np.float64:
        """
        Gets the distance between the i-th and j-th position in the source (calculates it if it was not calculated yet).

        Args:
            i: the index of the first position in the source.
            j: the index of the second position in the source.

        Returns:
            float: the distance between the positions.
        """

        if self.direct_compute:
            return np.linalg.norm(self.source_positions[i] - self.source_positions[j])

        i, j = min(i, j), max(i, j)
        if i == j: return 0
        if self.source_dist_matrix[i, j] == 0:
            self.source_dist_matrix[i, j] = np.linalg.norm(self.source_positions[i] - self.source_positions[j])

        return self.source_dist_matrix[i, j]

    def __getitem__(self, p):
        """
        Wrapper for the get_dist function.
        """

        i, j = p
        return self.get_dist(i, j)


class RandomConsensusMapper(BaseMapper):
    """
    Aligns the cavities based on the random consensus algorithm. Picks a triangle and expands it, the largest expanded
    set wins.

    The distances are calculated on demand and stored in an otherwise empty distance matrix. Distances in the hit
    structure are stored only for the duration of the match_to_source function and forgotten afterward.

    Attributes:
        allowed_error: the maximum allowed relative error between the distances of points in the target and source
        polygon to consider this a matching polygons (normalized to the source distance). Also used during the
        sample expansion to evaluate possible expansions.
        rounds: the number of random samples to pick.
        stop_at: the maximum number of successful rounds (rounds where the picked sample matched).
        polygon: how many points constitute the picked sample (min. recommended is 3).

    Methods:
        match_to_source(): Performs the RanSaC algorithm on the given data.

    """

    def __init__(self, allowed_error: float = 2.0, rounds: int = 30,
                 stop_at: int = 30, polygon: int = 3) -> None:
        """
        Initializes the random consensus aligner.

        Args:
            allowed_error: the maximum allowed relative error between the distances of points in the target and source
            polygon to consider this a matching polygons (normalized to the source distance). Also used during the
            sample expansion to evaluate possible expansions.
            rounds: the number of random samples to pick.
            stop_at: the maximum number of successful rounds (rounds where the picked sample matched).
            polygon: how many points constitute the picked sample (min. recommended is 3).
        """

        self.allowed_error = allowed_error
        self.rounds = rounds
        self.stop_at = stop_at
        self.polygon = polygon

        # self.source_dist_matrix = np.zeros((len(source_to_match_positions), len(source_to_match_positions)))
        self.sup_ = SVDSuperimposer()
        self.source_positions_ = None
        self.source_dist_matrix_ = None

    def set_source(self, source_to_match_positions):
        """
        Set the source for the mapping.

        Args:
            source_to_match_positions: a numpy array of the source positions.
        """

        self.source_positions_ = source_to_match_positions
        self.source_dist_matrix_ = DistanceMatrix(source_to_match_positions)

    @line_profiler.profile
    def match_to_source(self, possible_matches: dict[int, list[int]], hit_positions: np.ndarray) -> (
        float, np.ndarray, np.ndarray, dict[int, int], int
    ):
        """
        Run the Random Sample Consensus algorithm to find the best mapping from the possible matches.

        Args:
            possible_matches: A dict of all template K-mer sequential positions and sequential positions of
         K-mers in the target sequence with sufficient similarity (allowed mappings).
            hit_positions: A numpy array of the 3D positions of the aminoacids in the target structures.

        Returns:
            A tuple of RMSD of superposition of the best found mapping, a right multiplying rotation matrix resulting
            from the superposition, translation vector from the superposition, the best found alignment (a dictionary
            of sequential position from the query to the target) and the number of found matching positions.
        """
        if self.source_positions_ is None:
            raise Exception('RandomConsensusAligner: Set the source first!')

        expansions = []

        hit_matches = list(possible_matches.keys())
        hit_dist_matrix = DistanceMatrix(hit_positions)

        for r in range(self.rounds):

            triangle = random.choices(hit_matches, k=self.polygon)
            mapped_triangle = [random.choice(possible_matches[triangle[p]]) for p in range(self.polygon)]

            if len(set(mapped_triangle)) != len(mapped_triangle):   # skip redundant mappings
                continue

            m = sum([
                (
                    hit_dist_matrix[triangle[i], triangle[(i + 1) % self.polygon]] - # np.linalg.norm(hit_positions[triangle[i]] - hit_positions[triangle[(i + 1) % polygon]]) -
                    self.source_dist_matrix_[mapped_triangle[i], mapped_triangle[(i + 1) % self.polygon]]
                ) / self.source_dist_matrix_[mapped_triangle[i], mapped_triangle[(i + 1) % self.polygon]]
            for i in range(self.polygon)]) / self.polygon

            if m <= self.allowed_error:

                expansions.append(
                    self._expand_triangle(zip(triangle, mapped_triangle), possible_matches, hit_dist_matrix)
                )

                if len(expansions) > self.stop_at:
                    break

        if not len(expansions):
            return None, None, None, None, None

        best = sorted(expansions, key=lambda x: len(x), reverse=True)[0]

        if len(best) < 3:
            return None, None, None, None, None

        x = [self.source_positions_[one] for one in best.values()]
        y = [hit_positions[one] for one in best.keys()]

        self.sup_.set(np.array(x).astype(np.float32), np.array(y).astype(np.float32))
        self.sup_.run()

        return self.sup_.get_rms(), *self.sup_.get_rotran(), best, len(best)

    @line_profiler.profile
    def _expand_triangle(self, triangle: list[list[int, int]], possible_matches, hit_dist_matrix) -> [float, float]:
        """
        Tries to add all possible matches spatially constrained by the given triangle.
        :param triangle:
        :param possible_matches:
        :param hit_dist_matrix:
        :return: The contrained matches.
        """

        def option_spatially_correct(seq_position, template_seq_position):
            nonlocal self
            # check distance of the mapping to all points in the triangle
            md = 0
            for point, point_in_template in triangle:
                if self.source_dist_matrix_[template_seq_position, point_in_template] == 0: return False, 0

                d = (
                        hit_dist_matrix[seq_position, point] - # np.linalg.norm(positions[seq_position] - point) -
                        self.source_dist_matrix_[template_seq_position, point_in_template]
                ) / self.source_dist_matrix_[template_seq_position, point_in_template]
                if d > self.allowed_error:
                    return False, 0

                md += d # max(md, d)

            mean_d = md / self.polygon
            if mean_d > self.allowed_error:
                return False, 0

            return True, mean_d

        constrained_matches = defaultdict(list)
        for seq_position in possible_matches:
            for template_seq_position in possible_matches[seq_position]:
                correct, mean_dev = option_spatially_correct(seq_position, template_seq_position)
                if correct:
                    if seq_position in constrained_matches:
                        # in case of conflicts, keep the mapping with lower mean_dev
                        if constrained_matches[seq_position][1] > mean_dev:
                            constrained_matches[seq_position] = [template_seq_position, mean_dev]
                    else:
                        constrained_matches[seq_position] = [template_seq_position, mean_dev]

        matches = {}
        for match in constrained_matches:
            matches[match] = constrained_matches[match][0]

        return matches
