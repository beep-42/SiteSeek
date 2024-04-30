import random
import numpy as np
from collections import defaultdict
from Bio.PDB.Superimposer import SVDSuperimposer


class RandomConsensusAligner:
    """
    Aligns the cavities based on the random consensus algorithm. Picks a triangle and expands it, the largest expanded
    set wins.
    """

    def __init__(self, source_to_match_positions):

        self.source_dist_matrix = np.zeros((len(source_to_match_positions), len(source_to_match_positions)))
        self.sup = SVDSuperimposer()
        self.source_positions = source_to_match_positions

        for i in range(len(source_to_match_positions)):
            for j in range(len(source_to_match_positions)):
                self.source_dist_matrix[i, j] = np.linalg.norm(source_to_match_positions[i] - source_to_match_positions[j])

    def match_to_source(self, possible_matches, hit_positions, allowed_error = 2, rounds = 30, stop_at = 6, polygon = 3):

        expansions = []

        hit_matches = list(possible_matches.keys())
        for r in range(rounds):

            triangle = random.choices(hit_matches, k=polygon)
            mapped_triangle = [random.choice(possible_matches[triangle[p]]) for p in range(polygon)]

            if len(set(mapped_triangle)) != len(mapped_triangle):   # skip redundant mappings
                continue

            m = sum([
                (
                    np.linalg.norm(hit_positions[triangle[i]] - hit_positions[triangle[(i + 1) % polygon]]) -
                    self.source_dist_matrix[mapped_triangle[i], mapped_triangle[(i + 1) % polygon]]
                ) / self.source_dist_matrix[mapped_triangle[i], mapped_triangle[(i + 1) % polygon]]
            for i in range(polygon)]) / polygon

            if m <= allowed_error:

                expansions.append(
                    self._expand_triangle(zip(triangle, mapped_triangle), allowed_error, possible_matches, hit_positions, polygon)
                )

                if len(expansions) > stop_at:
                    break

        if not len(expansions):
            return None, None, None, None, None, None, None, None, None

        best = sorted(expansions, key=lambda x: len(x), reverse=True)[0]
        # if [len(x) for x in expansions].count(len(best)) > 1:
        #     print(best)
        #     print(sorted(expansions, key=lambda x: len(x), reverse=True)[1])
        # print("Sizes: ", [len(x) for x in expansions])

        if len(best) < 3:
            return None, None, None, None, None, None, None, None, None

        x = [self.source_positions[one] for one in best.values()]
        y = [hit_positions[one] for one in best.keys()]

        # print(np.array(x))
        # print(np.array(y))

        self.sup.set(np.array(x), np.array(y))
        self.sup.run()

        return self.sup.get_rms(), *self.sup.get_rotran(), best, r, len(best), 0, 0, 0

    def _expand_triangle(self, triangle: list[list[int, int]], allowed_dist_error: float, possible_matches, positions, polygon) -> [float, float]:
        """
        Tries to add all possible matches spatially constrained by the given triangle.
        :param triangle:
        :param allowed_dist_error:
        :param possible_matches:
        :param positions:
        :param template_positions:
        :return: RMSD, coverage (fraction)
        """

        def option_spatially_correct(seq_position, template_seq_position):
            nonlocal allowed_dist_error, triangle
            # check distance of the mapping to all points in the triangle
            md = 0
            for point, point_in_template in triangle:
                if self.source_dist_matrix[template_seq_position, point_in_template] == 0: return False, 0

                d = (
                        np.linalg.norm(positions[seq_position] - point) -
                        self.source_dist_matrix[template_seq_position, point_in_template]
                ) / self.source_dist_matrix[template_seq_position, point_in_template]
                if d > allowed_dist_error:
                    return False, 0

                md += d # max(md, d)

            mean_d = md / polygon
            if mean_d > allowed_dist_error:
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
