import numpy as np
from collections import defaultdict
from Bio.PDB.Superimposer import SVDSuperimposer
import random


class MostCommonMappingAligner:
    @staticmethod
    def get_rottrans_and_mapping(possible_matches, positions, template_positions, rmsd_cutoff: float = 4,
                                 max_dev: float = .05, rounds: int = 10, min_votes: int = 10, min_confirmations: int = 1):
        def most_common_mapping(lst):
            counts = defaultdict(lambda: defaultdict(int))

            # Count occurrences of each pair
            for key, value in lst:
                counts[key][value] += 1

            # Find the most common value for each key
            most_common_values = {}
            most_common_keys = {}

            for key, value_counts in counts.items():
                most_common_value = max(value_counts, key=value_counts.get)
                if counts[key][most_common_value] < min_confirmations:
                    continue

                if most_common_value in most_common_keys:
                    other = most_common_keys[most_common_value]
                    if counts[other][most_common_value] < counts[key][most_common_value]:
                        del most_common_values[other]
                        del most_common_keys[most_common_value]
                        most_common_values[key] = most_common_value
                        most_common_keys[most_common_value] = key

                else:
                    most_common_values[key] = most_common_value
                    most_common_keys[most_common_value] = key
                # print(f"Key: {key}, value: {most_common_value}, occurrence: {counts[key][most_common_value]}")

            return most_common_values

        def score_mapping_so_far(lst):

            # SLOW, BECAUSE POINTS FOUND IN MULTIPLE TRIANGLES HAVE DISTANCES CALCULATED MULTIPLE TIMES.

            counts = defaultdict(lambda: defaultdict(int))

            # Count occurrences of each pair
            for key, value in lst:
                counts[key][value] += 1

            deviations = []
            for point in counts:

                for equivalent in counts[point]:

                    for other in counts:

                        if other == point: continue

                        for other_equivalent in counts[other]:

                            if equivalent == other_equivalent: continue

                            s = np.linalg.norm(template_positions[equivalent] - template_positions[other_equivalent])
                            deviations += [(np.linalg.norm(positions[point] - positions[other]) - s) / s] * (
                                        counts[point][equivalent] + counts[other][other_equivalent])

            return np.average(np.abs(deviations)), (np.sum(np.power(deviations, 2)) / len(deviations)) ** .5, np.median(
                deviations)
            #
            # deviations = []
            # for point, equivalent in lst:
            #
            #     for other, other_equivalent in lst:
            #
            #         if point != other and equivalent != other_equivalent:
            #
            #             s = np.linalg.norm(template_positions[equivalent] - template_positions[other_equivalent])
            #             deviations.append(np.linalg.norm((positions[point] - positions[other])) / s)
            #
            # return np.average(np.abs(deviations)), (np.sum(np.power(deviations, 2)) / len(deviations)) ** .5, np.median(deviations)

        min_dist = 5  # minimum distance between the triangle vertices

        svd = SVDSuperimposer()

        total_votes = 0
        mappings: list[list[int:'hit kmer', int:'query kmer']] = []

        i = 0
        while total_votes < min_votes and i < rounds:  # for _ in range(rounds):
            i += 1
            # triangle from the hit
            triangle = random.choices(list(possible_matches.keys()), k=3)
            """
            while min((np.linalg.norm(positions[triangle[0]] - positions[triangle[1]]),
                       np.linalg.norm(positions[triangle[1]] - positions[triangle[2]]),
                       np.linalg.norm(positions[triangle[2]] - positions[triangle[0]]))) <= min_dist:
                triangle = random.choices(list(possible_matches.keys()), k=3)
            """

            mapped_triangle = [random.choice(possible_matches[triangle[p]]) for p in range(3)]

            # max_diff = max((
            #                 abs(
            #                         np.linalg.norm(positions[triangle[0]] - positions[triangle[1]]) -
            #                         np.linalg.norm(template_positions[mapped_triangle[0]] - template_positions[mapped_triangle[1]])
            #                 ),
            #                 abs(
            #                         np.linalg.norm(positions[triangle[1]] - positions[triangle[2]]) -
            #                         np.linalg.norm(template_positions[mapped_triangle[1]] - template_positions[mapped_triangle[2]])
            #                 ),
            #                 abs(
            #                         np.linalg.norm(positions[triangle[2]] - positions[triangle[0]]) -
            #                         np.linalg.norm(template_positions[mapped_triangle[2]] - template_positions[mapped_triangle[0]])
            #                 )
            #         ))

            # RMSD BASED SCORING
            # rmsd = (sum([
            #     (
            #             np.linalg.norm(positions[triangle[0]] - positions[triangle[1]]) -
            #             np.linalg.norm(template_positions[mapped_triangle[0]] - template_positions[mapped_triangle[1]])
            #     ) ** 2,
            #     (
            #             np.linalg.norm(positions[triangle[1]] - positions[triangle[2]]) -
            #             np.linalg.norm(template_positions[mapped_triangle[1]] - template_positions[mapped_triangle[2]])
            #     ) ** 2,
            #     (
            #             np.linalg.norm(positions[triangle[2]] - positions[triangle[0]]) -
            #             np.linalg.norm(template_positions[mapped_triangle[2]] - template_positions[mapped_triangle[0]])
            #     ) ** 2
            # ]) / 3) ** .5

            #
            # if rmsd <= rmsd_cutoff:
            #
            #     # add this "vote" to the mapping
            #     total_votes += 1
            #     for vertex in range(3):
            #         mappings.append([triangle[vertex], mapped_triangle[vertex]])
            #
            # FRACTIONAL DEVIATION BASED MAPPING
            found_bigger = False
            for i in range(3):
                if mapped_triangle[i] == mapped_triangle[(i + 1) % 3]:
                    # found_bigger = True
                    continue

                s = np.linalg.norm(template_positions[mapped_triangle[i]] - template_positions[mapped_triangle[(i + 1) % 3]])
                if abs(np.linalg.norm(positions[triangle[i]] - positions[triangle[(i + 1) % 3]]) - s) > max_dev * s:
                    found_bigger = True
                    break

            if not found_bigger:
                total_votes += 1
                for vertex in range(3):
                    mappings.append([triangle[vertex], mapped_triangle[vertex]])

        if total_votes < 3:
            return None, None, None, None, None, None, None, None, None

        # print(f"Votes: {total_votes}")

        mapping = most_common_mapping(mappings)
        if len(mapping) < 3:
            return None, None, None, None, None, None, None, None, None

        x = [template_positions[one] for one in mapping.values()]
        y = [positions[one] for one in mapping.keys()]

        svd.set(np.array(x), np.array(y))
        svd.run()

        # print(mapping)
        # print(svd.get_rms())

        avg_dev, rms_dev, med_dev = score_mapping_so_far(mappings)

        return svd.get_rms(), *svd.get_rotran(), mapping, i, total_votes, avg_dev, rms_dev, med_dev
