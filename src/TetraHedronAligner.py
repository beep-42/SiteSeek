import numpy as np
import random
from Bio.PDB.Superimposer import SVDSuperimposer
from collections import defaultdict


class TetraHedronAligner:

    @staticmethod
    def get_complete_mapping_avg_dev(mapping, positions, template_positions):
        # for each one:
        #   calculate deviation from the template
        # avg the devs

        deviations = []

        for one in mapping:

            for second in mapping:

                if one == second:
                    continue

                if mapping[one] == mapping[second]:
                    continue    # try to bypass collisions, should be reflected in the other deviations?

                s = np.linalg.norm(template_positions[mapping[one]] - template_positions[mapping[second]])
                deviations.append(min((np.linalg.norm(positions[one] - positions[second]) - s) / (s + 0.1), 20))

        if len(deviations) == 0:
            return None, None, None

        return (sum(np.abs(np.array(deviations))) / len(deviations),
                (np.power(np.array(deviations), 2).sum() / len(deviations)) ** .5,
                np.median(deviations))

    @staticmethod
    def get_rottrans_and_mapping(possible_matches, positions, template_positions, rmsd_cutoff: float = 4, rounds: int = 10, min_votes: int = 10, min_confirmations: int = 1, max_deviation = .3):

        def best_mappings(mappings: dict[int:'hit position', dict[int:'query position', int:'score sum']]):

            # for each key
            #   sort subkeys by score
            #   return subkey with the highest score

            best = {}
            best_scores = 0

            for key in mappings:
                val = sorted(mappings[key].keys(), key=lambda x: mappings[key][x], reverse=True)[0]
                if val not in best: # remove clashes
                    best[key] = val
                    best_scores += mappings[key][best[key]]

            return best, best_scores

        min_dist = 5  # minimum distance between the triangle vertices

        svd = SVDSuperimposer()

        total_votes = 0
        # mappings: list[list[int:'hit kmer', int:'query kmer', int:'score sum']] = []
        mappings: dict[int:'hit position', dict[int:'query position', int:'score sum']] = defaultdict(lambda: defaultdict(int))

        i = 0
        while total_votes < min_votes and i < rounds:  # for _ in range(rounds):
            i += 1
            # triangle from the hit
            tetrahedron = random.choices(list(possible_matches.keys()), k=4)

            mapped_tetrahedron = [random.choice(possible_matches[tetrahedron[p]]) for p in range(4)]

            deviations = []
            for a in range(len(tetrahedron)):
                for b in range(a, len(tetrahedron)):
                    if a == b: continue

                    s = np.linalg.norm(template_positions[mapped_tetrahedron[a]] - template_positions[mapped_tetrahedron[b]])
                    deviations.append((
                            np.linalg.norm(positions[tetrahedron[a]] - positions[tetrahedron[b]]) - s
                    ) / (s + 0.0001))   # 'pseudo-count' zero avoidance

            deviations = np.array(deviations)
            if deviations.max() > max_deviation or deviations.min() < -max_deviation: continue
            if np.NaN in deviations: continue

            score = int((max_deviation - np.sum(np.abs(deviations)) / 6) * 10 * (1/max_deviation))

            # the scores are summed together
            for point in range(len(tetrahedron)):
                mappings[tetrahedron[point]][mapped_tetrahedron[point]] += score

            total_votes += 1

        mapping, score = best_mappings(mappings)
        if len(mapping) < 3:
            return None, None, None, None, None, None, None, None, None, None

        x = [template_positions[one] for one in mapping.values()]
        y = [positions[one] for one in mapping.keys()]

        svd.set(np.array(x), np.array(y))
        svd.run()

        avg_dev, rms_dev, median_dev = TetraHedronAligner.get_complete_mapping_avg_dev(mapping, positions, template_positions)

        if avg_dev is None:
            return None, None, None, None, None, None, None, None, None, None

        # print(mapping)
        # print(svd.get_rms())

        return svd.get_rms(), *svd.get_rotran(), mapping, i, total_votes, score / len(mapping), avg_dev, rms_dev, median_dev

    @staticmethod
    def iterative_get_rottrans_and_mapping(possible_matches, positions, template_positions, rmsd_cutoff: float = 4,
                                 rounds: int = 10, min_votes: int = 10, min_confirmations: int = 1, max_deviation=.3, mega_rounds = 5):

        def best_mappings(mappings: dict[int:'hit position', dict[int:'query position', int:'score sum']]):

            # for each key
            #   sort subkeys by score
            #   return subkey with the highest score

            best = {}
            best_scores = 0

            for key in mappings:
                val = sorted(mappings[key].keys(), key=lambda x: mappings[key][x], reverse=True)[0]
                if val not in best:  # remove clashes
                    best[key] = val
                    best_scores += mappings[key][best[key]]

            return best, best_scores

        min_dist = 5  # minimum distance between the triangle vertices

        svd = SVDSuperimposer()

        total_votes = 0
        # mappings: list[list[int:'hit kmer', int:'query kmer', int:'score sum']] = []
        mappings: dict[int:'hit position', dict[int:'query position', int:'score sum']] = defaultdict(
            lambda: defaultdict(int))


        i = 0
        def collect_mappings(rounds = 1000):
            nonlocal total_votes, mappings, i
            i = 0
            while total_votes < min_votes and i < rounds:  # for _ in range(rounds):
                i += 1
                # triangle from the hit
                tetrahedron = random.choices(list(possible_matches.keys()), k=4)

                mapped_tetrahedron = [random.choice(possible_matches[tetrahedron[p]]) for p in range(4)]

                deviations = []
                for a in range(len(tetrahedron)):
                    for b in range(a, len(tetrahedron)):
                        if a == b: continue

                        s = np.linalg.norm(
                            template_positions[mapped_tetrahedron[a]] - template_positions[mapped_tetrahedron[b]])
                        deviations.append((
                                                  np.linalg.norm(positions[tetrahedron[a]] - positions[tetrahedron[b]]) - s
                                          ) / (s + 0.0001))  # 'pseudo-count' zero avoidance

                deviations = np.array(deviations)
                if deviations.max() > max_deviation or deviations.min() < -max_deviation: continue
                if np.NaN in deviations: continue

                # print(deviations)
                score = int((max_deviation - np.sum(np.abs(deviations)) / 6) * 10 * (1 / max_deviation))
                # dprint(f"Score: {score}")

                # the scores are summed together
                for point in range(len(tetrahedron)):
                    mappings[tetrahedron[point]][mapped_tetrahedron[point]] += score

                total_votes += 1

        def score_mapping_so_far():
            nonlocal mappings

            median_dev_map = defaultdict(dict)
            best_map, _ = best_mappings(mappings)

            for one in mappings:
                for img in mappings[one]:
                    deviations = []
                    for second in best_map:
                        if one == second:
                            continue
                        if img == best_map[second]:
                            continue  # try to bypass collisions, should be reflected in the other deviations?

                        s = np.linalg.norm(template_positions[img] - template_positions[best_map[second]])
                        deviations.append(abs(np.linalg.norm(positions[one] - positions[second]) - s) / s)

                    median_dev_map[one][img] = np.median(deviations)

            return median_dev_map

        def filter_suboptimal_mappings(median_dev_map, q = .25):
            nonlocal mappings

            if len(mappings) == 0: return mappings

            values = [median_dev_map[a][b] for a in mappings for b in mappings[a]]
            qntl = np.quantile(values, 1 - q)
            print("Quantile:", qntl)
            new_mappings = defaultdict(lambda: defaultdict(int))
            for a in mappings:
                for b in mappings[a]:
                    if median_dev_map[a][b] <= qntl:
                        new_mappings[a][b] = mappings[a][b]

            return new_mappings

        for MR in range(mega_rounds):
            collect_mappings(1000)
            median_dev_map = score_mapping_so_far()
            mappings = filter_suboptimal_mappings(median_dev_map, q=.25)

        mapping, score = best_mappings(mappings)
        if len(mapping) < 3:
            return None, None, None, None, None, None, None, None, None, None

        x = [template_positions[one] for one in mapping.values()]
        y = [positions[one] for one in mapping.keys()]

        svd.set(np.array(x), np.array(y))
        svd.run()

        avg_dev, rms_dev, median_dev = TetraHedronAligner.get_complete_mapping_avg_dev(mapping, positions,
                                                                                       template_positions)

        if avg_dev is None:
            return None, None, None, None, None, None, None, None, None, None

        # print(mapping)
        # print(svd.get_rms())

        return svd.get_rms(), *svd.get_rotran(), mapping, i, total_votes, score / len(
            mapping), avg_dev, rms_dev, median_dev