import numpy as np
from Bio.PDB.Superimposer import SVDSuperimposer


class ICP:

    @staticmethod
    def align_closest_mappings(possible_matches, positions, template_positions, rot, trans, cutoff: float = None):
        """
        Superimposes the structures using the provided rot & translation and returns the closest mapping from
        possible_matches.
        :param possible_matches:
        :param positions:
        :param template_positions:
        :param rot:
        :param trans:
        :param cutoff: distance cutoff, mapping more distant than cutoff won't be mapped.
        :return:
        """

        transformed = np.dot(positions, rot) + trans
        mapped = {}

        for kmer in possible_matches:

            # with repeating
            dists = [np.linalg.norm(transformed[kmer] - template_positions[x]) for x in possible_matches[kmer]]
            mapping = possible_matches[kmer][np.argmin(dists)]

            # # without repeating
            # dists = [np.linalg.norm(transformed[kmer] - template_positions[x]) if kmer not in mapped else 99999999 for x
            #          in possible_matches[kmer]]
            # mapping = possible_matches[kmer][np.argmin(dists)]

            if cutoff is None or min(dists) <= cutoff:
                mapped[kmer] = mapping

        x = []
        y = []

        for kmer in mapped:
            y.append(positions[kmer])
            x.append(template_positions[mapped[kmer]])

        if len(x) < 3:
            return None, None, None, None

        svd = SVDSuperimposer()
        svd.set(np.array(x), np.array(y))
        svd.run()

        return mapped, svd.get_rms(), *svd.get_rotran()