import numpy as np
from Bio.PDB.kdtrees import KDTree
from Bio.SVDSuperimposer import SVDSuperimposer
from KmerGen import BLOSUM62


class FullSiteMapper:

    @staticmethod
    def get_full_closest_mappings(tree: KDTree, positions: list[int], template_positions: list[int], seq: str,
                                  template_seq: str, site: list[int], mapping: dict[int, int], rot: np.ndarray,
                                  trans: np.ndarray, min_score: float = None,
                                  nearby_distance: float = 1.0, scoring_matrix: list[str] = BLOSUM62):
        """
        Superimposes the structures using the provided rot & translation and returns the closest mapping.
        """

        # transform the query so we can use the prepared KDtree
        transformed_query = np.dot(template_positions[site] - trans, rot.T)     # use A^-1 = A^T and - trans

        mapped = {}

        nearby_count = 0

        for site_i in range(len(site)):
            # for each transformed point search the kd tree until we find suitable hit and add it to the mapping
            k = 1
            [dist], [result] = tree.query(transformed_query[site_i], k=[k])

            if min_score is not None:
                while scoring_matrix[seq[result], template_seq[site[site_i]]] < min_score:
                    k += 1
                    [dist], [result] = tree.query(transformed_query[site_i], k=[k])
                    if result >= len(seq):
                        result = len(seq) - 1
                        break

            mapped[result] = site[site_i]
            if dist <= nearby_distance:
                nearby_count += 1

        x = []
        y = []
        for i in mapped:
            y.append(positions[i])
            x.append(template_positions[mapped[i]])

        svd = SVDSuperimposer()
        svd.set(np.array(x).astype(np.float32), np.array(y).astype(np.float32))
        svd.run()

        # calculate the mapping loss
        remain = 0
        persistent_mappings = 0
        for one in mapped:
            if one in mapping:
                remain += 1

                if mapped[one] == mapping[one]:
                    persistent_mappings += 1

        mapping_loss = (len(mapping) - remain) / len(mapping)
        mapping_persistence = persistent_mappings / len(mapping)

        return mapped, svd.get_rms(), *svd.get_rotran(), nearby_count / len(site), mapping_loss, mapping_persistence

    @staticmethod
    def get_full_closest_mappings_notree(positions: list[int], template_positions: list[int], seq: str,
                                  template_seq: str, site: list[int], mapping: dict[int, int], rot: np.ndarray,
                                  trans: np.ndarray, min_score: float = None,
                                  nearby_distance: float = 1.0, scoring_matrix: list[str] = BLOSUM62):
        """
        Superimposes the structures using the provided rot & translation and returns the closest mapping.
        """

        # transform the query so we can use the prepared KDtree
        transformed_query = np.dot(template_positions[site] - trans, rot.T)  # use A^-1 = A^T and - trans

        mapped = {}

        nearby_count = 0

        if min_score is not None: raise NotImplementedError("The minimum required score in full site mapping is not implemented.")

        for site_i in range(len(site)):

            # min_dist, result = np.inf, None
            # for point_i in range(len(positions)):
            #     d = np.linalg.norm(transformed_query[site_i] - positions[point_i])
            #     if d < min_dist:
            #         min_dist = d
            #         result = point_i
            dists = positions - transformed_query[site_i]
            norms = np.linalg.norm(dists, axis=1)
            result = np.argmin(norms)
            min_dist = np.min(norms)

            mapped[result] = site[site_i]
            if min_dist <= nearby_distance:
                nearby_count += 1

        x = []
        y = []
        for i in mapped:
            y.append(positions[i])
            x.append(template_positions[mapped[i]])

        svd = SVDSuperimposer()
        svd.set(np.array(x).astype(np.float32), np.array(y).astype(np.float32)) # float16 not supported
        svd.run()

        # calculate the mapping loss
        remain = 0
        persistent_mappings = 0
        for one in mapped:
            if one in mapping:
                remain += 1

                if mapped[one] == mapping[one]:
                    persistent_mappings += 1

        mapping_loss = (len(mapping) - remain) / len(mapping)
        mapping_persistence = persistent_mappings / len(mapping)

        return mapped, svd.get_rms(), *svd.get_rotran(), nearby_count / len(site), mapping_loss, mapping_persistence