import numpy as np
from KmerGen import generate_kmers
from sklearn.cluster import OPTICS
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def load_protein_ligand_contacts(pdb_code):
    DATASET_PATH = '../../TOUGH M1/TOUGH-M1_dataset/'
    begin = """Legend:
N     - ligand atom number in PDB entry
Dist  - distance (A) between the ligand and protein atoms
Surf  - contact surface area (A**2) between the ligand and protein atoms
*     - indicates destabilizing contacts
------------------------------------------------------------------------
    Ligand atom            Protein atom
-----------------   ----------------------------    Dist     Surf
  N   Name   Class    Residue       Name   Class
------------------------------------------------------------------------"""
    end = """------------------------------------------------------------------------"""

    contacts = []
    with open(DATASET_PATH + f'{pdb_code}/{pdb_code}00.lpc', 'r') as file:
        cut = file.read().split(begin)[1]
        cut = cut.split(end)[0]

        for line in cut.split('\n'):
            if len(line):

                res_chain = line.split()[4]
                res, chain = int(res_chain[:-1]), res_chain[-1]

                contacts.append([res, chain])

    return contacts


class Clustering:

    @staticmethod
    def pair_mean_clustering(candidate_ids, positions_3d, sequences, template_positions_3d, site_indices, all_possible_kmers,
                           cavity_scale_error: float, min_kmer_in_cavity_fraction: float) -> dict[int: list[int]]:

        """
        Clusters the site using pair mean cavity center approximation - this function supposes the cavity mean is a
        center of a line anchored by two residues which are part of the cavity.
        It exhaustively searches all possible lines (pairs of points), where the points are close enough (not farther
        than in the original cavity * cavity_scale_error).
        :param candidate_ids:
        :param positions_3d:
        :param sequences:
        :param template_positions_3d:
        :param site_indices:
        :param all_possible_kmers:
        :param cavity_scale_error:
        :param min_kmer_in_cavity_fraction:
        :return:
        """

        original_cavity_mean = np.mean(template_positions_3d[site_indices])
        max_distance: float = 0
        distances = []
        for res in site_indices:
            dist = np.linalg.norm(template_positions_3d[res] - original_cavity_mean)
            distances.append(dist)
            max_distance = max(max_distance, dist)
        max_distance *= cavity_scale_error
        q_distance = np.quantile(distances, 1)
        print(f"Max distance: {max_distance}, q distance: {q_distance}, mean distance: {np.mean(distances)}")

        accepted_clusters: list[dict[int, list[int]]] = []

        for candidate in candidate_ids:

            # find valid positions
            possible_positions = []
            for pos, kmer in enumerate(generate_kmers(sequences[candidate])):
                if kmer in all_possible_kmers:
                    possible_positions.append(pos)

            for A in range(len(possible_positions)):
                for B in range(A+1, len(possible_positions)):

                    # proceed if A, B are close enough to be in the same pocket
                    if np.linalg.norm(positions_3d[candidate][possible_positions[A]] - positions_3d[candidate][possible_positions[B]]) > q_distance*2:
                        continue

                    mean = np.mean(positions_3d[candidate][[possible_positions[A], possible_positions[B]]])
                    cluster = []
                    for C in possible_positions:
                        if np.linalg.norm(positions_3d[candidate][C] - mean) <= q_distance:
                            cluster.append(C)

                    if len(cluster) >= min_kmer_in_cavity_fraction * len(site_indices):
                        accepted_clusters.append({candidate: cluster})

        # print(accepted_clusters)
        return accepted_clusters


    @staticmethod
    def optics(candidate_ids, positions_3d, sequences, id_translations, template_positions_3d, site_indices, all_possible_kmers,
                           cavity_scale_error: float, min_kmer_in_cavity_fraction: float, graphics = False) -> dict[int: list[int]]:

        # original_cavity_mean = np.mean(template_positions_3d[site_indices])
        # max_distance: float = 0
        # distances = []
        # for res in site_indices:
        #     dist = np.linalg.norm(template_positions_3d[res] - original_cavity_mean)
        #     distances.append(dist)
        #     max_distance = max(max_distance, dist)
        #
        #
        # center_point = None
        # dist_to_mean = -1
        # for res in site_indices:
        #     d = np.linalg.norm(template_positions_3d[res] - original_cavity_mean)
        #     if d < dist_to_mean or center_point is None:
        #         center_point = res
        #         dist_to_mean = d
        #
        # max_dist_to_center_point = 0
        # for res in site_indices:
        #     max_dist_to_center_point = max(max_dist_to_center_point, np.linalg.norm(template_positions_3d[res] -
        #                                                                             template_positions_3d[center_point]))
        #
        # q_distance = np.quantile(distances, 1)
        # print(f"Max distance: {max_distance}, q distance: {q_distance}, mean distance: {np.mean(distances)}, max dist to center point: {max_dist_to_center_point}")

        max_distance = 0
        for a in site_indices:
            for b in site_indices:
                max_distance = max(max_distance, np.linalg.norm(template_positions_3d[a] - template_positions_3d[b]))

        max_distance *= cavity_scale_error


        accepted_clusters: list[dict[int, list[int]]] = []

        optics = OPTICS(min_samples=3, max_eps=max_distance)
                            # min_cluster_size=int(len(site_indices) * min_kmer_in_cavity_fraction),
                            # n_jobs=1)

        for candidate in tqdm(candidate_ids, desc='Clustering'):

            # find valid positions
            possible_positions = []
            for pos, kmer in enumerate(generate_kmers(sequences[candidate])):
                if kmer in all_possible_kmers:
                    possible_positions.append(pos)

            if len(possible_positions) <= max(3, int(len(site_indices) * min_kmer_in_cavity_fraction)):
                continue

            labels = optics.fit_predict(positions_3d[candidate][possible_positions])

            if graphics:

                print(f"\nClusters for {id_translations[candidate]}")

                correct = list(set([x[0] for x in load_protein_ligand_contacts(id_translations[candidate])]))
                #print(load_protein_ligand_contacts(id_translations[candidate]))
                #print(len(sequences[candidate]))

                overlaps = {}
                for label in np.unique(labels):
                    overlap = 0
                    for i, one in enumerate(possible_positions):
                        if labels[i] == label and one in correct:
                           overlap += 1
                    print(f"Cluster {label} overlap: {overlap} ~ {round(100 * overlap / len(correct))} % ({round(100 * overlap / len(site_indices))} % from query.)")
                    overlaps[label] = round(100 * overlap / len(correct))

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Define colors for each label
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'brown']

                # Plot each point with its corresponding label color
                for i, label in enumerate(np.unique(labels)):
                    if label < 0:
                        continue

                    ax.scatter(positions_3d[candidate][possible_positions][labels == label, 0],
                               positions_3d[candidate][possible_positions][labels == label, 1],
                               positions_3d[candidate][possible_positions][labels == label, 2],
                               alpha=0.6, label=f'Cluster {label} ({overlaps[label]}%)')
                ax.scatter(positions_3d[candidate][correct][:, 0],
                        positions_3d[candidate][correct][:, 1], positions_3d[candidate][correct][:, 2], "red", alpha=1, s=100, linewidths=.6, marker='+')
                ax.plot(positions_3d[candidate][possible_positions][labels == -1, 0],
                        positions_3d[candidate][possible_positions][labels == -1, 1], positions_3d[candidate][possible_positions][labels == -1, 2], "k+", alpha=0.1)

                # Set labels and legend
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()

                plt.title(id_translations[candidate])
                plt.show()

            clusters = {}
            for i in range(len(possible_positions)):
                # skip unclustered
                if labels[i] == -1:
                    continue

                if labels[i] in clusters:
                    clusters[labels[i]].append(possible_positions[i])
                else:
                    clusters[labels[i]] = [possible_positions[i]]

            for cluster_i in clusters:

                if len(clusters[cluster_i]) >= len(site_indices) * min_kmer_in_cavity_fraction:

                    accepted_clusters.append({candidate: clusters[cluster_i]})

        return accepted_clusters