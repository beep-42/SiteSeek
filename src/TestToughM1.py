import random
from multiprocessing import Pool

from ClusterVectorizer import Vectorizer
from DB import DB, SuitableTriangleExpansion
import pickle
import csv
import re
from tqdm import tqdm
from Bio import SeqIO, PDB
import os
from Demo import struct_2_seq
import numpy as np

from FindSurfaceResidues import SurfaceFinder
from FullSiteMapping import FullSiteMapper
from Helpers import CavitySelect
from MultithreadedSearch import search_multithread
from ToughM1Helpers import load_protein_ligand_contacts, load_list_to_dict


PICKLE_NAME = 'database_tough_m1.picle'
SUPERIMPOSED_FOLDER = 'TOUGH-M1-Results/Superimposed/'
DATASET_PATH = '../../TOUGH M1/TOUGH-M1_dataset/'
POSITIVE_LIST = '../../TOUGH M1/TOUGH-M1_positive.list'
NEGATIVE_LIST = '../../TOUGH M1/TOUGH-M1_negative.list'


def index_tough_folder_pdb(path, db):
    parser = PDB.PDBParser(QUIET=True)
    for folder in tqdm(list(os.walk(path)), desc='Indexing db'):

        direc = folder[0]
        files = folder[2]

        for file in files:
            if file.split('.')[-1] != 'pdb' or '00' in file or 'cavity' in file:
                continue

            struct = parser.get_structure('test', direc + '/' + file)
            seq = struct_2_seq(struct)
            name = file.split('.')[0]
            db.add(name, seq, struct)


def load_tough_m1():
    global PICKLE_NAME, DATASET_PATH
    try:
        with open(PICKLE_NAME, 'rb') as file:
            db = pickle.load(file)

    except:
        db = DB()
        index_tough_folder_pdb(DATASET_PATH, db)
        with open(PICKLE_NAME, 'wb') as file:
            pickle.dump(db, file)

    return db


def write_superimposed(id1, id2, mapping, cluster, rot, trans, prefix=''):
    """Superimposes the first one on the second one and saves them to the
    Superimposed folder."""
    parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    io = PDB.PDBIO()
    struct1 = parser.get_structure(id1, DATASET_PATH + f'{id1}/{id1}.pdb')
    struct1cav = parser.get_structure(id1, DATASET_PATH + f'{id1}/{id1}-cavity.pdb')
    struct1lig = parser.get_structure(id1, DATASET_PATH + f'{id1}/{id1}00.pdb')
    # struct2 = parser.get_structure(id2, DATASET_PATH + f'{id2}/{id2}.pdb')

    for atom in struct1.get_atoms():
        atom.transform(rot, trans)
    for atom in struct1cav.get_atoms():
        atom.transform(rot, trans)
    for atom in struct1lig.get_atoms():
        atom.transform(rot, trans)

    # select mapped residues
    mapped = []
    cluster_res = []
    i = 0
    for residue in struct1.get_residues():
        if i in mapping:
            mapped.append(residue)
        if i in cluster:
            cluster_res.append(residue)
        i += 1

    # # Join the structs to one file
    # chains = list(struct2.get_chains())
    # # Rename the chain
    # chains[0].id = 'Z'
    # # Detach this chain from structure2
    # chains[0].detach_parent()
    # # Add it onto structure1
    # struct1[0].add(chains[0])

    target_file = SUPERIMPOSED_FOLDER + f'{prefix}{id1} superimposed to {id2}.pdb'
    io.set_structure(struct1)
    io.save(target_file)
    target_file = SUPERIMPOSED_FOLDER + f'{prefix}{id1}-cavity superimposed to {id2}.pdb'
    io.set_structure(struct1cav)
    io.save(target_file)
    target_file = SUPERIMPOSED_FOLDER + f'{prefix}{id1} ligand superimposed to {id2}.pdb'
    io.set_structure(struct1lig)
    io.save(target_file)
    target_file = SUPERIMPOSED_FOLDER + f'{prefix}{id1} mappings superimposed to {id2}.pdb'
    io.set_structure(struct1)
    io.save(target_file, select=CavitySelect(mapped))
    target_file = SUPERIMPOSED_FOLDER + f'{prefix}{id1} cluster superimposed to {id2}.pdb'
    io.set_structure(struct1)
    io.save(target_file, select=CavitySelect(cluster_res))


def save_clusters():
    global POSITIVE_LIST, NEGATIVE_LIST
    db = load_tough_m1()
    positive = load_list_to_dict(POSITIVE_LIST)
    negative = load_list_to_dict(NEGATIVE_LIST)

    parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    io = PDB.PDBIO()
    for case in positive:
        if case not in db.ids:
            print(f"{case} not found. Skipping.")
            continue

        print('Getting:', case)
        db_id = db.ids.index(case)
        seq = db.seqs[db_id]
        positions = db.pos[db_id]
        contacts = load_protein_ligand_contacts(case)
        site = list(set([x[0] for x in contacts]))

        results = db.get_clusters(positions, seq, site, k_mer_min_found_fraction=.1, k_mer_similarity_threshold=7,
                                   allowed_dist_error=2, cavity_scale_error=1, min_kmer_in_cavity_fraction=.1,
                                   max_rmsd=30, check_only=positive[case], dist_to_surface_cutoff=4) #positive[case])

        seen = {}
        for result in results:
            if result['hit_id'] not in seen:
                seen[result['hit_id']] = 0
            else:
                seen[result['hit_id']] += 1
            target_file = SUPERIMPOSED_FOLDER + f'{result["hit_id"]} cluster{seen[result["hit_id"]]} with respect to {case}.pdb'
            struct = parser.get_structure(result['hit_id'], DATASET_PATH + f"{result['hit_id']}/{result['hit_id']}.pdb")
            io.set_structure(struct)
            res = list(struct.get_residues())
            cav = [res[x] for x in result['positions']]
            io.save(target_file, select=CavitySelect(cav))

        return


def test_kmer_search_on_tough_m1():
    global POSITIVE_LIST, NEGATIVE_LIST
    db = load_tough_m1()
    positive = load_list_to_dict(POSITIVE_LIST)
    negative = load_list_to_dict(NEGATIVE_LIST)

    correct = 0
    incorrect = 0
    ambiguous = 0

    for case in positive:
        correct = 0
        incorrect = 0
        ambiguous = 0

        if case not in db.ids:
            print(f"{case} not found. Skipping.")
            continue

        print('Getting:', case)
        db_id = db.ids.index(case)
        seq = db.seqs[db_id]
        positions = db.pos[db_id]
        contacts = load_protein_ligand_contacts(case)
        site = [x[0] for x in contacts]

        results = db.kmer_search(seq, site, .7, 10)

        for result in results:

            if result == case:
                correct += 1
            elif result in positive[case]:
                correct += 1
                # print(f"{case} reported VALID {result['hit_id']} with rmsd {result['rmsd']} and similarity score of {result['similarity score']}")

            elif result in negative[case] if case in negative else []:
                incorrect += 1
                # print(f"{case} reported invalid {result['hit_id']} with rmsd {result['rmsd']} and similarity score of {result['similarity score']}")
            else:
                ambiguous += 1

        print(f'Accuracy: \t{correct / (correct + incorrect + 1)}\t({correct}/{len(positive[case])})')
        print(f'Inaccuracy: \t{incorrect / (correct + incorrect + 1)}\t({incorrect}/{len(negative[case]) if case in negative else 0})')
        print(f'Ambiguity: \t{ambiguous / (correct + incorrect + ambiguous)}\t({ambiguous})')
        print(f'Left out: \t{(len(positive[case]) - correct) / len(positive[case])}\t({len(positive[case]) - correct})')


def calc_cavity_mean_and_coverage():
    db = load_tough_m1()

    with open('TOUGH-M1-Results/1lkxD.pickle', 'rb') as handle:
        results = pickle.load(handle)

    with open('TOUGH-M1-Results/Results-1lkxD.pickle', 'rb') as file:
        results_log = pickle.load(file)

    for i in range(len(results_log)):

        for result in results:
            if result['hit_id'] == results_log[i]['id']:
                break

        if result['hit_id'] != results_log[i]['id']:
            raise ValueError

        cavity = [x[0] for x in load_protein_ligand_contacts(result['hit_id'])]
        cav = []
        for one in cavity:
            if one < len(db.seqs[db.ids.index(results_log[i]['id'])]):
                cav.append(one)
        cavity = cav

        union_size = 0
        extra_size = 0
        for one in result['cluster']:
            if one in cavity:
                union_size += 1
            else:
                extra_size += 1
        results_log[i]['cluster_cavity_coverage'] = union_size / len(cavity)
        results_log[i]['cluster_extra_fraction'] = extra_size/len(result['cluster'])

        union_size = 0
        extra_size = 0
        for one in result['mapping']:
            if one in cavity:
                union_size += 1
            else:
                extra_size += 1
        results_log[i]['mapping_cavity_coverage'] = union_size / len(cavity)
        results_log[i]['mapping_extra_fraction'] = extra_size/len(result['cluster'])

        cavity_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][cavity])
        cluster_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][result['cluster']])
        mapping_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][list(result['mapping'].keys())])

        cluster_dist = np.linalg.norm(cavity_mean - cluster_mean)
        mapping_dist = np.linalg.norm(cavity_mean - mapping_mean)

        results_log[i]['cluster_dist'] = cluster_dist
        results_log[i]['mapping_dist'] = mapping_dist

        i += 1

    print("I: ", i)
    with open('TOUGH-M1-Results/Results-1lkxD.pickle', 'wb') as file:
        pickle.dump(results_log, file)


def get_transposed_and_query_ligand_distance(pdb_code_hit, rot, trans, pdb_code_query):

    parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    hit_ligand = parser.get_structure(pdb_code_hit, DATASET_PATH + f'{pdb_code_hit}/{pdb_code_hit}00.pdb')
    query_ligand = parser.get_structure(pdb_code_query, DATASET_PATH + f'{pdb_code_query}/{pdb_code_query}00.pdb')

    for atom in hit_ligand.get_atoms():
        atom.transform(rot, trans)

    distance = np.linalg.norm(hit_ligand.center_of_mass() - query_ligand.center_of_mass())

    return distance


def expand_site_with_neighbors(positions, site, distance_cutoof):

    expanded = []

    for origin in site:
        i = 0
        for neighbor in positions:
            if np.linalg.norm(neighbor - positions[origin]) <= distance_cutoof:
                expanded.append(i)
            i += 1

    return list(set(expanded))


def run(db, positions, seq, site, check_only):
    return db.kmer_cluster_svd_search(template_positions=positions,
                                     seq=seq, site=site, k_mer_min_found_fraction=.05, k_mer_similarity_threshold=11,
                               allowed_dist_error=2, cavity_scale_error=1.3, min_kmer_in_cavity_fraction=.05,
                               max_rmsd=300, min_coverage = 0.0, min_score = -44440, check_only = check_only) #['1dj3B'])



def test_tough_m1():
    global POSITIVE_LIST, NEGATIVE_LIST

    def run_parallel(db, positions, seq, site, check_only, n=6):
        data_split = [list(x) for x in np.array_split(check_only, n)]
        pool = Pool(processes=n)
        args = zip([db]*n, [positions]*n, [seq]*n, [site]*n, data_split)
        data = pool.starmap(run, args)
        pool.close()
        pool.join()

        ret = []
        for part in data:
            ret += part

        return ret

    db = load_tough_m1()
    positive = load_list_to_dict(POSITIVE_LIST)
    negative = load_list_to_dict(NEGATIVE_LIST)

    correct = 0
    incorrect = 0
    ambiguous = 0

    all_results = []

    r = 0
    total_positive = []

    for case in tqdm(list(positive.keys())[:10]):
    # searched_positive = positive[list(positive.keys())[0]]
    # for case in tqdm(list(random.choices(positive[list(positive.keys())[0]], k=4)) + [list(positive.keys())[0]]):

        total_positive += positive[case]
        r += 1
        # if r < 10:
        #     r += 1
        #     continuefound_kmers

        correct = 0
        incorrect = 0
        ambiguous = 0

        if case not in db.ids:
            print(f"{case} not found. Skipping.")
            continue

        print('Getting:', case)
        db_id = db.ids.index(case)
        seq = db.seqs[db_id]
        positions = db.pos[db_id]
        contacts = load_protein_ligand_contacts(case)
        site = list(set([x[0] for x in contacts]))
        if len(seq) in site: site.remove(len(seq))   # remove weird residue index sometimes appearing at the end
        expanded_site = expand_site_with_neighbors(positions, site, 4)
        # print(f"sele site, {' '.join([f'resi {x}' for x in site])}")

        try:
            raise ValueError
            with open('TOUGH-M1-Results/triangular-experimental-' + case + '.pickle', 'rb') as handle:
                results = pickle.load(handle)
        except:
            check_only = db.ids
            # check_only = positive[case]
            # if case in negative:
            #    check_only += negative[case]

            results = run_parallel(db, positions, seq, site, check_only)

            with open('TOUGH-M1-Results/RC-experimental-' + case + '.pickle', 'wb') as handle:
                pickle.dump(results, handle)

        results_log = []
        seen = []
        for result in tqdm(results, desc='Processing results'):
            # if result['hit_id'] not in positive[case] and result['hit_id'] not in negative[case]:
            #    continue

            #if result['coverage'] < .05 or result['similarity score'] < 50:
            #    continue


            if False and len(results_log) < 10:
                write_superimposed(
                    result['hit_id'],
                    case,
                    result['mapping'],
                    result['cluster'],
                    result['rotation'],
                    result['translation']
                )

            if result['hit_id'] not in seen:
                if result['hit_id'] == case:
                    correct += 1
                    # print('Identity:', result)
                elif result['hit_id'] in positive[case]:
                    correct += 1
                    # print(f"{case} reported VALID {result['hit_id']} with rmsd {result['rmsd']} and similarity score of {result['similarity score']}")
                    # print('Positive:', result)

                elif result['hit_id'] in negative[case] if case in negative else []:
                    incorrect += 1
                    # print('Negative:', result)
                    # print(f"{case} reported invalid {result['hit_id']} with rmsd {result['rmsd']} and similarity score of {result['similarity score']}")
                else:
                    # print('Ambiguous:', result)
                    ambiguous += 1
                seen.append(result['hit_id'])

            # surface = SurfaceFinder(result['hit_id'], save=True)
            # total_exposed = 0
            # for res_i in result['mapping']:
            #     total_exposed += 1 if surface.get_distance_from_res(res_i) <= 2 else 0
            #
            # print("Total exposed area:", total_exposed, 'about', total_exposed / len(result['mapping']))

            full_site_rmsds = []
            full_site_scores = []
            nearby_fractions = []
            MIN_SCORES = [None]
            for i, min_score in enumerate(MIN_SCORES):

                full_site_rot, full_site_trans = result['rotation'], result['translation']

                for _ in range(1):
                    full_site_mapping, full_site_rmsd, full_site_rot, full_site_trans, nearby_fraction, mapping_loss, mapping_persistence = FullSiteMapper.get_full_closest_mappings(
                        db.kdtrees[db.ids.index(result['hit_id'])],
                        db.pos[db.ids.index(result['hit_id'])], db.pos[db.ids.index(case)],
                        db.seqs[
                            db.ids.index(
                                result[
                                    'hit_id'])],
                        db.seqs[
                            db.ids.index(
                                case)],
                        expanded_site,
                        result['mapping'],
                        full_site_rot, full_site_trans, min_score=min_score, nearby_distance=1.0)

                full_site_score = db.compute_mapping_similarity_score(full_site_mapping, db.seqs[db.ids.index(result['hit_id'])], db.seqs[db.ids.index(case)])

                full_site_rmsds.append(full_site_rmsd)
                full_site_scores.append(full_site_score)
                nearby_fractions.append(nearby_fraction)

            _, full_site_rmsd_noext, _, _, _, _, _ = FullSiteMapper.get_full_closest_mappings(
                db.kdtrees[db.ids.index(result['hit_id'])],
                db.pos[db.ids.index(result['hit_id'])], db.pos[db.ids.index(case)],
                db.seqs[
                    db.ids.index(
                        result[
                            'hit_id'])],
                db.seqs[
                    db.ids.index(
                        case)],
                site,
                result['mapping'],
                full_site_rot, full_site_trans, min_score=min_score)

            cavity = list(set([x[0] for x in load_protein_ligand_contacts(result['hit_id'])]))
            cav = []
            for one in cavity:
                if one < len(db.seqs[db.ids.index(result['hit_id'])]):
                    cav.append(one)
            cavity = cav

            inter_size = 0
            extra_size = 0
            for one in result['cluster']:
                if one in cavity:
                    inter_size += 1
                else:
                    extra_size += 1
            cluster_cavity_coverage = inter_size / len(cavity)
            cluster_accuracy = inter_size / len(result['cluster'])
            cluster_extra_fraction = extra_size / len(result['cluster'])
            cluster_n_hits = inter_size
            cluster_size = len(result['cluster'])

            inter_size = 0
            extra_size = 0
            for one in result['mapping']:
                if one in cavity:
                    inter_size += 1
                else:
                    extra_size += 1
            mapping_cavity_coverage = inter_size / len(cavity)
            mapping_accuracy = inter_size / len(result['cluster'])
            mapping_extra_fraction = extra_size / len(result['cluster'])

            cavity_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][cavity])
            """Traceback (most recent call last):
  File "/home/jakubt/Documents/SiteSeek/v0.2/SiteSeek/./TestToughM1.py", line 457, in <module>
    test_tough_m1()
  File "/home/jakubt/Documents/SiteSeek/v0.2/SiteSeek/./TestToughM1.py", line 419, in test_tough_m1
    cavity_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][cavity])
IndexError: index 676 is out of bounds for axis 0 with size 676
"""
            cluster_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][result['cluster']], axis=0)
            mapping_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][list(result['mapping'].keys())], axis=0)

            cluster_dist = np.linalg.norm(cavity_mean - cluster_mean)
            mapping_dist = np.linalg.norm(cavity_mean - mapping_mean)

            if result['hit_id'] in positive[case]:
                label = 'positive'
            elif case in negative and result['hit_id'] in negative[case]:
                label = 'negative'
            else:
                label = 'ambiguous'


            this = {
                'id': result['hit_id'],
                'score': result['similarity score'],
                'found kmers': result['found kmers'],
                'found kmer frac': result['found kmer frac'],
                'rmsd': result['rmsd'],
                # 'full site rmsd': full_site_rmsd,
                # 'full site score': full_site_score,
                'coverage': result['coverage'],
                'cluster size fraction': len(result['cluster']) / len(site),
                # 'total exposed': total_exposed,
                # 'fraction exposed': total_exposed / len(result['mapping']),
                'refined rmsd': result['refined rmsd'],
                'triangle votes': result['triangle votes'],
                'triangle rounds': result['triangle rounds'],
                'hedron score': result['hedron score'],
                'avg dev': result['avg dev'],
                'rms dev': result['rms dev'],
                'med dev': result['med dev'],
                'mapping loss': mapping_loss,
                'mapping persistence': mapping_persistence,
                'cluster cavity coverage': cluster_cavity_coverage,
                'cluster extra fraction': cluster_extra_fraction,
                'cluster accuracy': cluster_accuracy,
                'cluster size': cluster_size,
                'cluster n hits': cluster_n_hits,
                'mapping cavity coverage': mapping_cavity_coverage,
                'mapping extra fraction': mapping_extra_fraction,
                'mapping accuracy': mapping_accuracy,
                'cluster dist': cluster_dist,
                'mapping dist': mapping_dist,
                'ligand dist': get_transposed_and_query_ligand_distance(result['hit_id'], result['rotation'], result['translation'], case),
                'label': label
            }
            for i, min_score in enumerate(MIN_SCORES):
                this[f'full site rmsd at {min_score}'] = full_site_rmsds[i]
                this[f'full site score at {min_score}'] = full_site_scores[i]
                this[f'nearby fraction at min score {min_score}'] = nearby_fractions[i]
            this['full site rmsd no ext'] = full_site_rmsd_noext

            if result['refined rotation'] is not None:
                this['refined ligand dist'] = get_transposed_and_query_ligand_distance(result['hit_id'], result['refined rotation'],
                                                                        result['refined translation'], case)

#             ligand_dist = get_transposed_and_query_ligand_distance(result['hit_id'], result['refined rotation'], result['refined translation'], case)
#             # if ligand_dist > 3 and cluster_dist < 3: # full_site_rmsd_noext <= 4 and ligand_dist < 3:
# # nearby_fractions[0] >= .9 and result['hit_id'] in negative[case]:
#             if (random.random() >= .5 and ligand_dist <= 5) or (random.random() >= .99 and ligand_dist > 5) or ligand_dist < 1:
#                 write_superimposed(
#                     result['hit_id'],
#                     case,
#                     result['mapping'],
#                     result['cluster'],
#                     result['refined rotation'],
#                     result['refined translation'],
#                     prefix=f'close clust dist lig/{label} - cluster dist {cluster_dist} refined ligand dist {ligand_dist} '
#                 )

            results_log.append(this)

            another = this.copy()
            # another['query'] = case
            all_results.append(another)

        # with open(f'TOUGH-M1-Results/Triangular-Experimental-Results-{case}.pickle', 'wb') as file:
        #     pickle.dump(results_log, file)

        # with open(f'TOUGH-M1-Results/RC-experimental-{case}-results.pickle', 'wb') as file:
        #    pickle.dump(results_log, file)

        print(f'Accuracy: \t{correct / (correct + incorrect + 1)}\t({correct})')
        print(f'Inaccuracy: \t{incorrect / (correct + incorrect + 1)}\t({incorrect})')
        print(f'Ambiguity: \t{ambiguous / (correct + incorrect + ambiguous + 1)}\t({ambiguous})')
        print(f'Left out: \t{(len(positive[case]) - correct) / len(positive[case])}\t({len(positive[case]) - correct})')

        # return

    with open(f'TOUGH-M1-Results/RC-10-results.pickle', 'wb') as file:
        pickle.dump(all_results, file)

    print(f"Expected positive hits: {len(total_positive)}")
    print(f"Expected unique positive hits: {len(set(total_positive))}")

def test_clusters_on_M1():
    global POSITIVE_LIST, NEGATIVE_LIST
    db = load_tough_m1()
    positive = load_list_to_dict(POSITIVE_LIST)
    negative = load_list_to_dict(NEGATIVE_LIST)

    kmer_sim = 10

    correct = 0
    incorrect = 0
    ambiguous = 0

    r = 0
    for case in positive:
        # if r < 10:
        #     r += 1
        #     continue
        # if case not in negative:
        #     continue

        correct = 0
        incorrect = 0
        ambiguous = 0

        if case not in db.ids:
            print(f"{case} not found. Skipping.")
            continue

        print('Getting clusters for:', case)
        db_id = db.ids.index(case)
        seq = db.seqs[db_id]
        positions = db.pos[db_id]
        contacts = load_protein_ligand_contacts(case)
        site = [x[0] for x in contacts]
        site_f = []
        for one in site:
            if one < len(seq):
                site_f.append(one)
        site = list(tuple(site_f))

        vectorizer = Vectorizer(site, seq, kmer_sim)
        site_vector = vectorizer.vectorize(site, seq)

        print("Site vector:", site_vector)
        print(f"sele site, {' '.join([f'resi {x}' for x in site])}")

        results = db.get_clusters(template_positions=positions,
                                             seq=seq, site=site, k_mer_min_found_fraction=.1,
                                             k_mer_similarity_threshold=kmer_sim,
                                             allowed_dist_error=2, cavity_scale_error=1,
                                             min_kmer_in_cavity_fraction=.1, max_rmsd=999999)

        results_log = []
        seen = []
        for result in tqdm(results):

            if result['hit_id'] not in seen:
                if result['hit_id'] == case:
                    correct += 1
                    print('Identity:', result)
                elif result['hit_id'] in positive[case]:
                    correct += 1
                    # print(f"{case} reported VALID {result['hit_id']} with rmsd {result['rmsd']} and similarity score of {result['similarity score']}")
                    print('Positive:', result)

                elif result['hit_id'] in negative[case] if case in negative else []:
                    incorrect += 1
                    print('Negative:', result)
                    # print(f"{case} reported invalid {result['hit_id']} with rmsd {result['rmsd']} and similarity score of {result['similarity score']}")
                else:
                    print('Ambiguous:', result)
                    ambiguous += 1
                seen.append(result['hit_id'])

            cavity = list(set([x[0] for x in load_protein_ligand_contacts(result['hit_id'])]))
            cav = []
            for one in cavity:
                if one < len(db.seqs[db.ids.index(result['hit_id'])]):
                    cav.append(one)
            cavity = cav

            union_size = 0
            extra_size = 0
            for one in result['positions']:
                if one in cavity:
                    union_size += 1
                else:
                    extra_size += 1
            cluster_cavity_coverage = union_size / len(cavity)
            cluster_accuracy = union_size / len(result['positions'])
            cluster_extra_fraction = extra_size / len(result['positions'])

            simple_size_fraction = len(result['positions']) / len(site)

            cavity_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][cavity], axis=0)
            cluster_mean = np.mean(db.pos[db.ids.index(result['hit_id'])][result['positions']], axis=0)

            cluster_dist = np.linalg.norm(cavity_mean - cluster_mean)

            # calculate cluster cosine similarity
            cluster_vector = vectorizer.vectorize(result['positions'], db.seqs[db.ids.index(result['hit_id'])])
            cosine_similarity = np.dot(site_vector, cluster_vector) / (np.linalg.norm(site_vector) * np.linalg.norm(cluster_vector))

            label = 'ambiguous'
            if result['hit_id'] in positive[case]:
                label = 'positive'
            elif result['hit_id'] in negative[case]:
                label = 'negative'

            this = {
                'id': result['hit_id'],
                'cluster cavity coverage': cluster_cavity_coverage,
                'cluster extra fraction': cluster_extra_fraction,
                'cluster accuracy': cluster_accuracy,
                'cluster dist': cluster_dist,
                'simple size fraction': simple_size_fraction,
                'cosine similarity': cosine_similarity,
                'label': label
            }
            results_log.append(this)

        with open(f'TOUGH-M1-Results/Cluster-Results-{case}.pickle', 'wb') as file:
            pickle.dump(results_log, file)

        print(f'Accuracy: \t{correct / (correct + incorrect + 1)}\t({correct})')
        print(f'Inaccuracy: \t{incorrect / (correct + incorrect + 1)}\t({incorrect})')
        print(f'Ambiguity: \t{ambiguous / (correct + incorrect + ambiguous + 1)}\t({ambiguous})')
        print(f'Left out: \t{(len(positive[case]) - correct) / len(positive[case])}\t({len(positive[case]) - correct})')

        return


if __name__ == '__main__':
    #save_clusters()
    #test_kmer_search_on_tough_m1()
    test_tough_m1()
    #test_clusters_on_M1()
    #calc_cavity_mean_and_coverage()
