#!/usr/bin/env python3
from collections import defaultdict

import csv
import Helpers
from DB import DB
import pickle
from tqdm import tqdm
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import os
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay
import random
import numpy as np
from timeit import default_timer as timer

from Helpers import *

DATASET_PATH = '../../prospeccts'
SEED = 0


np.random.seed(SEED)
random.seed(SEED)


def load_db(name):
    try:
        with open(f'database_{name}.picle', 'rb') as file:
            db = pickle.load(file)
    except:
        db = DB()
        # index_folder("3ECs", db)
        # IndexFolder('1kStructs', db)
        index_folder_pdb(f'{DATASET_PATH}/{name}/{name}', db)
        print(f'Database loaded! Size: {len(db.ids)}')
        with open(f'database_{name}.picle', 'wb') as file:
            pickle.dump(db, file)

    return db


def get_struct(dataset, id):
    p = PDB.PDBParser(QUIET=True)
    return p.get_structure(id, f'{DATASET_PATH}/{dataset}/{dataset}/{id}.pdb')


def load_similarity_dicts(name):
    with open(f'../../prospeccts/{name}/{name}.csv', 'r') as file:
        expected = list(csv.reader(file))
    active = defaultdict(list)
    inactive = defaultdict(list)
    all_ids = dict()
    count = 0
    for one in expected:
        # if one[0] in expected_dict:
        #     expected_dict[one[0]][one[1]] = one[2]
        # else:
        #     expected_dict[one[0]] = {one[1]: one[2]}
        if one[2] == 'active':
            active[one[0]].append(one[1])
            count += 1
        elif one[2] == 'inactive':
            inactive[one[0]].append(one[1])

        if one[0] not in all_ids:
            all_ids[one[0]] = 1
    print("TOTAL ACTIVE", count)
    return active, inactive, list(all_ids.keys())


def index_folder_mmcif(path, db):
    parser = PDB.MMCIFParser(QUIET=True)
    for folder in os.walk(path):

        direc = folder[0]
        files = folder[2]

        print(f'Indexing folder {direc}...')

        for file in tqdm(files):
            # print(direc + '/' + file)
            struct = parser.get_structure('test', direc + '/' + file)
            info = MMCIF2Dict(direc + '/' + file)
            seq = info['_entity_poly.pdbx_seq_one_letter_code'][0].replace('\n', '')
            db.add(seq, struct)


def struct_2_seq(structure):
    """
    Some error somewhere, it returns the sequence twice.
    """

    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    s = ''
    for res in structure.get_residues():
        if res.resname in d3to1.keys():
            s += d3to1[res.resname]

    return s


def index_folder_pdb(path, db):
    parser = PDB.PDBParser(QUIET=True)
    for folder in os.walk(path):

        direc = folder[0]
        files = folder[2]

        print(f'Indexing folder {direc}...')

        print(f"Total files: {len(files)}")

        for file in tqdm(files):
            # print(direc + '/' + file)
            struct = parser.get_structure(file, direc + '/' + file)
            seq = struct_2_seq(struct)
            name = file.split('.')[0]
            db.add(name, seq, struct)


def get_transposed_and_query_ligand_distance(db_name, pdb_code_hit, rot, trans, pdb_code_query):
    hit_ligand = get_ligand_mean_position_with_mem_cache(pdb_code_hit,
                                                         f'{DATASET_PATH}/{db_name}/{db_name}/{pdb_code_hit}.pdb')
    query_ligand = get_ligand_mean_position_with_mem_cache(pdb_code_query,
                                                           f'{DATASET_PATH}/{db_name}/{db_name}/{pdb_code_query}.pdb')

    return np.linalg.norm((hit_ligand @ rot + trans) - query_ligand)


def process_prospeccts():
    dataset = 'NMR_structures'
    cutoff = 4  # distance cutoff for residues to be considered part of the pocket, angstroms

    db = load_db(dataset)
    active, inactive, all_ids = load_similarity_dicts(dataset)

    expected = []
    scores = []

    results_log = []

    elapsed = 0

    for struct_id in tqdm(all_ids, desc='running all comparisons'):

        print(f"Processing {struct_id}")
        db_id = db.pdb_code_to_index[struct_id]
        struct = get_struct(dataset, struct_id)
        # site = Helpers.get_ligand_contacts_with_cashing(dataset, struct_id, struct, cutoff)
        site = Helpers.get_ligand_contacts_np(struct, cutoff)
        for resi in site.copy():
            if resi >= len(db.seqs[db_id]): site.remove(resi)
        if len(site) == 0: continue  # why is this happening?

        check_only = active[struct_id] + inactive[struct_id]
        start = timer()
        results = db.score_list(template_positions=db.pos[db_id],
                                seq=db.seqs[db_id],
                                site=site,
                                k_mer_min_found_fraction=.5,
                                k_mer_similarity_threshold=13,
                                cavity_scale_error=1.2,
                                min_kmer_in_cavity_fraction=.05,
                                align_rounds=1000,
                                align_sufficient_samples=15,
                                align_delta=0.15,
                                icp_rounds=3,
                                icp_cutoff=10,
                                ids_to_score=check_only)
        end = timer()
        elapsed += end - start

        for result_id in results.keys():

            expected.append(result_id in active[struct_id])
            scores.append(results[result_id]['score'])

            results[result_id]['label'] = 'positive' if result_id in active[struct_id] else 'negative'
            if 'rot' in results[result_id]:
                results[result_id]['ligand_dist'] = get_transposed_and_query_ligand_distance(dataset, result_id,
                                                                                             results[result_id]['rot'],
                                                                                             results[result_id][
                                                                                                 'trans'],
                                                                                             struct_id)

            results_log.append(results[result_id])

            if result_id == '1gvrA' or result_id == '3p74A':
                if 'ligand_dist' in results[result_id]:
                    if results[result_id]['ligand_dist'] < 3 and result_id in inactive[struct_id]:
                        write_superimposed(dataset, result_id, struct_id, results[result_id]['mapping'], results[result_id]['rot'], results[result_id]['trans'])



    RocCurveDisplay.from_predictions(expected, scores)

    plt.show()

    print(f"Total elapsed time during searching: {elapsed}, total compared: {len(results_log)}, time per comparison: {elapsed/len(results_log)*1000} ms.")

    with open(f'ProSPECCTs results/{dataset}.pickle', 'wb') as f:
        pickle.dump(results_log, f)

def write_superimposed(dataset_name, id1, id2, mapping, rot, trans, prefix=''):
    """Superimposes the first one on the second one and saves them to the
    Superimposed folder."""
    SUPERIMPOSED_FOLDER = 'Superimposed/'
    parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    io = PDB.PDBIO()
    struct1 = parser.get_structure(id1, DATASET_PATH + f'/{dataset_name}/{dataset_name}/{id1}.pdb')
    struct2 = parser.get_structure(id2, DATASET_PATH + f'/{dataset_name}/{dataset_name}/{id2}.pdb')

    for atom in struct1.get_atoms():
        atom.transform(rot, trans)

    # select mapped residues
    mapped = []
    i = 0
    for residue in struct1.get_residues():
        if i in mapping:
            mapped.append(residue)
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
    target_file = SUPERIMPOSED_FOLDER + f'{prefix}{id1} mappings superimposed to {id2}.pdb'
    io.set_structure(struct1)
    io.save(target_file, select=CavitySelect(mapped))
    target_file = SUPERIMPOSED_FOLDER + f'{id2}.pdb'
    io.set_structure(struct2)
    io.save(target_file)

if __name__ == '__main__':
    process_prospeccts()
