#!/usr/bin/env python3
"""
This scripts generates a json file containing info about structures and their respective binding sites used to
calculate the background distribution for the searches.
"""

import json
from Bio import PDB
from Database import Database
from tqdm import tqdm
import Helpers


STRUCTURE_SET_PATH = '../../prospeccts/review_structures/review_structures/'
OUTPUT_PATH = 'background_estimation_set.json'

def get_struct(id):
    p = PDB.PDBParser(QUIET=True)
    return p.get_structure(id, f'{STRUCTURE_SET_PATH}/{id}.pdb')

if __name__ == '__main__':

    cutoff = 4  # distance cutoff for residues to be considered part of the pocket, angstroms

    # db: Database = load_db(dataset)

    db: Database = Database()
    db.add_from_directory(f'{STRUCTURE_SET_PATH}')

    compiled_set = {}


    for struct_id in tqdm(db.ids, desc='Compiling json'): # if not only_return else all_ids[:]:

        # print(f"Processing {struct_id}")
        db_id = db.pdb_code_to_index[struct_id]
        struct = get_struct(struct_id)
        # site = Helpers.get_ligand_contacts_with_cashing(dataset, struct_id, struct, cutoff)
        site = Helpers.get_ligand_contacts_np(struct, cutoff)
        for resi in site.copy():
            if resi >= len(db.sequences[db_id]): site.remove(resi)
        if len(site) == 0: continue  # why is this happening?

        # convert site to chain_res notation
        converted_site: list[str] = []
        for pos in site:
            converted_site.append(db.delimitations[db_id][pos])

        compiled_set[struct_id[0:-1]] = converted_site

    print(f"Writing output to {OUTPUT_PATH}...")

    json.dump(compiled_set, open(OUTPUT_PATH, 'w'), indent=2)

    print("Done!")