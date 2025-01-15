#! /bin/env/python3
import lzma
import pickle
import random
import sys
import tracemalloc
from copy import deepcopy
from timeit import Timer
from typing import Callable

import numpy as np

import Helpers
import KMerMapper
from Database import Database, Result
import biotite.database.rcsb
import biotite.structure.io.pdbx
from io import BytesIO
import json
import requests
from tqdm import tqdm


def get_pdb_ids():
    """
    Downloads and returns all PDB ids (in memory).

    Returns:
        List of the IDs
    """

    url = 'https://data.rcsb.org/rest/v1/holdings/current/entry_ids'
    return json.loads(requests.get(url).text)

def get_protein_ids():
    """
    Returns only protein IDs from the RCSB PDB database.

    Returns: List of IDs
    """

    query = biotite.database.rcsb.FieldQuery("rcsb_entry_info.selected_polymer_entity_types", exact_match="Protein (only)")
    return biotite.database.rcsb.search(query)

def downloadPDB():
    """
    Downloads the PDB files into memory and indexes them to the database.
    """

    db: Database = Database()
    batch_size: int = 1
    # print(', '.join(get_pdb_ids()))

    ids = get_protein_ids()
    for i in tqdm(range(0, len(ids), batch_size)):

        batch = ids[i:i+batch_size]
        print(batch)

        # biotite.database.rcsb.fetch(batch, 'bcif', 'fetched', verbose=True)
        db.add_file(f'fetched/{batch[0]}.bcif')

        # files = biotite.database.rcsb.fetch(batch, 'bcif', verbose=True)
        #
        # for sub_i, file in enumerate(files):
        #
        #     pdb_id = ids[i + sub_i]
        #
        #     bcif = biotite.structure.io.pdbx.BinaryCIFFile().read(file)
        #     atoms = biotite.structure.io.pdbx.get_structure(bcif)
        #     mc = biotite.structure.io.pdbx.get_model_count(bcif)
        #
        #     if mc > 1: print(f"Found {mc} models!")
        #
        #     atoms = atoms.get_array(0)
        #
        #     # if len(biotite.structure.get_chains(atoms)) > 1:
        #     #     print("Multiple chains, skipping")
        #     #     continue
        #
        #     try:
        #         db.add_file(None, atoms=atoms, database_identifier=pdb_id)
        #     except Exception as e:
        #         print(f"Caught exception: {e} for PDB ID {pdb_id}, skipping...")

def offline_add():

    db: Database = Database()

    # print("Fetching PDB ids...")
    # ids = get_protein_ids()
    # biotite.database.rcsb.fetch(ids[1300:], 'bcif', 'fetched/', verbose=True)

    db.add_from_directory(f'fetched/')

    # db.add_dir_mp('fetched/', n_jobs=12)
    # db.add_file('../../1kStructs/1TAP.cif')

    print("Database size:")
    print(len(db))

    return db

def fetch_db(size: int | None = 20000, db_path: str | None = None):

    print("Random subsampling version.")
    db: Database = Database()
    if db_path is not None:
        with lzma.open(db_path) as handle:
            db = pickle.load(handle)
        db.search(db.ids[0], [1, 2, 3], 13)
        db.kmer_db.bypass_pickle_view()

    ids_to_fetch = get_protein_ids()
    for one_id in ids_to_fetch.copy():
        if one_id in db.pdb_code_to_index: ids_to_fetch.remove(one_id)
    if size is not None and size < len(ids_to_fetch): ids_to_fetch = random.sample(ids_to_fetch, size)# ids_to_fetch = ids_to_fetch[:size]

    db.fetch_structures(ids_to_fetch)
    name = str(size) if size is not None else 'full'
    with lzma.open(f'PDB/database_{name}.pickle' if not db_path else db_path, 'wb') as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)

def print_db_memory_stats(db_create_func: Callable[[],Database]) -> None:

    # print(f'Database size: \t{len(db)}\n'
    #       f'Database mem footprint: \t{sys.getsizeof(db) / 10e6} MB\n'
    #       f'K-mer table size: \t{sys.getsizeof(db.kmer_db) / 10e6} MB\n'
    #       f'Ids size: \t{sys.getsizeof(db.ids) / 10e6} MB\n'
    #       f'Pdb code to index size: \t{sys.getsizeof(db.pdb_code_to_index) / 10e6} MB\n'
    #       f'Sequences size: \t{sys.getsizeof(db.sequences) / 10e6} MB\n'
    #       f'Pos_vectors size: \t{sys.getsizeof(db.pos_vectors) / 10e6} MB\n'
    #       f'Kdtrees size: \t{sys.getsizeof(db.kdtrees) / 10e6} MB\n')

    tracemalloc.start()

    c = tracemalloc.get_traced_memory()[0]
    db: Database = db_create_func()
    print(f"Database memory utilization: {tracemalloc.get_traced_memory()}")
    print(f"Current memory utilization:  {(tracemalloc.get_traced_memory()[0]) / 1000000} MB")

    # dists = list()
    # for one in db.pos_vectors:
    #     for i in range(len(one) - 1):
    #         dists.append(np.linalg.norm(one[i] - one[i + 1]))
    # import pandas as pd
    # df = pd.DataFrame(dists)
    # print(df.describe())

    c = tracemalloc.get_traced_memory()[0]
    del db.sequences
    print(f"Database memory utilization without sequences: {(c - tracemalloc.get_traced_memory()[0]) / 1000000} MB")
    c = tracemalloc.get_traced_memory()[0]
    del db.pos_vectors
    print(f"Database memory utilization without pos_vectors:  {(c - tracemalloc.get_traced_memory()[0]) / 1e6} MB")
    c = tracemalloc.get_traced_memory()[0]
    del db.kdtrees
    print(f"Database memory ulitization without kdtrees:  {(c - tracemalloc.get_traced_memory()[0]) / 1e6} MB")
    c = tracemalloc.get_traced_memory()[0]
    del db.kmer_db
    print(f"Database memory utilization without kmer_db:  {(c - tracemalloc.get_traced_memory()[0]) / 1e6} MB")

    print(f"Current memory utilization:  {(tracemalloc.get_traced_memory()[0]) / 1e6} MB")

def test_4k_search(db_path='database_20000.pickle', dataset = 'kahraman_structures', db = None):

    n = 100
    cutoff = 4.0 # A
    k_mer_sim = 13
    lr = 0.90

    import tracemalloc
    tracemalloc.start()

    from copy import deepcopy

    print("Loading the database!")

    print(f"Current memory utilization: {tracemalloc.get_traced_memory()}")
    if db is None:
        with lzma.open(db_path, 'rb') as handle:
            db: Database = pickle.load(handle)

    print(f"Database size: {len(db)}")

    # # convert the positions into float16
    # for i in range(len(db.pos_vectors)):
    #     db.pos_vectors[i] = db.pos_vectors[i].astype(np.float16)
        # db.kdtrees[i] = KDTree(db.pos_vectors[i].astype(np.uint8))
    #
    # c = tracemalloc.get_traced_memory()[0]
    # print(f"Database memory utilization: {tracemalloc.get_traced_memory()}")
    # print(f"Current memory utilization:  {(tracemalloc.get_traced_memory()[0]) / 1000000} MB")
    #
    # c = tracemalloc.get_traced_memory()[0]
    # # kmap = KMerMapper.KmerMapper()
    # # for kmer in db.kmer_db:
    # #     kmap[kmer] = np.array(deepcopy(db.kmer_db[kmer]), copy=True, dtype=np.uint16)
    # # with open('test-kmap', 'wb') as handle:
    # #     pickle.dump(kmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # # print(f"Footprint of the new KMer map: {(tracemalloc.get_traced_memory()[0] - c) / 1e6} MB")
    # # c = tracemalloc.get_traced_memory()[0]
    #
    #
    # del db.sequences
    # print(f"Database memory utilization without sequences: {(c - tracemalloc.get_traced_memory()[0]) / 1000000} MB")
    # c = tracemalloc.get_traced_memory()[0]
    # del db.pos_vectors
    # print(f"Database memory utilization without pos_vectors:  {(c - tracemalloc.get_traced_memory()[0]) / 1e6} MB")
    # # c = tracemalloc.get_traced_memory()[0]
    # # del db.kdtrees
    # # print(f"Database memory ulitization without kdtrees:  {(c - tracemalloc.get_traced_memory()[0]) / 1e6} MB")
    # c = tracemalloc.get_traced_memory()[0]
    # del db.kmer_db
    # print(f"Database memory utilization without kmer_db:  {(c - tracemalloc.get_traced_memory()[0]) / 1e6} MB")
    #
    # print(f"Current memory utilization:  {(tracemalloc.get_traced_memory()[0]) / 1e6} MB")
    #
    # with lzma.open(db_path, 'rb') as handle:
    #     db: Database = pickle.load(handle)


    from TestProSPECCTs import load_similarity_dicts, get_struct
    from Helpers import get_ligand_contacts_np
    import random
    from timeit import default_timer as timer

    active, inactive, all_ids = load_similarity_dicts(dataset)

    elapsed = 0
    times = []
    total_results = 0

    for struct_id in tqdm(random.sample(all_ids, n)):
        struct = get_struct(dataset, struct_id)
        # site = Helpers.get_ligand_contacts_with_cashing(dataset, struct_id, struct, cutoff)
        pos = db._structure_to_pos(struct)

        seq = Helpers.struct2seq(struct)

        site = (get_ligand_contacts_np(struct, cutoff))
        for resi in site.copy():
            if resi >= len(seq): site.remove(resi)
        if len(site) == 0: continue  # why is this happening?

        # print(f"Site: {site}, pos: {pos[:10]}, seq: {seq[:10]}")

        start = timer()
        search_result = db.search(None, site, k_mer_sim, positions=pos, sequence=seq, lr=lr)
        current = timer() - start
        elapsed += current
        times.append(current)
        total_results += len(search_result)

    print(f"Total elapsed time: {elapsed:.2f} seconds. Time per query: {elapsed / n:.2f} seconds."
          f"Time per structure pair: {elapsed / n / len(db) * 1000:.3f} milliseconds.")
    print(f"Time per structural comparison: {elapsed / total_results * 1000:.3f} milliseconds.") if total_results > 0 else None
    print(f"Linear estimation for 198,000 structures: {elapsed / len(db) / n * 198000:.2f} seconds.")

    print(f"Recorded times: {str(times)}")

    print("DB info:")
    print(f"\tNumber of structures: {len(db)}\n"
          f"\tAverage length: {np.average([len(x) for x in db.sequences])}"
          f"\tMedian length: {np.median([len(x) for x in db.sequences])}")

if __name__ == '__main__':

    # print_db_memory_stats(offline_add)

    def get_db():
        db = Database()
        db.add_from_directory(f'fetched/')
        return db

    # db = Database()
    # db.add_from_directory('../../1kStructs/')
    # test_4k_search(None, 'identical_structures', db)
    print_db_memory_stats(lambda: get_db())

    # fetch_db(1000, 'PDB/database_1000.pickle')
    # offline_add()
    # test_4k_search('PDB/database_10000.pickle')
    # downloadPDB()
