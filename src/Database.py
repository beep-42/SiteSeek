#!/usr/bin/env python3
"""
The database module containing the database interface for db initiation and searching. For more information on usage
see the README.md.
"""
import array
import concurrent
import lzma
import multiprocessing
import pickle
import sys
import time
from bisect import bisect_left
from collections import defaultdict, namedtuple
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Literal, Tuple, Dict, Callable
import re

import line_profiler
from typing_extensions import Self
from scipy.spatial import KDTree

import biotite.sequence
import biotite.database.rcsb
import numpy as np
from biotite import structure
from biotite.structure.io import pdbx, pdb
from tqdm import tqdm

import KMerMapper
from BaseClustering import BaseClustering
from BaseKMerFilter import BaseKMerFilter
from BaseRefiner import BaseRefiner
from DummyClustering import DummyClustering
from HardKMerFilter import HardKMerFilter
from Helpers import d3to1
from ICPRefiner import ICP
from KMerMapper import KmerMapperNumpy, KmerMapperArray
from KmerGen import *
from OpticsClustering import ClusteringOptics
from RandomConsensusMapper import RandomConsensusMapper, BaseMapper
from Scoring import score_hit
from SoftKMerFilter import SoftKMerFilter


__author__ = "Jakub Telcer"
__copyright__ = "Copyright 2024, Jakub Telcer"
__credits__ = ["Jakub Telcer"]
__license__ = "MIT"
__maintainer__ = "Jakub Telcer"
__email__ = "telcerj@natur.cuni.cz"
__status__ = "Pre-Alpha"
__version__ = "0.8"





@dataclass
class Result:
    """
    A dataclass containing the results of binding site search.

    Attributes:
        database_id(int): id of the structure in the database
        score(float): the score of the found binding site
        rotation(np.ndarray): A right multiplying rotation matrix from the superposition of the pockets.
        translation(np.ndarray): A translation vector from the superposition of the pockets.
        mapping(dict[str, str]): A dictionary mapping the sequence positions between the query and the hit.
        int_mapping(dict[int, int]): int form of mapping.
    """
    def __init__(self, structure_id: str, database_id: int, score: float, rmsd, rotation: np.ndarray, translation: np.ndarray, mapping: dict[str, str], int_mapping: dict[int, int]):
        self.label = None   # TODO: Remove, for debugging purposes
        self.ligand_dist = None     # TODO: Remove, for debugging purposes
        self.structure_id = structure_id
        self.database_id = database_id
        self.score = score
        self.rmsd = rmsd
        self.rotation = rotation
        self.translation = translation
        self.mapping = mapping
        self.int_mapping = {int(x): int(int_mapping[x]) for x in int_mapping}

    def __str__(self):
        return f'{self.structure_id}\t{self.score:.{2}f}\t{self.rmsd:.{2}f}\t{str(self.mapping)}\t{self.int_mapping}\t{self.rotation}\t{self.translation}\n'


    @staticmethod
    def get_header():
        return 'ID\tSCORE\tRMSD\tMAPPING\tINT_MAP\tROT\tTRANS\n'


class SeqChainDelimitation:

    def __init__(self, chain_ids: np.ndarray, chain_positions: np.ndarray) -> None:
        """
        Initializes the SeqChainDelimitation class.

        :param chain_ids: list of ids of the chains for every residue in the protein.
        :param chain_positions: list of starting positions for each chain.
        """

        self.chain_ids = list(chain_ids[chain_positions])
        self.chain_positions = list(chain_positions)
        # self.primitive_conversion_array = chain_ids
    #
    # def primitive_index_to_id(self, index):
    #
    #     chain = self.primitive_conversion_array[index]
    #     chain_position = np.sum(self.primitive_conversion_array[:index+1] == chain)
    #     return f'{chain}_{chain_position}'
    #
    # def primitive_id_to_index(self, id):
    #
    #     chain, pos = SeqChainDelimitation._validate_and_extract(id)
    #     index = 0
    #     while self.primitive_conversion_array[index] != chain:
    #         index += 1
    #     index += pos
    #     assert self.primitive_conversion_array[index] == chain
    #
    #     return index

    def __getitem__(self, index: int) -> str:
        """
        Converts the given index to chain id-pos notation like B_36 (36th residue in the chain B).

        :param index: index of the residue in the sequence of the protein.
        """

        chain_idx = len(self.chain_positions) - 1
        while self.chain_positions[chain_idx] >= index and chain_idx > 0:
            chain_idx -= 1

        # print(f"Primitive output: {self.primitive_index_to_id(index)}, Complex: {self.chain_ids[chain_idx]}_{index - self.chain_positions[chain_idx] + 1}")
        # assert f'{self.chain_ids[chain_idx]}_{index - self.chain_positions[chain_idx] + 1}' == self.primitive_index_to_id(index), "Mismatch in encoding"

        return f'{self.chain_ids[chain_idx]}_{index - self.chain_positions[chain_idx] + 1}'

    def chain_pos_identifier_to_index(self, chain_pos_identifier: str) -> int:
        """
        Converts the given chain position identifier into sequence index.
        Example: B_26 -> 136

        :param chain_pos_identifier: chain position identifier in the format CHAINCODE_INDEX.
        """

        chain_id, position = SeqChainDelimitation._validate_and_extract(chain_pos_identifier)
        chain_idx = self.chain_ids.index(chain_id)

        try:
            # rtx = self.chain_positions[self.chain_ids.index(chain_id)] + position
            # assert rtx == self.primitive_id_to_index(chain_pos_identifier), "Mismatch in decoding"
            return self.chain_positions[self.chain_ids.index(chain_id)] + position - 1
        except ValueError:
            raise ValueError(f"The identifier {chain_pos_identifier} does not match any residue in the sequence.")

    @staticmethod
    def _validate_and_extract(identifier: str) -> Tuple[str, int]:
        # Define the regex pattern for the format L_I
        # To allow only alphabet chain identifiers: r'^([A-Za-z]+)_(\d+)$'
        pattern = r'^([A-Za-z0-9]+)_(\d+)$' # support numerical identifiers

        # Match the input string against the pattern
        match = re.match(pattern, identifier)

        if match:
            # Extract L and I from the matched groups
            chain_id = match.group(1)  # The letter code
            position = int(match.group(2))  # The positive integer
            return chain_id, position
        else:
            raise ValueError("Residue identifier is not in the correct format CHAINCODE_INDEX.")


class Database:
    """
    Database class for storing and searching similar sites.
    """

    def __init__(self) -> None:
        self.position_dtype: np.dtype = np.float16

        self.kmer_db: KMerMapper = KmerMapperArray() # KmerMapper(k = 3)
        self.ids: List[str] = []
        self.pdb_code_to_index: Dict[str, int] = {}
        self.sequences: List[str] = []
        self.delimitations: List[SeqChainDelimitation] = []
        self.pos_vectors: List[np.ndarray] = []
        # self.kdtrees = []

    def add(self, name: str, sequence: str, delimitations: SeqChainDelimitation, ca_positions: np.ndarray, kmers: List[str] | None = None) -> None:
        """
        Add the given entry to the database.
        :param name: ID of the structures
        :param sequence: Sequence of the structure
        :param delimitations: SeqChainDelimitation associated with the protein
        :param ca_positions: Numpy array of 3D  coordinates of the C-alpha atoms of the structure
        :param kmers: List of Kmers generated from the given sequence. Kmers will be generated if None.
        """

        if len(sequence) != len(ca_positions):
            raise Exception("The length of the sequence and the length of the C-alpha positions do not match.")

        index = len(self.sequences)
        self.ids.append(name)
        self.sequences.append(sequence)
        self.delimitations.append(delimitations)

        if kmers is None:
            kmers = generate_kmers(sequence)

        for km in set(kmers):   # filter the K-mers so we cannot assign any id to a kmer twice

            if self.kmer_db.valid_kmer(km):
                self.kmer_db.append(km, index)

        self.pos_vectors.append(ca_positions.astype(self.position_dtype))
        self.pdb_code_to_index[name] = index

        # build and index kd-tree
        # tree = KDTree(ca_positions)
        # self.kdtrees.append(tree)

    @staticmethod
    def _process_atom_array(atoms: biotite.structure.AtomArray, sequence: str | None = None)\
            -> Tuple[str, SeqChainDelimitation, np.ndarray]:
        """
        Processes the Biotite's AtomArray and convert it to Ca coordinates np.array and get the sequence.
        Ignores the hetero atoms.

        :param atoms: Biotite's AtomArray of the structure
        :param sequence: Sequence of the structure, automatically inferred if None
        :return: The sequence string and Ca coordinates np.array
        """
        # remove hetero atoms and all non-CA atoms
        filtered_atoms = atoms[(atoms.atom_name == 'CA') & (atoms.hetero == False)]

        if sequence is None:
            seqs, chain_pos = structure.to_sequence(filtered_atoms, allow_hetero=False)  # hetero atoms should NOT be present
            delimitations = SeqChainDelimitation(filtered_atoms.chain_id, chain_pos)

            sequence = ''.join([str(x) for x in seqs])
            # sequence = sequence.replace('X', '')

            # het = tuple(biotite.sequence.find_symbol(seq, 'X'))
            # sequence = ''
            # for i in range(len(seq)):
            #     if i not in het:
            #         sequence += seq[i]

        else:
            chain_indices = list(sorted([filtered_atoms.chain_id.index(x) for x in set(filtered_atoms.chain_id)]))
            delimitations = SeqChainDelimitation(filtered_atoms.chain_id, chain_indices)

        # ignore hetero atoms
        ca = structure.coord(filtered_atoms)  # & structure.filter_canonical_amino_acids(array)])

        # check any potential length mismatches
        if len(sequence) != len(ca):
            raise Exception("The length of the sequence and the number of Ca atoms do not match!")

        return sequence, delimitations, ca

    @staticmethod
    def _load_atom_array(file_path: str):
        """
        Loads the atom array, if multiple models are present the first one is selected.

        :file_path: Path to the structure file (PDB and PDBx supported).
        :return: The atom array of the structure.
        """
        array = structure.io.load_structure(file_path)
        # if multiple models
        if type(array) is biotite.structure.AtomArrayStack:
            # print(f"Found multiple models in {file_path}, using only the first one!", file=sys.stderr)
            array = array[0]

        return array

    @staticmethod
    def _fetch_atom_array(pdb_id: str) -> structure.AtomArray:
        """
        Fetched the given structure from the RCSB PDB. If multiple models are present the first one is selected.

        :param pdb_id: PDB ID of the structure to fetch.
        :return: The biotide.structure.AtomArray of the structure.
        """
        bytes_io = biotite.database.rcsb.fetch(pdb_id, 'bcif', verbose=False)
        bcif = biotite.structure.io.pdbx.BinaryCIFFile().read(bytes_io)
        atoms = biotite.structure.io.pdbx.get_structure(bcif)
        if type(atoms) is biotite.structure.AtomArrayStack:
            # print(f"Found multiple models in {pdb_id}, using only the first one!", file=sys.stderr)
            atoms = atoms[0]

        return atoms

    @staticmethod
    def _parallel_addition_worker(id_args_queue: multiprocessing.Queue,
                                  get_function: Callable[[str], structure.AtomArray])\
            -> List[Tuple[str, str, np.ndarray, np.ndarray]]:

        # identifier, sequence, ca, generated k-mers
        out: List[(str, str, np.ndarray, List[str])] = []

        while not id_args_queue.empty():
            try:
                pdb_id, arg = id_args_queue.get_nowait()
                try:
                    seq, delimitation, struct  = Database._process_atom_array(get_function(arg))
                    out.append((pdb_id, seq, delimitation, struct, generate_kmers(seq)))
                except Exception as e:
                    print(f"Failed to add {pdb_id}, skipping. Reason: {e}", file=sys.stderr)
            except Exception as e: # queue empty
                pass

        return out

    def _parallel_addition(self,
                           id_arg_list: List[Tuple[str, str]],
                           loading_function: Callable[[str], structure.AtomArray],
                           message: str,
                           n_jobs : int = -1,
                           memory_split: int = 10000) -> None:

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        print(f'Running on {n_jobs} processes!', file=sys.stderr)

        total = len(id_arg_list)
        pbar = tqdm(desc=message, total=total)

        with multiprocessing.Manager() as manager:

            # queue with pdb_id, pdb_id pair (first is id, the second is arg to the fetch function)
            queue = manager.Queue()

            # split the addition into chunks to prevent memory overflow (unprocessed data are larger)
            for split_position in range(0, total, memory_split):

                # insert the id-arg pairs to the queue
                for item in id_arg_list[split_position:split_position+memory_split]: queue.put(item)
                current_total = queue.qsize()

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    # Map the function f to the chunks
                    future_to_queue = [
                        executor.submit(
                            Database._parallel_addition_worker, id_args_queue = queue,
                            get_function = loading_function
                        ) for _ in range(n_jobs)
                    ]

                    while sum([not future.done() for future in future_to_queue]) > 0:
                        # Update the progress bar
                        pbar.n = split_position + current_total - queue.qsize()
                        pbar.refresh()
                        time.sleep(1)

                    pbar.n = split_position + current_total - queue.qsize() # qsize should be zero but rather leave it here
                    pbar.refresh()

                    ubar = tqdm(desc='Unifying results', total=current_total)
                    for future in concurrent.futures.as_completed(future_to_queue):
                        for one_struct in future.result():
                            self.add(*one_struct) # one_struct[0], one_struct[1], one_struct[2], one_struct[3])
                            ubar.update(1)

                    ubar.close()

        pbar.close()

        # self.kmer_db.drop_empty()

    def add_file(self, file_path: str | None, database_identifier: Optional[str] = None, sequence: Optional[str] = None) -> None:
        """
        Adds a PDB/mmCIF/BinaryCIF file to the database. Database identifier and sequence are not required, if not
        provided they are inferred from the given file.

        :param file_path: Path to the PDB/mmCIF/BinaryCIF file
        :param database_identifier: Database identifier (that is later used to identify the structure in the database).
        The filename without suffix is used if None.
        :param sequence: Aminoacid sequence of the structure (passed as a string containing single letter residue
        codes like A, K, N, ...). Inferred from the file if None.

        :return: None
        """

        sequence, delimitation, ca_positions = Database._process_atom_array(Database._load_atom_array(file_path))

        if database_identifier is None: database_identifier = Path(file_path).stem

        self.add(database_identifier, sequence, delimitation, ca_positions)


    def fetch_structures(self, pdb_ids: List[str], n_jobs : int = -1, memory_split: int = 10000) -> None:
        """
        Fetches the provided structures from the RCSB PDB and adds them to the database.

        :param pdb_id: PDB structure identifier or List of identifiers.
        """

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        id_arg_list = list(zip(pdb_ids, pdb_ids))

        self._parallel_addition(id_arg_list = id_arg_list,
                                loading_function = Database._fetch_atom_array,
                                message = 'Downloading and processing',
                                n_jobs = n_jobs,
                                memory_split = memory_split)


    def add_from_directory(self, directory_path: str, n_jobs: int = -1, memory_split: int = 10000) -> None:
        """
        Adds all PDB/mmCIF/BinaryCIF files to the database parallely.

        :param directory_path: Path to the directory containing the files.
        :param n_jobs: Number of jobs to run in parallel.
        :return: None
        """

        id_path_list: List[tuple[str, str]] = []
        dir_path = Path(directory_path)
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                id_path_list.append((file_path.stem, file_path))

        self._parallel_addition(id_arg_list = id_path_list,
                                loading_function = Database._load_atom_array,
                                message = 'Processing',
                                n_jobs = n_jobs,
                                memory_split = memory_split)

    @staticmethod
    def _structure_to_pos(structure):
        """Converts pdb structure to np array of positions"""
        positions = []

        for res in structure.get_residues():
            if res.resname in d3to1:
                positions.append(res.center_of_mass())

        return np.array(positions)

    @staticmethod
    def _find_potential_matches(positions: list[int],
                                seq1: str, seq2: str,
                                k_mer_similarity_threshold: int,
                                kmers: list[str],
                                add_all_points=True,
                                restrict_seq2:list[int] | None = None) -> defaultdict[int, list[int]]:

        """
        Generates non-redundant dictionary, that maps every position from positions on the first sequence, to positions
        on the second, provided the kmers around these positions are at least as similar as k_mer_similarity_threshold.
        :param positions: Positions on the first sequence to match
        :param seq1:
        :param seq2:
        :param k_mer_similarity_threshold: Minimal required similarity
        :param kmers: All kmers generated from the second sequence
        :param add_all_points: Whether to match only center points, or all points from the kmer (this trimer
        :param restrict_seq2: Whether to restrict the matches on the second sequence only to the given list
        implementation matches all three)

        :return: Mapping of the points.
        """

        potential_matches = defaultdict(list)


        for res_pos in positions:

            kmer = get_one_trimer_slice(seq1, res_pos)  # kmer around the position

            if restrict_seq2 is None:
                kmer_walk = enumerate(kmers)
            else: # restrict to the given indicies only
                kmer_walk = zip(restrict_seq2, [kmers[x] for x in restrict_seq2])

            for ref_pos, ref_kmer in kmer_walk: # enumerate(kmers):
                if score_seq_similarity(kmer, ref_kmer) >= k_mer_similarity_threshold:

                    if add_all_points:

                        for shift in [-1, 0, 1]:

                            if 0 <= res_pos + shift < len(seq1) and 0 <= ref_pos + shift < len(seq2):
                                potential_matches[res_pos + shift].append(ref_pos + shift)

                    else:

                        potential_matches[res_pos].append(ref_pos)

        # remove redundancies
        for m in list(potential_matches.keys()):
            potential_matches[m] = list(set(potential_matches[m]))

        return potential_matches

    @staticmethod
    def _find_potential_matches_dict(positions: list[int], kmers_to_positions: dict[str, set[int]], sequence: str, k:int=3) \
            -> defaultdict[int, list[int]]:

        potential_matches = defaultdict(set)
        # for i in range(0, len(sequence) - k + 1):
        #
        #     if sequence[i:i + k] in kmers_to_positions:
        #         potential_matches[]

        for res_pos in positions:
            if sequence[res_pos - 1: res_pos + 2] in kmers_to_positions:
                potential_matches[res_pos].update(kmers_to_positions[sequence[res_pos - 1: res_pos + 2]])
        for one in potential_matches:
            potential_matches[one] = list(potential_matches[one])
        print(potential_matches)
        return potential_matches

    @staticmethod
    def _prepare_kmers_to_positions_dict(kmers_dict, sequence, k=3):

        prepared = defaultdict(set)
        for i in range(0, len(sequence) - k + 1):
            slice = sequence[i:i + k]
            if slice in kmers_dict:
                prepared[slice].add(i + 1)
                for one in kmers_dict[slice]:
                    prepared[one].add(i+1)
        return prepared

    def _convert_mapping(self, hit_id: int, query_delimitations: SeqChainDelimitation, mapping: Dict[int, int]) -> Dict[str, str]:

        resulting_mapping = {}
        for key in mapping:

            resulting_mapping[query_delimitations[key]] = self.delimitations[hit_id][mapping[key]]

        return resulting_mapping


    @line_profiler.profile
    def search_pipe(self, template_positions: list[np.ndarray],
                    seq: str,
                    delimitations: SeqChainDelimitation,
                    site: list[int],
                    k_mer_similarity_threshold: float,
                    filter: BaseKMerFilter,
                    clustering: BaseClustering,
                    mapper: BaseMapper,
                    refiner: BaseRefiner,
                    search_subset: Optional[List[str]] = None,
                    progress: bool = False) -> List[Result]: # dict[str, Result]:

        """
        Calculates scores for each structure ID supplied in ids_to_score list. Structures that do not receive any score
        (due e.g., low similar Kmer content) receive a default score of -1.

        :param template_positions: List of vectors of positions of residues in the query structures
        :param seq: The sequence of the query structure
        :param delimitations: The delimitations of the chains in the query structure
        :param site: List of indices of the residues participating in the ligand binding (indices of pocket residues)
        :param k_mer_min_found_fraction: Minimal allowed fraction of all similar Kmers found in the target sequence,
        compared to the total count of searched Kmers originating from the query cavity (all Kmers around residues
        defined in the site list)
        :param k_mer_similarity_threshold: What is the minimal similarity between Kmers to be considered similar and
        to be potentially mapped on each other. Similarity is the sum of scores from the BLOSUM 62 table for each
        pair of aligned residues in directly aligned Kmers (with no shifts or gaps).
        :param cavity_scale_error: Scale factor for the possible change of size of the target cavity compared to the
        query cavity. Measured as the fraction of the largest distance of any two residues in the cavities.
        :param min_kmer_in_cavity_fraction: The minimal allowed fraction of similar Kmers in the target cavity to be
        considered as a potentially similar cavity. Measured as fraction of Kmers in the target to the number of Kmers
        from the query cavity.
        :param align_rounds: The number of rounds to perform using the RandomConsensusAligner
        :param align_sufficient_samples: After how many successful overlaps to terminate
        :param align_delta: the allowed fractional error of the distance between labeled points (centers of Kmers) in
        the target cavity and distance between points in the query cavity, normalized to the distance in the query
        cavity.
        :param icp_rounds: How many rounds of the Iterative Closest Point algorithm to perform
        :param icp_cutoff: Maximal distance of two similar Kmers to be still mapped in the Iterative Closest Point
        algorithm
        :param top_k: How many top hits for each Kmer to consider during clustering. None = consider all.
        :param ids_to_score: list of ids to score against the query

        :return: A dictionary of dicts of scores and found rotation, translation and mapping (when applicable) for
        each ID from the ids_to_score list.
        """

        # results: dict[str, Result] = {} # defaultdict(re)
        # for sid in ids_to_score:
        #     results[sid] = Result(-1, None, None, None)

        kmer_tuple, kmers_dict, scored_kmer_dict = generate_similar_kmers_around(seq, tuple(site), k_mer_similarity_threshold)
        # todo: return flattened as well? Use a flattening function?
        all_kmers: List[str] = []
        for kmer_key in kmers_dict:
            all_kmers += kmers_dict[kmer_key]

        # PART 1: K-mer filtering
        subset: Optional[List[int]] = [self.pdb_code_to_index[x] for x in search_subset] if search_subset is not None else None
        all_seq, kmer_found = filter.search_kmers(self, kmer_tuple, kmers_dict, rated_kmers_dict=scored_kmer_dict, use_subset=subset)
        # print(f"Total identifier by kmer pref-filter: {len(all_seq)}")

        # PART 2: CLUSTERING
        clustering.set_template(template_positions, site)
        accepted_clusters = clustering.cluster(all_seq, self.pos_vectors, self.sequences,
                                               all_kmers, rated_kmer_dict=scored_kmer_dict)


        # PART 3: MAPPING
        kmers: List[str] = generate_kmers(seq)
        mapper.set_source(template_positions)

        # kmers_to_positions_dict = Database._prepare_kmers_to_positions_dict(kmers_dict, seq)

        results: List[Result] = []

        for cluster in accepted_clusters if not progress else tqdm(accepted_clusters, desc='Evaluating hits'):

            hit = list(cluster.keys())[0]  # seq. index

            # TODO: The positions are used from the clustering instead of using position from the binding site! -> sometimes leads to better results
            positions = cluster[hit]

            # Finds matches in the found cluster in the hit matching ANYTHING in the query sequence.
            # potential_matches = Database._find_potential_matches(positions, self.sequences[hit], seq, k_mer_similarity_threshold,
            #                                                     kmers, add_all_points=False)

            # Strict version: Finds only matches between the cluster and the binding site
            potential_matches = Database._find_potential_matches(positions, self.sequences[hit], seq,
                                                                 k_mer_similarity_threshold,
                                                                 kmers, add_all_points=False,
                                                                 restrict_seq2=site)


            # print(Database._find_potential_matches(positions, self.seqs[hit], seq, k_mer_similarity_threshold,
            #                                                      kmers, add_all_points=False))
            # potential_matches = Database._find_potential_matches_dict(positions, kmers_to_positions_dict, self.seqs[hit])

            # filter to have at least min_kmer_in_cavity_fraction
            # unique_found_positions = set([item for sublist in potential_matches.values() for item in sublist])
            # if len(unique_found_positions) <= len(site) * min_kmer_in_cavity_fraction:
            #     continue

            rmsd, rot, trans, mapping, rounds = mapper.match_to_source(
                potential_matches, self.pos_vectors[hit])

            # rmsd, rot, trans, mapping, rounds, votes, avg_dev, rms_dev, med_dev = MostCommonMappingAligner.get_rottrans_and_mapping(potential_matches,
            #                                                                                                             self.pos[hit],
            #                                                                                                             template_positions,
            #                                                                                                             rmsd_cutoff = 1.5,
            #                                                                                                             max_dev = 0.15,
            #                                                                                                             rounds = 1000,
            #                                                                                                             min_votes=15
            #                                                                                                             )

            if rot is None:  # no hit
                continue

            if refiner is not None:
                mapping, rmsd, rot, trans = refiner.refine(potential_matches, self.pos_vectors[hit],
                                                           template_positions, rot, trans)
                if mapping is None:
                    continue

            found_kmer_fraction = kmer_found[hit] # / len(site)
            mapping_coverage = len(mapping) / len(site)
            votes_ratio = len(mapping) / len(site)
            score = score_hit(found_kmer_fraction, mapping_coverage, votes_ratio,
                              self.pos_vectors[hit], template_positions, self.sequences[hit], seq, site, mapping,
                              rot, trans)

            results.append(
                Result(self.ids[hit], hit, score, rmsd, rot, trans, self._convert_mapping(hit, delimitations, mapping), mapping)
            )

            # if self.ids[hit] not in results or results[self.ids[hit]].score < score:    # accumulate the hit with the highest score for each structure
            #     results[self.ids[hit]].score = score
            #     results[self.ids[hit]].rot = rot
            #     results[self.ids[hit]].trans = trans
            #     results[self.ids[hit]].mapping = mapper
            #     results[self.ids[hit]].database_id = self.ids[hit]

        return results


    def search(self, structure_id: str | None, site: list[str], k_mer_similarity_threshold: float,
               search_subset: list[str] | None = None, positions: np.typing.NDArray = None, sequence: str = None,
               delimitations: SeqChainDelimitation | None = None,
               lr: float = 0.9, skip_clustering: bool = False, skip_icp: bool = False,
               ransac_min: int = 15, progress: bool = True) -> List[Result]:
        """
        Searches the database for similar substructures. The query is provided using Biopython's structure and indices
        of aminoacids forming the query substructures. The searched regions can be discontinuous.

        Internally this method uses the Database.search_pipe and initializes it's own KmerFilter, Clustering, Mapper
        and Refiner with the default parameters.

        :param structure: The template positions of the query substructures.
        :param site: The sequential indices of the aminoacids forming the substructure.
        :param k_mer_similarity_threshold: The similarity threshold for the KmerFilter. The lower it is, the more noise
        is picked up.
        :param search_subset: If only a subset of the database should be searched, provide the structure IDs of such
        subset.
        :progress: Whether to display a progress bar.

        :return: A list of Results.
        """

        # positions = Database._structure_to_pos(structure)
        # sequence = Helpers.struct2seq(structure)

        positions = self.pos_vectors[self.pdb_code_to_index[structure_id]] if structure_id is not None else positions
        sequence = self.sequences[self.pdb_code_to_index[structure_id]] if structure_id is not None else sequence
        delimitations = self.delimitations[self.pdb_code_to_index[structure_id]] if structure_id is not None else delimitations

        converted_site = []
        for chain_res_id in site:
            converted_site.append(delimitations.chain_pos_identifier_to_index(chain_res_id))

        # filtering = HardKMerFilter(k_mer_min_found_fraction=.5)
        filtering =  SoftKMerFilter(overall_required_similarity=lr)

        clustering = ClusteringOptics(cavity_scale_error=1.1,
                                      min_kmer_in_cavity_fraction=.01,
                                      top_k=3)
        if skip_clustering: clustering = DummyClustering(top_k=3)

        mapper = RandomConsensusMapper(allowed_error=.15,
                                       rounds=1500,
                                       stop_at=ransac_min,
                                       polygon=3)
        refiner = ICP(rounds=3, cutoff=10) if not skip_icp else None

        return self.search_pipe(positions, sequence, delimitations, converted_site, k_mer_similarity_threshold, filtering, clustering,
                                mapper, refiner, search_subset=search_subset, progress=progress)

    def get_sequence(self, db_id):
        return self.sequences[db_id]

    def __len__(self) -> int:
        """
        Returns the size of the database.
        """

        return len(self.ids)

    def save(self, file_path: str, compress: bool = True) -> None:
        """
        Saves the database into a file.

        :param file_path: The path to the file.
        :param compress: Whether to compress the file.
        """

        if compress:
            with lzma.open(file_path, "wb") as file:
                pickle.dump(self, file)
        else:
            with open(file_path, "wb") as file:
                pickle.dump(self, file)

    @staticmethod
    def load(file_path: str, compressed: bool = True) -> Self:
        """
        Loads the database from a file and lods it's datafields.

        :param file_path: The path to the file.
        :param compressed: Whether the database file is compressed.
        """

        if compressed:
            with lzma.open(file_path, "rb") as file:
                db: Database = pickle.load(file)
        else:
            with open(file_path, "rb") as file:
                db: Database = pickle.load(file)

        # TODO: Fix this sketchy implementation
        # Just load the db data fields
        new_db = Database()
        new_db.position_dtype = db.position_dtype
        new_db.kmer_db = db.kmer_db
        new_db.ids = db.ids
        new_db.pdb_code_to_index = db.pdb_code_to_index
        new_db.sequences = db.sequences
        new_db.delimitations = db.delimitations
        new_db.pos_vectors = db.pos_vectors

        return new_db