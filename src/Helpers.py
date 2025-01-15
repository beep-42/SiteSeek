import pickle
import Bio.PDB.Structure
from Bio import PDB
import numpy as np
from Bio.PDB import PDBParser

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


class CavitySelect(PDB.Select):

    def __init__(self, cavity) -> None:
        self.cavity = cavity

    def accept_residue(self, residue):
        return residue in self.cavity


def extract_neighborhood(structure, origin, distance):

    """
    Returns all residue numbers in range of the origin from the given structure.
    """

    neighborhood = []
    for chain in structure:

        for residue in chain.get_residues():

            if np.linalg.norm(residue.center_of_mass() - origin) <= distance:

                neighborhood.append(residue)

    return neighborhood


def get_contacts(s1, s2, distance):

    """
    Finds and returns AAs from s1 (protein) in contact with s2 (atoms).
    """

    res = []

    for atom2 in s2.get_atoms():

        for atom1 in s1.get_atoms():

            if atom1 - atom2 <= distance:

                if atom1.get_parent().id[0] == ' ':
                    res.append(atom1.get_parent())

    return res


def get_ligand_contacts(struct: Bio.PDB.Structure.Structure, distance=4.0):

    ligand = []
    for res in struct.get_residues():
        if res.id[0] != ' ' and res.id[0] != 'W':
            ligand.append(res)

    contacts = []

    i = 0
    # print("Length:", len(list(struct.get_residues())))
    for res in struct.get_residues():

        res_added = False
        for atom in res.get_atoms():
            for lig in ligand:
                for lig_atom in lig.get_atoms():
                    if atom - lig_atom <= distance:
                        if res.resname in d3to1:
                            contacts.append(i) # res.id[1])
                            # print(res.id, i)
                            res_added = True
                            break
                if res_added: break
            if res_added: break
        i += 1

    return list(set(contacts))


def get_ligand_contacts_np(struct: Bio.PDB.Structure, distance=4.0):

    ligand_coords = []
    for res in struct.get_residues():
        if res.id[0] != ' ' and res.id[0] != 'W':
            for atom in res.get_atoms():
                ligand_coords.append(atom.coord)
    ligand_coords = np.array(ligand_coords)

    max_res_length = 12

    contacts = []
    i = 0
    for res in struct.get_residues():
        found = False
        if res.id[0] == ' ':
            if np.min(np.linalg.norm(ligand_coords - np.array(res.center_of_mass()), axis=1)) <= max_res_length:
                for atom in res.get_atoms():
                    if np.min(
                        np.linalg.norm(ligand_coords - np.array(atom.coord), axis=1)) <= distance:
                        found = True
                        contacts.append(i)
                        break
                    if found: break

        i += 1

    return contacts


def save_ligand_contacts(path, pdb_id, struct: Bio.PDB.Structure, distance=4.0):

    contacts = get_ligand_contacts(struct, distance)
    with open(f'cache/{path}/{pdb_id}_contacts_{distance}', 'w') as file:

        file.write(','.join([str(x) for x in contacts]))


def load_ligand_contacts(path, pdb_id, distance=4.0):

    with open(f'cache/{path}/{pdb_id}_contacts_{distance}', 'r') as file:
        contacts = [int(x) for x in file.read().split(',')]
    return contacts


def get_ligand_contacts_with_cashing(path, pdb_id, struct: Bio.PDB.Structure, distance=4):

    try:
        contacts = load_ligand_contacts(path, pdb_id, distance=distance)
    except FileNotFoundError:
        save_ligand_contacts(path, pdb_id, struct, distance=distance)
        contacts = load_ligand_contacts(path, pdb_id, distance=distance)

    return contacts


def get_ligand_contacts_fast(struct: Bio.PDB.Structure, distance: float = 4.0):

    ligand = []
    for res in struct.get_residues():
        if res.id[0] != ' ' and res.id[0] != 'W':
            ligand += list(res.get_atoms())

    contacts = []

    ligand_mean = np.mean([x.coord for x in ligand])
    ligand_size = max([np.linalg.norm(ligand_mean - x.coord) for x in ligand])
    res_size = 12   # length of the longest residue chain / 2 + margin

    i = 0
    # print("Length:", len(list(struct.get_residues())))
    for res in struct.get_residues():

        if np.linalg.norm(res.center_of_mass() - ligand_mean) <= ligand_size + res_size:

            found = False
            for atom in res.get_atoms():
                if np.linalg.norm(atom.coord - ligand_mean) >= ligand_size: continue

                for lig_atom in ligand:
                    if atom - lig_atom <= distance:
                        contacts.append(i)
                        found = True
                        break
                if found: break

        i += 1

    return contacts


def get_ligand_mean_position(struct: Bio.PDB.Structure):

    ligand = []
    for res in struct.get_residues():
        if res.id[0] != ' ' and res.id[0] != 'W':
            ligand.append(res)

    return np.mean([x.center_of_mass() for x in ligand], axis=0)


ligand_positions_cache = {}
def get_ligand_mean_position_with_mem_cache(id, struct_path: str):

    if id in ligand_positions_cache:
        return ligand_positions_cache[id]

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    struct = parser.get_structure(id, struct_path)

    ligand = []
    for res in struct.get_residues():
        if res.id[0] != ' ' and res.id[0] != 'W':
            ligand.append(res)

    mean = np.mean([x.center_of_mass() for x in ligand], axis=0)
    ligand_positions_cache[id] = mean
    return mean


def load_ligand_mean_position(path, pdb_id):

    with open(f'cache/{path}/{pdb_id}_ligand_pos.pickle', 'rb') as file:

        return pickle.load(file)


def save_ligand_mean_position(position, path, pdb_id):

    with open(f'cache/{path}/{pdb_id}_ligand_pos.pickle', 'wb') as file:

        pickle.dump(position, file)


def struct2seq(struct: Bio.PDB.Structure):
        """
        Some error somewhere, it returns the sequence twice.
        """

        d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

        s = ''
        for res in struct.get_residues():
            if res.resname in d3to1.keys():
                s += d3to1[res.resname]

        return s
