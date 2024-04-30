import csv
DATASET_PATH = '../../TOUGH M1/TOUGH-M1_dataset/'


def load_list_to_dict(path_to_tough_m1_list_file: str):
    with open(path_to_tough_m1_list_file, 'r') as file:
        expected = list(csv.reader(file, delimiter=' '))
    expected_dict = {}
    for one in expected:
        if one[0] in expected_dict:
            expected_dict[one[0]].append(one[1])
        else:
            expected_dict[one[0]] = [one[1]]

    return expected_dict


def load_protein_ligand_contacts(pdb_code, no_chains: bool = False):
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

                if no_chains:
                    contacts.append(res)
                else:
                    contacts.append([res, chain])

    return list(tuple(contacts))
