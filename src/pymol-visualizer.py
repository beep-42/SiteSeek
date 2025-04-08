import json
import sys

import requests
import numpy as np
from pymol import cmd, cgo, finish_launching


def fetch_pdb(pdb_id: str) -> str:
    """
    Fetch the PDB file text from RCSB.

    :param pdb_id: The PDB ID.
    :return: The PDB file as a string.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Could not fetch PDB '{pdb_id}' from RCSB.")


def load_structure_from_str(pdb_str: str, object_name: str) -> None:
    """
    Load a PDB structure into PyMOL from a string.

    :param pdb_str: The PDB file content as a string.
    :param object_name: The name to assign to the loaded object.
    """
    cmd.read_pdbstr(pdb_str, object_name)


def transform_structure(object_name: str,
                        new_object_name: str,
                        rot: np.ndarray,
                        trans: np.ndarray) -> None:
    """
    Applies a rotation and translation to the coordinates of an object in PyMOL.

    :param object_name: The name of the PyMOL object to transform.
    :param new_object_name: The name to assign to the transformed object.
    :param rot: Rotation matrix (3x3 numpy array).
    :param trans: Translation vector (3-element numpy array).
    """
    # Get the current model data from PyMOL.
    model = cmd.get_model(object_name, state=1)
    # Extract the coordinates as a NumPy array.
    coords = np.array([atom.coord for atom in model.atom])
    # Apply transformation: new_coord = rot * old_coord + trans.
    transformed_coords = (np.dot(coords, rot.T) + trans).tolist()

    # Replace coordinates in the model with the transformed ones.
    for i, atom in enumerate(model.atom):
        atom.coord = transformed_coords[i]

    # Load the transformed model into a new object.
    cmd.load_model(model, new_object_name)


def get_ca_coord(model_name: str, resi: int) -> list[float] | None:
    """
    Get the coordinate of the CA atom for a given residue number.

    :param model_name: The name of the PyMOL object.
    :param resi: The residue number.
    :return: A list of 3 floats (x, y, z) if found, else None.
    """
    stored = []
    selection = f"model {model_name} and resi {resi} and name CA"
    cmd.iterate_state(1, selection, "stored.append([x,y,z])", space={"stored": stored})
    if stored:
        return stored[0]
    else:
        return None


def create_arrow(start: list[float], end: list[float], color: list[float] = [1.0, 0.0, 0.0],
                 radius: float = 0.1) -> list:
    """
    Create a CGO arrow from start to end coordinates.

    :param start: The starting coordinate [x, y, z].
    :param end: The ending coordinate [x, y, z].
    :param color: The RGB color for the arrow.
    :param radius: The radius of the arrow shaft.
    :return: A list representing the CGO object.
    """
    # PyMOL's cgo_arrow helper is not available by default, so we build
    # a simple arrow as a cylinder + cone.
    # Define parameters for the arrow.
    cone_length = 1.5 * radius  # length of the cone
    # Direction vector from start to end.
    vec = np.array(end) - np.array(start)
    length = np.linalg.norm(vec)
    if length == 0:
        return []
    # Normalize vector.
    vec_norm = vec / length
    # The cylinder ends where the cone begins.
    cylinder_end = (np.array(end) - vec_norm * cone_length).tolist()

    arrow_cgo = [
        # Cylinder (arrow shaft)
        cgo.CYLINDER,
        start[0], start[1], start[2],
        cylinder_end[0], cylinder_end[1], cylinder_end[2],
        radius,
        color[0], color[1], color[2],  # color at start
        color[0], color[1], color[2],  # color at end

        # Cone (arrow head)
        cgo.CONE,
        cylinder_end[0], cylinder_end[1], cylinder_end[2],
        end[0], end[1], end[2],
        1.5 * radius, 0.0,  # radius of base, radius of tip (0.0 for a point)
        color[0], color[1], color[2],  # color at base
        color[0], color[1], color[2],  # color at tip
        1.0,  # resolution
        0.0  # no cap
    ]
    return arrow_cgo


def show_in_pymol(pdb_id1: str,
                  pdb_id2: str,
                  query_site: list[int],
                  mapping: dict[int, int],
                  rot: np.ndarray,
                  trans: np.ndarray) -> None:
    """
    Shows the resulting superposition in PyMOL.

    :param pdb_id1: PDB ID of the first structure.
    :param pdb_id2: PDB ID of the second structure.
    :param query_site: Query site on the first structure as a list of residue numbers.
    :param mapping: Mapping between the two structures
                    (residue in structure1 -> residue in structure2).
    :param rot: Rotation matrix (3x3 numpy array).
    :param trans: Translation vector (3-element numpy array).
    """

    finish_launching(['pymol', '-q'])

    # Fetch PDB files directly from RCSB.
    pdb_str1 = fetch_pdb(pdb_id1)
    pdb_str2 = fetch_pdb(pdb_id2)

    # Delete any existing objects.
    cmd.delete("all")

    # Load structure1 and structure2 into PyMOL.
    load_structure_from_str(pdb_str1, pdb_id1)
    load_structure_from_str(pdb_str2, pdb_id2)

    # Transform structure2 with the given rotation and translation
    new_object_name = f"{pdb_id2}_sup"
    transform_structure(pdb_id2, new_object_name, rot, trans)

    # Create a selection for the query site on the first structure.
    # query_sel_str = " or ".join([f"resi {r}" for r in query_site])
    # cmd.select("query_site", f"model {pdb_id1} and ({query_sel_str})")
    # cmd.show("spheres", "query_site")
    # cmd.color("yellow", "query_site")

    # Create selections for the mapped residues in the first structure and the transformed second.
    mapping_sel1 = " or ".join([f"resi {r}" for r in sorted(mapping.keys())])
    mapping_sel2 = " or ".join([f"resi {r}" for r in sorted(mapping.values())])

    cmd.select("mapping_site1", f"model {pdb_id1} and ({mapping_sel1})")
    cmd.select("mapping_site2", f"model {new_object_name} and ({mapping_sel2})")
    cmd.show("sticks", "mapping_site1")
    cmd.show("sticks", "mapping_site2")
    cmd.color("blue", "mapping_site1")
    cmd.color("magenta", "mapping_site2")

    cmd.hide(f"lines", f"model {new_object_name}")
    cmd.show("cartoon", f"{new_object_name}")
    cmd.remove("solvent")
    #
    # Build CGO objects (arrows) between each mapping pair.
    arrow_cgo = []
    print(mapping)
    for res1 in mapping:
        res2 = mapping[res1]
        coord1 = get_ca_coord(pdb_id1, res1)
        coord2 = get_ca_coord(new_object_name, res2)
        if coord1 is None or coord2 is None:
            continue
        arrow_cgo.extend(create_arrow(coord1, coord2, color=[0.5, 0.5, 0.5], radius=0.1))

    if arrow_cgo:
        cmd.load_cgo(arrow_cgo, "mapping_arrows")

    cmd.zoom("all")
    # cmd.bg_color("white")


def run_json_selector(file: str, query: str):
    """
    Open file, ask for a specific entry in the json file and visualize it.
    """

    query_id = query # input("Query ID: ")
    n = input("Entry no: ")

    data = json.load(open(file))

    print("Showing:")
    print(data[n])

    show_in_pymol(
        query_id,
        data[n]['structure_id'],
        [],
        data[n]['int_mapping'],
        np.array(data[n]['rotation']),
        np.array(data[n]['translation'])
    )



# Example usage:
if __name__ == "__main__":
    # # For demonstration, use an identity rotation and zero translation.
    # rot = np.eye(3)
    # trans = np.zeros(3)
    #
    # # Define mapping: residue in structure1 -> residue in structure2
    # mapping = {45: 50, 46: 51, 47: 52}
    #
    # # Define query_site as a list of residue numbers in structure1.
    # query_site = [30, 31]
    #
    # show_in_pymol("1xyx", "1xyy", query_site, mapping, rot, trans)

    run_json_selector(sys.argv[1], sys.argv[2])
