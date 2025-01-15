**Warning: This repository is under active development and no release is currently available!**
Please note that the repository is still in development and the code is subject to change. Use at your own risk.

# SiteSeek

A proof-of-concept tool for searching of large protein structure databases without the need of prediction of all the putative binding sites.

The current implementation currently only supports testing on the ProSPECCTs dataset and cannot be used as a standalone tool.
This software is provided as a code reference for Bachelor's thesis *Detection of Similar Binding Sites in
Protein Structure Databases* by Jakub Telcer.

<!-- TOC -->
* [SiteSeek](#siteseek)
  * [Prerequisites](#prerequisites)
  * [Program Structure](#program-structure)
  * [How to Run](#how-to-run)
    * [Using the Database](#using-the-database)
    * [Evaluation on Existing Datasets](#evaluation-on-existing-datasets)
  * [Author and Affiliation](#author-and-affiliation)
  * [License](#license)
<!-- TOC -->

## Prerequisites

Python 3.10 is required. The full list of requirements is in `requirements.txt`, to install them, run:

`python3 -m pip install -r requirements.txt`

Most notably, Numpy, Pandas, Scikit-learn and BioPython are required.

## Program Structure

The program does not contain any `main.py` file, as it is not meant to be used as a standalone program or tool.
It contains the implemented database for storing protein structures and supporting fast searching of 3D structural
(and sequential) motives. 

The database is implemented in the file `DB.py`. Many modules providing parts of the functionality of the search are
implemented in their respective files. Such as the `RandomConsensusAligner.py`, which contains class used to find the
optimal mapping and superposition between two sets of labeled points (K-mers).

For testing of the datasets, the `TestProSPECCTs.py` is provided.

The included Jupyter notebooks were used for the data analysis and the figure generation.

## How to Run

### Using the Database

To use the database, firstly import it and initialize it using:

```python3
from Database import Database

db = Database()
```

To add a structure, use:

```python3
db.add(text_id, sequence, structure)
```

where `text_id` is e.g., the PDB ID of the added structure, `sequence` is a string of one-letter protein codes 
and `structure` is the BioPython's structure from the module `Bio.PDB.Structure`.

For searching, the `DB` class contains two methods, function `kmer_cluster_svd_search` and function `score_list`.
Their interface is almost identical, but they differ in their goal:

1. `kmer_cluster_svd_search` searches the entire database and uses multiple stages of filtering; then it returns
all results, that were superposed along with their score (structures filtered prior to superposition are not reported).
2. `score_list` accepts additional parameter `ids_to_score`, a list of strings containing the Ids of structures to score
   (Ids are used as they were added in the `add` function).

The interface of score_list looks as follows:

```python3
score_list(self, template_positions: list[np.ndarray],
                   seq: str,
                   site: list[int],
                   k_mer_min_found_fraction: float,
                   k_mer_similarity_threshold: float,
                   cavity_scale_error: float,
                   min_kmer_in_cavity_fraction: float,
                   align_rounds: int,
                   align_sufficient_samples: int,
                   align_delta: float,
                   icp_rounds: int,
                   icp_cutoff: float,
                   ids_to_score: list[str]) -> defaultdict[dict]
```

where:

- template_positions: List of vectors of positions of residues in the query structure
- seq: The sequence of the query structure
- site: List of indices of the residues participating in the ligand binding (indices of pocket residues)
- k_mer_min_found_fraction: Minimal allowed fraction of all similar Kmers found in the target sequence,
        compared to the total count of searched Kmers originating from the query cavity (all Kmers around residues
        defined in the site list)
- k_mer_similarity_threshold: What is the minimal similarity between Kmers to be considered similar and
        to be potentially mapped on each other. Similarity is the sum of scores from the BLOSUM 62 table for each
        pair of aligned residues in directly aligned Kmers (with no shifts or gaps).
- cavity_scale_error: Scale factor for the possible change of size of the target cavity compared to the
        query cavity. Measured as the fraction of the largest distance of any two residues in the cavities.
- min_kmer_in_cavity_fraction: The minimal allowed fraction of similar Kmers in the target cavity to be
        considered as a potentially similar cavity. Measured as fraction of Kmers in the target to the number of Kmers
        from the query cavity.
- align_rounds: The number of rounds to perform using the RandomConsensusAligner
- align_sufficient_samples: After how many successful overlaps to terminate
- align_delta: the allowed fractional error of the distance between labeled points (centers of Kmers) in
        the target cavity and distance between points in the query cavity, normalized to the distance in the query
        cavity.
- icp_rounds: How many rounds of the Iterative Closest Point algorithm to perform
- icp_cutoff: Maximal distance of two similar Kmers to be still mapped in the Iterative Closest Point
        algorithm
- ids_to_score: list of ids to score against the query

And the function returns a dictionary of dicts of scores and found rotation, translation and mapping (when applicable) for
        each ID from the ids_to_score list.

The `kmer_cluster_svd_search` has identical interface, except for the missing `ids_to_score`, and returns a list of dicts of scores and various metrics and information for each found hit. The results might
        contain multiple hits from a single structure (multiple putative pockets).

The `score_list` function is particularly useful for evaluation on datasets with predefined similar and dissimilar pairs.

### Evaluation on Existing Datasets

Currently implemented is only the [ProSPECCTs dataset](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006483#sec016)
by Ehrt et al., to test a single subset of the dataset, ensure the path is set at the header of the `TestProSPECCTs.py` file:

```python3
DATASET_PATH = '../../prospeccts'
```

This folder shall contain the subset for testing, such as `NMR_structures` (a folder containing a subfolder of the same name, as in the supplied dataset by Ehrt et al.).
Then set the desired subset for testing at the very bottom of the file:

```python3
if __name__ == '__main__':

    dataset = 'NMR_structures'  # or any other dataset from the ProSPECCTs benchmark
    process_and_test_prospeccts(dataset)
```

The function will add all the structures to the database, save it as a pickle file for subsequent runs, and
perform searching of all structures listed in at least one pair (both active and inactive - meaning it has at least one
similar binding site in the database or at least one non-similar). ROC curve graph is generated using `matplotlib` and the
time is measured using the `timeit` module.

The results will be saved in the RESULTS_DIR (set at the header of the file) as a pickle file, which contains 
the results of the search and to each hit the distance of its and the query's ligands after the found superposition
along with the label from the dataset (translated to positive for active pairs and negative for inactive) are added.

These results can be subsequently analyzed in the supplied `ResultsAnalysis.ipynb`, a Jupyter notebook.

## Author and Affiliation

Jakub Telcer, Faculty of Science, Charles University in Prague

Contact: telcerj@natur.cuni.cz

## License

This software is distributed under the GNU GPL v3 License, see LICENSE.
