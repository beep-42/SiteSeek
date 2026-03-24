from Database import Database
db = Database.load('../../pdb-database-mirror', compressed=False)
db.estimate_background_distribution('background_estimation_set.json', {'k_mer_similarity_threshold':14, 'lr': 0.90, 'skip_clustering': False, 'progress': False}, sub_sample=0.1, n_permutations=3, length=5)