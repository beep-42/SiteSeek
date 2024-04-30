
class KMerFilter:

    @staticmethod
    def search_kmers(db, kmers_dict, k_mer_min_found_fraction: float, use_subset: list[int] = None):

        all_seqs = {x: 0 for x in range(len(db.seqs))} if use_subset is None else {x: 0 for x in use_subset}

        max_incorrect = len(kmers_dict) - int(k_mer_min_found_fraction * len(kmers_dict))

        for i, km in enumerate(kmers_dict):  # find all that have at least threshold% of k-mers

            found = []
            for one in kmers_dict[km]:
                if one in db.db:

                    for s in all_seqs:

                        if s in db.db[one] and s not in found:
                            all_seqs[s] += 1
                            found.append(s)

            if i >= max_incorrect:

                new = {}
                for one in all_seqs:
                    if i - all_seqs[one] <= max_incorrect:
                        new[one] = all_seqs[one]

                all_seqs = new

        all_seq_keys = list(all_seqs.keys())

        return all_seq_keys, all_seqs
