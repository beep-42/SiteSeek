import line_profiler

from FullSiteMapping import FullSiteMapper
from KmerGen import compute_mapping_similarity_score


@line_profiler.profile
def score_hit(found_kmer_fraction: float, coverage, mapped_ratio, hit_positions, query_positions, hit_seq,
              query_seq, query_site, mapping, rot, trans):


    (full_site_mapping, full_site_rmsd, full_site_rot, full_site_trans,
     nearby_fraction, mapping_loss, mapping_persistence) = FullSiteMapper.get_full_closest_mappings_notree(
                hit_positions, query_positions, hit_seq, query_seq, query_site, mapping, rot, trans
    )

    # full_site_score = compute_mapping_similarity_score(full_site_mapping, hit_seq, query_seq)

    # score = coverage - 2*full_site_rmsd + nearby_fraction + mapped_ratio + 1/2*found_kmer_fraction + mapping_persistence - mapping_loss
    #
    # if mapping_loss == 1:
    #    score /= 10

    # use the pseudo counted inverse of full_site_rmsd
    return 1 / (full_site_rmsd+0.01) # * (100 ** found_kmer_fraction - 1)
