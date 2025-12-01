import warnings
from functools import partial
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import pyopal

from mDeepFRI.mmseqs import MMseqsResult
from mDeepFRI.utils import (load_fasta_as_dict, retrieve_fasta_entries_as_dict,
                            stdout_warn)

warnings.showwarning = stdout_warn


def insert_gaps(sequence: str, reference: str,
                alignment_string: str) -> Tuple[str, str]:
    """
    Inserts gaps into query and target sequences.

    Args:
        sequence (str): Query sequence.
        reference (str): Target sequence.
        alignment_string (str): Alignment string.

    Returns:
        gapped_sequence (str): Query sequence with gaps.
        gapped_target (str): Target sequence with gaps.
    """

    sequence = list(sequence)
    reference = list(reference)
    alignment_string = list(alignment_string)

    for i, a in enumerate(alignment_string):
        if a == "I":
            sequence.insert(i, "-")
        elif a == "D":
            reference.insert(i, "-")
    return "".join(sequence), "".join(reference)


class AlignmentResult:
    """
    Class for storing pairwise alignment results.

    Attributes:
        query_name (str): Name of the query sequence.
        query_sequence (str): Query sequence.
        target_name (str): Name of the target sequence.
        target_sequence (str): Target sequence.
        alignment (str): Alignment string.
        gapped_sequence (str): Query sequence with gaps.
        gapped_target (str): Target sequence with gaps.
        identity (float): Identity between two sequences.
        coords (np.ndarray): Coordinates of the C-alpha atoms in structure.
    """
    def __init__(self,
                 query_name: str = "",
                 query_sequence: str = "",
                 target_name: str = "",
                 target_sequence: str = "",
                 alignment: str = "",
                 query_identity: float = None,
                 query_coverage: float = None,
                 db_name: str = None,
                 coords: np.ndarray = None,
                 alignment_score: float = None):

        self.query_name = query_name
        self.query_sequence = query_sequence
        self.target_name = target_name
        self.target_sequence = target_sequence
        self.alignment = alignment
        self.query_identity = query_identity
        self.query_coverage = query_coverage
        self.alignment_score = alignment_score
        self.insert_gaps()
        self.db_name = db_name
        self.target_coords = None
        self.cmap = None
        self.aligned_cmap = None

    def __str__(self):
        return f"AlignmentResult(query_name={self.query_name}, target_name={self.target_name}, " \
               f"query_identity={self.query_identity}, query_coverage={self.query_coverage})"

    def __repr__(self):
        return f"AlignmentResult(query_name={self.query_name}, target_name={self.target_name}, " \
               f"query_identity={self.query_identity}, query_coverage={self.query_coverage})"

    def insert_gaps(self):
        """
        Inserts gaps into query and target sequences.

        Returns:
            AlignmentResult: The object with gapped sequences.
        """

        self.gapped_sequence, self.gapped_target = insert_gaps(
            self.query_sequence, self.target_sequence, self.alignment)


def best_hit_database(query,
                      target_sequences,
                      gap_open: int = 10,
                      gap_extend: int = 1):
    """
    Find the best hit in the database and return index.

    Args:
        query (str): The query sequence.
        target_sequences (dict): The target sequences.

    Returns:
        int: The index of the best hit.
        str: The best hit sequence.
    """

    aligner = pyopal.Aligner(gap_open=gap_open, gap_extend=gap_extend)
    target_database = pyopal.Database(target_sequences.values())

    # Retrieve the best hit
    results = aligner.align(query,
                            target_database,
                            mode="score",
                            overflow="buckets",
                            algorithm="nw")
    best_hit = max(results, key=lambda x: x.score)
    best_index = best_hit.target_index
    best_idx = list(target_sequences.keys())[best_index]
    best_seq = target_sequences[best_idx]

    return best_idx, best_seq


def align_pairwise(query, target, gap_open: int = 10, gap_extend: int = 1):
    """
    Aligns the query against the target and returns the alignment.

    Args:
        query (str): The query sequence.
        target (str): The target sequence.
        gap_open (int): Gap open penalty.
        gap_extend (int): Gap extend penalty.

    Returns:
        str: The alignment of the query against the target.
        float: The identity of the alignment.
        float: The coverage of the alignment.
        float: The alignment score.

    """

    aligner = pyopal.Aligner(gap_open=gap_open, gap_extend=gap_extend)
    database = pyopal.Database([target])
    # Align the sequences
    alignment = aligner.align(query, database, algorithm="nw", mode="full")
    alignment_result = alignment[0]
    alignment_string = alignment_result.alignment
    identity = alignment_result.identity()
    coverage = alignment_result.coverage(reference="query")
    score = alignment_result.score

    return alignment_string, identity, coverage, score


def pairwise_against_database(query_id,
                              query_sequence,
                              target_sequences,
                              gap_open: int = 10,
                              gap_extend: int = 1):
    """
    Finds the best alignment of the query against the target.
    """

    # find best hits in the database
    best_idx, best_target = best_hit_database(query_sequence, target_sequences,
                                              gap_open, gap_extend)
    # align the query against the best hit
    alignment, identity, coverage, score = align_pairwise(query_sequence, best_target,
                                                          gap_open, gap_extend)
    # create an alignment object
    alignment_result = AlignmentResult(query_id, query_sequence, best_idx,
                                       best_target, alignment, identity,
                                       coverage, alignment_score=score)

    return alignment_result


def create_partial_database(top_matches, target_sequences):
    """
    Create a partial database from the target sequences.

    Args:

    """

    partial_db = {k: target_sequences[k] for k in top_matches}

    return partial_db


def align_mmseqs_results(best_matches_filepath: str,
                         sequence_db: str,
                         alignment_gap_open: int = 10,
                         alignment_gap_extend: int = 1,
                         threads: int = 1,
                         output_path: str = None,
                         save_raw_alignments: bool = False,
                         raw_alignments_path: str = None):
    import logging
    import csv
    from pathlib import Path
    logger = logging.getLogger(__name__)

    best_matches = MMseqsResult.from_best_matches(best_matches_filepath)
    # check if there are any best matches
    if best_matches.size == 0:
        return []

    logger.info("Loading query sequences from %s...", best_matches.query_fasta)
    query_dict = load_fasta_as_dict(best_matches.query_fasta)
    logger.info("Loaded %d query sequences.", len(query_dict))
    
    logger.info("Extracting unique queries and targets...")
    unique_queries = {
        query: best_matches.get_query_targets(query)
        for query in best_matches.get_queries()
    }
    target_ids = best_matches.get_targets()
    logger.info("Found %d unique queries and %d target IDs.", len(unique_queries), len(target_ids))
    
    logger.info("Loading target sequences from %s (this may take a while for large databases)...", sequence_db)
    target_seqs = retrieve_fasta_entries_as_dict(sequence_db, target_ids)
    logger.info("Loaded %d target sequences.", len(target_seqs))

    # create partial databases
    logger.info("Creating partial databases for %d queries...", len(unique_queries))
    partial_databases = {
        k: create_partial_database(v, target_seqs)
        for k, v in unique_queries.items()
    }
    logger.info("Created partial databases.")

    align_partial = partial(pairwise_against_database,
                            gap_open=alignment_gap_open,
                            gap_extend=alignment_gap_extend)

    # align the query against the best hit
    logger.info("Starting pairwise alignment with %d threads for %d queries...", threads, len(unique_queries))
    with Pool(threads) as pool:
        alignments = pool.starmap(
            align_partial,
            [(query_id, query_dict[query_id], partial_databases[query_id])
             for query_id in unique_queries])
    logger.info("Completed pairwise alignment, generated %d alignments.", len(alignments))

    # Save alignment scores to file if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving alignment scores to %s", output_path)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['query_name', 'target_name', 'alignment_score', 'identity', 'coverage', 'query_length', 'target_length'])
            for aln in alignments:
                writer.writerow([
                    aln.query_name,
                    aln.target_name,
                    aln.alignment_score if aln.alignment_score is not None else '',
                    aln.query_identity if aln.query_identity is not None else '',
                    aln.query_coverage if aln.query_coverage is not None else '',
                    len(aln.query_sequence) if aln.query_sequence else '',
                    len(aln.target_sequence) if aln.target_sequence else ''
                ])
        logger.info("Saved alignment scores for %d alignments", len(alignments))

    # Save raw alignments (gapped sequences) to FASTA file if requested
    if save_raw_alignments and raw_alignments_path is not None:
        raw_alignments_path = Path(raw_alignments_path)
        raw_alignments_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving raw alignments to %s", raw_alignments_path)
        
        with open(raw_alignments_path, 'w') as f:
            for aln in alignments:
                # Write query sequence with gaps
                header_query = f">{aln.query_name}|target={aln.target_name}|identity={aln.query_identity:.4f}|coverage={aln.query_coverage:.4f}|score={aln.alignment_score if aln.alignment_score is not None else 'N/A'}"
                f.write(header_query + "\n")
                # Write sequence in chunks of 80 characters (standard FASTA format)
                gapped_query = aln.gapped_sequence if hasattr(aln, 'gapped_sequence') and aln.gapped_sequence else aln.query_sequence
                for i in range(0, len(gapped_query), 80):
                    f.write(gapped_query[i:i+80] + "\n")
                
                # Write target sequence with gaps
                header_target = f">{aln.target_name}|query={aln.query_name}|identity={aln.query_identity:.4f}|coverage={aln.query_coverage:.4f}|score={aln.alignment_score if aln.alignment_score is not None else 'N/A'}"
                f.write(header_target + "\n")
                gapped_target = aln.gapped_target if hasattr(aln, 'gapped_target') and aln.gapped_target else aln.target_sequence
                for i in range(0, len(gapped_target), 80):
                    f.write(gapped_target[i:i+80] + "\n")
                
                # Write alignment string (optional, for reference)
                f.write(f"#alignment_string: {aln.alignment}\n")
        
        logger.info("Saved raw alignments for %d alignments", len(alignments))

    return alignments
