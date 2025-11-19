import csv
import logging
import pathlib
import pickle
import sys
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from mDeepFRI import BAR_FORMAT, DEEPFRI_MODES, OUTPUT_HEADER
from mDeepFRI.alignment import align_mmseqs_results
from mDeepFRI.bio_utils import build_align_contact_map
from mDeepFRI.database import Database, build_database
from mDeepFRI.mmseqs import MMseqsResult, QueryFile
from mDeepFRI.pdb import create_pdb_mmseqs, extract_calpha_coords
from mDeepFRI.predict import Predictor
from mDeepFRI.utils import load_deepfri_config, remove_intermediate_files
from mDeepFRI.mmseqs import filter_mmseqs_best_matches

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# load sequences in query filtered by length
def load_query_file(query_file: str, min_length: int = None, max_length: int = None):
    query_file = QueryFile(filepath=query_file)
    query_file.load_sequences()
    # filter out sequences
    if min_length or max_length:
        query_file.filter_sequences(
            lambda x: min_length <= len(x) <= max_length)

    return query_file


def hierarchical_database_search(query_file: QueryFile,
                                 output_path: str,
                                 databases: Iterable[str] = [],
                                 sensitivity: float = 5.7,
                                 min_bits: float = 0,
                                 max_eval: float = 1e-5,
                                 min_ident: float = 0.5,
                                 min_coverage: float = 0.9,
                                 top_k: int = 5,
                                 skip_pdb: bool = False,
                                 overwrite: bool = False,
                                 tmpdir: str = None,
                                 threads: int = 1):

    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # logging variable
    sequence_num_start = len(query_file.sequences)

    for idx, seq in query_file.filtered_out.items():
        logger.info(f"Skipping {idx}; sequence length {len(seq)} aa.")

    dbs = []
    # PDB100 database
    if not skip_pdb:
        logger.info("Creating PDB100 database.")
        pdb100 = create_pdb_mmseqs(threads=threads)
        dbs.append(pdb100)
        logger.info("PDB100 database created.")

    for database in databases:
        database = pathlib.Path(database)
        db = build_database(
            input_path=database,
            output_path=database.parent,
            overwrite=overwrite,
            threads=threads,
        )
        dbs.append(db)

    aligned_total = 0

    for db in dbs:
        results = query_file.search(db.mmseqs_db,
                                    sensitivity=sensitivity,
                                    eval=max_eval,
                                    threads=threads,
                                    tmpdir=tmpdir)

        filtered = results.apply_filters(min_cov=min_coverage,
                                         min_bits=min_bits,
                                         min_ident=min_ident)

        try:
            best_matches = filtered.find_best_matches(top_k, threads=threads)
        except ValueError:
            best_matches = MMseqsResult([], results.query_fasta,
                                        db.sequence_db)

        mmseqs_results_path = output_path / f"{db.name}_results.tsv"
        # save intermediate results
        best_matches.save(mmseqs_results_path)
        # store the location of the result for the next step
        db.mmseqs_result = mmseqs_results_path

        # catch error if no matches to database
        # a case from phage proteins
        try:
            unique_hits = np.unique(best_matches["query"])
        except IndexError:
            unique_hits = np.array([])

        if "pdb100" in db.name:
            pdb_hits = unique_hits
        elif skip_pdb:
            unique_hits = [hit for hit in unique_hits]
        else:
            unique_hits = [hit for hit in unique_hits if hit not in pdb_hits]

        aligned_db = len(unique_hits)
        aligned_total += aligned_db
        aligned_perc = round(aligned_db / sequence_num_start * 100, 2)
        total_perc = round(aligned_total / sequence_num_start * 100, 2)
        logger.info(f"Aligned {aligned_db}/{sequence_num_start} "
                    f"({aligned_perc:.2f}%) proteins against {db.name}.")
        logger.info(
            f"Aligned {aligned_total}/{sequence_num_start} ({total_perc:.2f}%) proteins in total."
        )

        # this mechanism decreases the amount of sequences
        # on each iteration. Drastically improves execution times
        # for large datasets.
        # PDB100 hits are aligned second time to experimental
        # structures in order to save failed contact map alignemnts.
        #if 'pdb100' not in db.name:
            #query_file.remove_sequences(unique_hits)

    return dbs


def align_pairwise():
    pass


def prepare_alignments_for_identity_bin(
    query_file: QueryFile,
    databases: Tuple[Database],
    identity_bin: tuple[float, float] | None = None,
    identity_max: float | None = None,
    alignment_gap_open: float = 10,
    alignment_gap_continuation: float = 1,
    identity_threshold: float = 0.5,
    coverage_threshold: float = 0.9,
    selection: str = "topbits",
    seed: int = 0,
    drop_self_hits: bool = True,
    threads: int = 1,
    save_structures: bool = False,
    output_path: str = None,
):
    """
    Prepare alignments and coordinates for a given identity bin.
    This is expensive and only needs to be done once per identity bin.
    
    Returns:
        dict: Dictionary mapping database names to lists of AlignmentResult objects with coordinates
    """
    output_path = Path(output_path) if output_path else None
    all_alignments = {}
    
    # Determine identity bin name for organization (used in multiple places)
    if output_path and identity_bin:
        bin_name = f"identity_bin_{identity_bin[0]:.2f}-{identity_bin[1]:.2f}"
    elif output_path:
        bin_name = "identity_bin_all"
    else:
        bin_name = None
    
    for db in databases:
        if identity_bin or identity_max is not None:
            # Create organized directory for filtered MMSeqs2 results
            if output_path:
                filters_dir = output_path / "mmseqs2_filtered" / bin_name
                filters_dir.mkdir(parents=True, exist_ok=True)
                # Use meaningful filename
                filtered_filename = f"{db.name}_filtered.tsv"
                filtered_tsv_path = filters_dir / filtered_filename
            else:
                filters_dir = None
                filtered_tsv_path = None

            low, high = (identity_bin if identity_bin else (None, None))
            filtered_tsv = filter_mmseqs_best_matches(
                best_matches_path=db.mmseqs_result,
                ident_low=low, ident_high=high,
                ident_max=identity_max,
                min_qcov=coverage_threshold,
                min_tcov=coverage_threshold,
                drop_self_hits=drop_self_hits,
                per_query=selection,
                seed=seed,
                out_path=str(filtered_tsv_path) if filtered_tsv_path else None
            )
            db.mmseqs_result = filtered_tsv
        
        # SEQUENCE ALIGNMENT
        logger.info("Reading filtered best-matches from: %s", db.mmseqs_result)
        bm = MMseqsResult.from_best_matches(db.mmseqs_result)
        if bm.result_arr.ndim == 0:
            num_rows = 0
        else:
            num_rows = len(bm.result_arr)
        logger.info("Filtered best-matches rows for %s: %d", db.name, num_rows)
        
        if num_rows == 0:
            logger.warning("No matches found for %s after filtering (file: %s), skipping alignment.", 
                          db.name, db.mmseqs_result)
            # Check if file exists and has content
            if Path(db.mmseqs_result).exists():
                file_size = Path(db.mmseqs_result).stat().st_size
                logger.warning("File exists with size %d bytes. This may indicate a parsing issue.", file_size)
                # Try to read first few lines
                try:
                    with open(db.mmseqs_result, 'r') as f:
                        first_lines = [f.readline().strip() for _ in range(3)]
                    logger.warning("First lines of file: %s", first_lines)
                except Exception as e:
                    logger.warning("Could not read file: %s", e)
            all_alignments[db.name] = []
            continue
        
        logger.info("Starting sequence alignment for %s (%d matches)...", db.name, num_rows)
        # Save alignment scores to organized directory
        alignment_scores_path = None
        if output_path:
            alignment_scores_dir = output_path / "pyopal_alignments" / bin_name
            alignment_scores_dir.mkdir(parents=True, exist_ok=True)
            alignment_scores_path = alignment_scores_dir / f"{db.name}_alignment_scores.tsv"
        
        alignments = align_mmseqs_results(
            best_matches_filepath=db.mmseqs_result,
            sequence_db=db.sequence_db,
            alignment_gap_open=alignment_gap_open,
            alignment_gap_extend=alignment_gap_continuation,
            threads=threads,
            output_path=str(alignment_scores_path) if alignment_scores_path else None)
        logger.info("Completed sequence alignment for %s (%d alignments).", db.name, len(alignments))

        # filter alignments by identity and coverage
        # Note: thresholds are inclusive (>=), so identity == threshold passes
        alignments_before_filter = len(alignments)
        filtered_out_identity = 0
        filtered_out_coverage = 0
        filtered_alignments = []
        for aln in alignments:
            if aln.query_identity < identity_threshold:
                filtered_out_identity += 1
                logger.debug("Filtered out %s vs %s: identity %.3f < threshold %.3f", 
                            aln.query_name, aln.target_name, aln.query_identity, identity_threshold)
            elif aln.query_coverage < coverage_threshold:
                filtered_out_coverage += 1
                logger.debug("Filtered out %s vs %s: coverage %.3f < threshold %.3f", 
                            aln.query_name, aln.target_name, aln.query_coverage, coverage_threshold)
            else:
                filtered_alignments.append(aln)
        
        alignments = filtered_alignments
        if filtered_out_identity > 0 or filtered_out_coverage > 0:
            logger.info("Filtered alignments for %s: %d passed, %d failed identity (<%.3f), %d failed coverage (<%.3f)", 
                       db.name, len(alignments), filtered_out_identity, identity_threshold, 
                       filtered_out_coverage, coverage_threshold)

        try:
            for aln in alignments:
                aln.db_name = db.name

            new_alignments = {
                aln.query_name: aln
                for aln in alignments
                if aln.query_name in query_file.sequences
            }

            # remove broken structures
            if db.name == "highquality_clust30":
                data_path = pathlib.Path(__file__).parent / "assets"
                data_path = data_path.resolve()
                with open(data_path / "highquality_clust30_error_ids.pkl", "rb") as f:
                    error_ids = pickle.load(f)
                new_alignments = {
                    query_name: aln
                    for query_name, aln in new_alignments.items()
                    if aln.target_name not in error_ids
                }

            query_ids = [aln.query_name for aln in new_alignments.values()]
            target_ids = [
                aln.target_name.rsplit(".", 1)[0]
                for aln in new_alignments.values()
            ]

            # extract structural information
            if save_structures and output_path:
                structures_dir = output_path / "structures" / bin_name / db.name
                structures_dir.mkdir(parents=True, exist_ok=True)
                save_dir = structures_dir
            else:
                save_dir = None

            coords = extract_calpha_coords(db,
                                           target_ids,
                                           query_ids,
                                           save_directory=save_dir,
                                           threads=threads)

            # Log when coordinates extraction fails
            failed_coord_extractions = []
            for aln, coord in zip(new_alignments.values(), coords):
                aln.coords = coord
                if coord is None:
                    failed_coord_extractions.append(
                        (aln.query_name, aln.target_name, aln.query_identity, aln.query_coverage))
            
            if failed_coord_extractions:
                logger.warning(
                    f"Failed to extract coordinates for {len(failed_coord_extractions)}/{len(new_alignments)} "
                    f"alignments in database {db.name}:")
                for qname, tname, ident, cov in failed_coord_extractions[:10]:
                    logger.warning(
                        f"  - Query: {qname}, Target: {tname}, Identity: {ident:.3f}, Coverage: {cov:.3f}")
                if len(failed_coord_extractions) > 10:
                    logger.warning(f"  ... and {len(failed_coord_extractions) - 10} more")

            all_alignments[db.name] = list(new_alignments.values())

        except IndexError:
            logger.info("No alignments found for %s.", db.name)
            all_alignments[db.name] = []
    
    return all_alignments


def predict_protein_function(
    query_file: QueryFile,
    databases: Tuple[Database],
    weights: str,
    output_path: str,
    deepfri_processing_modes: List[str] = ["ec", "bp", "mf", "cc"],
    angstrom_contact_threshold: float = 6,
    generate_contacts: int = 2,
    alignment_gap_open: float = 10,
    alignment_gap_continuation: float = 1,
    identity_threshold: float = 0.5,
    coverage_threshold: float = 0.9,
    remove_intermediate=False,
    threads: int = 1,
    save_structures: bool = False,
    save_cmaps: bool = False,
    identity_bin: tuple[float, float] | None = None,   # e.g. (0.80, 0.90)
    identity_max: float | None = None,                 # e.g. 0.80  (cumulative)
    selection: str = "topbits",                        # or "topbits"
    seed: int = 0,
    drop_self_hits: bool = True,
    precomputed_alignments: dict = None,  # Optional: reuse alignments from prepare_alignments_for_identity_bin
):

    # load DeepFRI model
    deepfri_models_config = load_deepfri_config(weights)
    # version 1.1 drops support for ec
    if deepfri_models_config["version"] == "1.1":
        # remove "ec" from processing modes
        deepfri_processing_modes = [
            mode for mode in deepfri_processing_modes if mode != "ec"
        ]
        logger.info("EC number prediction is not supported in version 1.1.")

    # Allow empty processing modes to skip GO-term prediction
    skip_prediction = len(deepfri_processing_modes) == 0
    if skip_prediction:
        logger.info("No processing modes specified - skipping GO-term prediction. "
                   "Will still perform alignments and contact map generation.")

    weights = pathlib.Path(weights)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    aligned_cmaps = []
    
    # Use precomputed alignments if provided, otherwise compute them
    if precomputed_alignments is not None:
        # Reuse precomputed alignments (much faster!)
        logger.info("Using precomputed alignments (skipping alignment step)")
        all_alignments = precomputed_alignments
    else:
        # Compute alignments (original behavior for backward compatibility)
        all_alignments = prepare_alignments_for_identity_bin(
            query_file=query_file,
            databases=databases,
            identity_bin=identity_bin,
            identity_max=identity_max,
            alignment_gap_open=alignment_gap_open,
            alignment_gap_continuation=alignment_gap_continuation,
            identity_threshold=identity_threshold,
            coverage_threshold=coverage_threshold,
            selection=selection,
            seed=seed,
            drop_self_hits=drop_self_hits,
            threads=threads,
            save_structures=save_structures,
            output_path=output_path,
        )
    
    # Build contact maps from alignments
    for db in databases:
        new_alignments_list = all_alignments.get(db.name, [])
        if not new_alignments_list:
            continue
        
        # Convert list to dict for consistency with old code
        new_alignments = {aln.query_name: aln for aln in new_alignments_list}
        
        # Remove queries already aligned in previous databases
        aligned_queries = [aln[0].query_name for aln in aligned_cmaps]
        new_alignments = {
            qname: aln for qname, aln in new_alignments.items()
            if qname not in aligned_queries
        }
        
        if not new_alignments:
            continue

        # Build contact maps (this is the only part that depends on generate_contacts)
        partial_map_align = partial(build_align_contact_map,
                                    threshold=angstrom_contact_threshold,
                                    generated_contacts=generate_contacts)

        with Pool(threads) as p:
            cmaps = list(p.map(partial_map_align, new_alignments.values()))

        # filter errored contact maps
        # returned as Tuple[AlignmentResult, None] from `retrieve_align_contact_map`
        partial_cmaps = [cmap for cmap in cmaps if cmap[1] is not None]
        failed_cmaps = [cmap for cmap in cmaps if cmap[1] is None]
        
        # Log sequences that aligned successfully but failed to produce contact maps
        if failed_cmaps:
            logger.warning(
                f"Contact map generation failed for {len(failed_cmaps)}/{len(cmaps)} successfully aligned sequences "
                f"in database {db.name} (see warnings above for details).")
            # Summary statistics
            failed_with_coords = sum(1 for aln, cmap in failed_cmaps if aln.coords is not None)
            failed_without_coords = len(failed_cmaps) - failed_with_coords
            if failed_with_coords > 0:
                logger.warning(
                    f"  - {failed_with_coords} failed during contact map alignment (coordinates were available)")
            if failed_without_coords > 0:
                logger.warning(
                    f"  - {failed_without_coords} failed due to missing coordinates")
        
        aligned_cmaps.extend(partial_cmaps)
        aligned_database = round(
            len(partial_cmaps) / len(query_file.sequences) * 100, 2)
        aligned_total = round(
            len(aligned_cmaps) / len(query_file.sequences) * 100, 2)
        logger.info(
            f"Aligned {len(partial_cmaps)}/{len(query_file.sequences)} ({aligned_database}%) "
            f"proteins against {db.name} [without length ivalid].")
        logger.info(
            f"Aligned {len(aligned_cmaps)}/{len(query_file.sequences)} ({aligned_total}%) "
            "proteins in total [without length invalid].")

    if save_cmaps:
        cmap_dir = output_path / "contact_maps"
        cmap_dir.mkdir(parents=True, exist_ok=True)
        for i, (aln, cmap) in enumerate(aligned_cmaps):
            cmap_file = cmap_dir / f"{aln.query_name}.npy"
            np.save(cmap_file, cmap)

    # FUNCTION PREDICTION
    aligned_queries = [aln[0].query_name for aln in aligned_cmaps]
    unaligned_queries = {
        query_id: seq
        for query_id, seq in query_file.sequences.items()
        if query_id not in aligned_queries
    }

    # sort cmaps by length of query sequence
    aligned_cmaps = sorted(aligned_cmaps,
                           key=lambda x: len(x[0].query_sequence))
    # sort unaligned queries by length
    unaligned_queries = dict(
        sorted(unaligned_queries.items(), key=lambda x: len(x[1])))

    # FUNCTION PREDICTION (skip if processing_modes is empty)
    if not skip_prediction:
        output_file_name = output_path / "results.tsv"
        output_buffer = open(output_file_name, "w", encoding="utf-8")
        csv_writer = csv.writer(output_buffer, delimiter="\t")
        csv_writer.writerow(OUTPUT_HEADER)

        for i, mode in enumerate(deepfri_processing_modes):
            logger.info("Processing mode: %s; %i/%i", DEEPFRI_MODES[mode], i + 1,
                        len(deepfri_processing_modes))
            # GCN
            gcn_prots = len(aligned_cmaps)
            if gcn_prots > 0:
                net_type = "gcn"

                # GCN for queries with aligned contact map
                gcn_path = deepfri_models_config[net_type][mode]

                gcn = Predictor(gcn_path, threads=threads)

                for i, (aln, aligned_cmap) in tqdm(
                        enumerate(aligned_cmaps),
                        total=gcn_prots,
                        miniters=len(aligned_cmaps) // 10,
                        desc=f"Predicting with GCN ({DEEPFRI_MODES[mode]})",
                        bar_format=BAR_FORMAT):
                    # writing the results to the output file

                    prediction_rows = gcn.predict_function(
                        seqres=aln.query_sequence,
                        cmap=aligned_cmap,
                        chain=str(aln.query_name))

                    for row in prediction_rows:
                        deepfri_info = [net_type, mode]
                        row.extend(deepfri_info)

                        # additional alignment info
                        # corrected name for FoldComp inconsistency

                        row.extend([
                            aln.target_name.rsplit(".", 1)[0], aln.db_name,
                            aln.query_identity, aln.query_coverage
                        ])
                        csv_writer.writerow(row)

                del gcn

            # CNN for queries without satisfying alignments
            cnn_prots = len(unaligned_queries)
            if cnn_prots > 0:
                net_type = "cnn"
                cnn_path = deepfri_models_config[net_type][mode]
                cnn = Predictor(cnn_path, threads=threads)
                for i, query_id in tqdm(
                        enumerate(unaligned_queries),
                        total=cnn_prots,
                        miniters=len(unaligned_queries) // 10,
                        desc=f"Predicting with CNN ({DEEPFRI_MODES[mode]})",
                        bar_format=BAR_FORMAT):

                    prediction_rows = cnn.predict_function(
                        seqres=unaligned_queries[query_id], chain=str(query_id))
                    for row in prediction_rows:
                        row.extend([net_type, mode])
                        row.extend([np.nan, np.nan, np.nan])
                        csv_writer.writerow(row)

                del cnn

        output_buffer.close()
    else:
        logger.info("Skipped GO-term prediction (no processing modes specified). "
                   f"Generated {len(aligned_cmaps)} contact maps and saved them to {output_path / 'contact_maps'}.")

    if remove_intermediate:
        for db in databases:
            remove_intermediate_files([db.sequence_db, db.mmseqs_db])

    logger.info("meta-DeepFRI finished successfully.")
