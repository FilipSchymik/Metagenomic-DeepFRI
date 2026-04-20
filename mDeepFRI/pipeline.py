"""
Pipeline module for protein function prediction using DeepFRI.

This module orchestrates the complete Metagenomic-DeepFRI pipeline, including:
1. Hierarchical database searches using MMseqs2
2. Alignment of query sequences to database hits using PyOpal
3. Contact map alignment for structure-based predictions
4. DeepFRI-based functional annotation

The pipeline can process proteins with or without structural information,
using Graph Convolutional Networks (GCN) when structures are available,
and Convolutional Neural Networks (CNN) when only sequences are available.

Attributes:
    ALIGNMENT_HEADER (list): Column names for alignment results TSV file.
    FINAL_OUTPUT_HEADER (list): Column names for final prediction results.
    NAN_ALIGNMENT_INFO (list): Default values for missing alignment information.
"""

import csv
import logging
from collections import defaultdict
import pathlib
import pickle
import sys
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from mDeepFRI import DEEPFRI_MODES
from mDeepFRI.alignment import AlignmentResult, align_mmseqs_results, align_pairwise
from mDeepFRI.bio_utils import build_align_contact_map, extract_residues_coordinates
from mDeepFRI.database import Database, build_database
from mDeepFRI.mmseqs import MMseqsResult, QueryFile
from mDeepFRI.pdb import create_pdb_mmseqs, extract_calpha_coords
from mDeepFRI.predict import Predictor
from mDeepFRI.structure_export import write_carved_structure_pdb
from mDeepFRI.utils import (get_json_values, load_deepfri_config,
                            remove_intermediate_files)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.propagate = False
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ALIGNMENT_HEADER = [
    "query_id", "aligned", "target_id", "db_name", "query_identity",
    "query_coverage", "target_coverage"
]
FINAL_OUTPUT_HEADER = [
    "protein", "network_type", "prediction_mode", "go_term", "score",
    "go_name", "aligned", "target_id", "db_name", "query_identity",
    "query_coverage", "target_coverage"
]

NAN_ALIGNMENT_INFO = [np.nan] * 6


def _write_carved_structure_worker(
        args: Tuple[AlignmentResult, pathlib.Path]) -> None:
    """Picklable worker for parallel carved PDB export (``save_aligned_structures``)."""
    aln, out_path = args
    write_carved_structure_pdb(aln, out_path)


def _raw_alignment_fasta_record(aln: AlignmentResult) -> str:
    """One pairwise block: two FASTA records, gapped sequences, alignment comment."""
    qid = aln.query_name
    tid = aln.target_name
    ident = float(aln.query_identity) if aln.query_identity is not None else float(
        "nan")
    cov = float(aln.query_coverage) if aln.query_coverage is not None else float(
        "nan")
    score = aln.alignment_score
    score_str = f"{score:g}" if score is not None else "NA"
    meta = f"|identity={ident:.4f}|coverage={cov:.4f}|score={score_str}"
    return "\n".join([
        f">{qid}|target={tid}{meta}",
        aln.gapped_sequence,
        f">{tid}|query={qid}{meta}",
        aln.gapped_target,
        f"#alignment_string: {aln.alignment}",
    ])


def load_query_file(query_file: str,
                    min_length: Optional[int] = None,
                    max_length: Optional[int] = None) -> QueryFile:
    """
    Load and filter protein sequences from a FASTA file.

    This function loads protein sequences from a FASTA file and optionally filters
    them based on sequence length constraints.

    Args:
        query_file (str): Path to input FASTA file containing protein sequences.
        min_length (int, optional): Minimum protein length in amino acids.
            Sequences shorter than this will be filtered out. Defaults to None.
        max_length (int, optional): Maximum protein length in amino acids.
            Sequences longer than this will be filtered out. Defaults to None.

    Returns:
        QueryFile: QueryFile object containing loaded and filtered sequences.

    Example:
        >>> qf = load_query_file("proteins.fasta", min_length=30, max_length=5000)
        >>> len(qf.sequences)
        42

    Raises:
        FileNotFoundError: If the query_file does not exist.
    """
    query_file = QueryFile(filepath=query_file)
    query_file.load_sequences()
    removed_seleno = query_file.remove_selenocysteine()
    if removed_seleno:
        logger.info("Removed %d selenoproteins (U residues): %s",
                    len(removed_seleno), ", ".join(removed_seleno))
    # filter out sequences
    if min_length or max_length:
        query_file.filter_sequences(
            lambda x: min_length <= len(x) <= max_length)

    return query_file


def hierarchical_database_search(query_file: QueryFile,
                                 output_path: str,
                                 databases: Iterable[str] = [],
                                 mmseqs_sensitivity: float = 5.7,
                                 min_bits: float = 0,
                                 max_eval: float = 1e-5,
                                 min_ident: float = 0.5,
                                 min_coverage: float = 0.9,
                                 top_k: int = 5,
                                 skip_pdb: bool = False,
                                 overwrite: bool = False,
                                 tmpdir: Optional[str] = None,
                                 threads: int = 1) -> List[Database]:
    """
    Perform hierarchical database searches for protein homologs.

    Searches query sequences against multiple databases in a hierarchical manner,
    starting with PDB100 (unless skipped), followed by user-specified databases.
    Results are filtered and the best matches are retained for structure-based
    annotation.

    Args:
        query_file (QueryFile): Object containing query sequences to search.
        output_path (str): Path to directory for saving search results.
        databases (Iterable[str], optional): List of paths to FoldComp databases
            to search (in order). Common databases include afdb_swissprot,
            esmatlas, etc. Defaults to empty list (only PDB if not skipped).
        mmseqs_sensitivity (float, optional): Sensitivity for MMseqs2 search.
            Range: 1.0-7.5, higher values are more sensitive but slower.
            Defaults to 5.7.
        min_bits (float, optional): Minimum bitscore threshold for hits.
            Defaults to 0.
        max_eval (float, optional): Maximum E-value threshold for hits.
            Defaults to 1e-5.
        min_ident (float, optional): Minimum sequence identity for alignment.
            Range: 0.0-1.0. Defaults to 0.5 (50%).
        min_coverage (float, optional): Minimum query/target coverage.
            Range: 0.0-1.0. Defaults to 0.9 (90%).
        top_k (int, optional): Maximum number of top hits to retain per sequence.
            Defaults to 5.
        skip_pdb (bool, optional): Skip searching against PDB100 database.
            Defaults to False.
        overwrite (bool, optional): Overwrite existing database files.
            Defaults to False.
        tmpdir (str, optional): Temporary directory for intermediate files.
            If None, system default temp directory is used. Defaults to None.
        threads (int, optional): Number of threads for parallel processing.
            Defaults to 1.

    Returns:
        Tuple[Dict, Set]: Tuple containing:
            - Dictionary mapping query IDs to list of alignment information
            - Set of PDB hits (for tracking unique structures)

    Raises:
        FileNotFoundError: If database paths do not exist.
        ValueError: If parameters are out of valid ranges.

    Note:
        The function creates intermediate files including MMseqs2 databases
        and search results. These can be removed after prediction with
        the remove_intermediate_files() function if storage is a concern.

    Example:
        >>> qf = load_query_file("proteins.fasta")
        >>> alignments, pdb_hits = hierarchical_database_search(
        ...     qf,
        ...     output_path="./results",
        ...     databases=["path/to/afdb_swissprot"],
        ...     threads=4
        ... )
    """

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
    pdb_hits = set()

    for db in dbs:
        results = query_file.search(db.mmseqs_db,
                                    mmseqs_sensitivity=mmseqs_sensitivity,
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
            all_hits = np.unique(best_matches["query"])
        except IndexError:
            all_hits = np.array([])
        # cover skip_pdb case
        unique_hits = all_hits

        if "pdb100" in db.name:
            pdb_hits.update(all_hits)
        elif not skip_pdb:
            unique_hits = [hit for hit in all_hits if hit not in pdb_hits]

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
        if 'pdb100' not in db.name:
            query_file.remove_sequences(all_hits)

    return dbs


STRUCTURE_EXTENSIONS = (".pdb", ".cif")


def _resolve_structure_path(
        structure_ref: str,
        mapping_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Resolve a structure reference from the mapping CSV to an actual file path.

    Accepts:
        - Absolute paths (e.g. /home/user/structures/7qpl_A.pdb)
        - Filenames with extension (e.g. 7qpl_A.pdb, 7qpl_A.cif)
        - Bare identifiers (e.g. 7qpl_A) — resolved by trying .pdb then .cif

    Args:
        structure_ref (str): Structure reference from the mapping CSV.
        mapping_dir (pathlib.Path): Directory containing the mapping CSV file,
            used as the base for resolving relative references.

    Returns:
        pathlib.Path or None: Resolved path to the structure file, or None if
            the file could not be found.
    """
    ref_path = pathlib.Path(structure_ref)

    # Absolute path
    if ref_path.is_absolute():
        if ref_path.exists():
            return ref_path
        return None

    # Filename with recognized extension (.pdb or .cif)
    if ref_path.suffix.lower() in STRUCTURE_EXTENSIONS:
        full_path = mapping_dir / ref_path
        if full_path.exists():
            return full_path
        if ref_path.exists():
            return ref_path.resolve()
        return None

    # Bare identifier — try adding extensions
    for ext in STRUCTURE_EXTENSIONS:
        full_path = mapping_dir / (structure_ref + ext)
        if full_path.exists():
            return full_path

    return None


def _stem_has_pdb_entry_id_prefix(stem: str) -> bool:
    """
    True if the stem looks like ``<PDB_ID>_<suffix>`` (wwPDB four-character id).

    Only then do we infer the chain from the filename; metagenome / AF / ESM
    identifiers with underscores (e.g. ``gene_00769_model``) default to chain A.
    """
    if "_" not in stem:
        return False
    head = stem.split("_", 1)[0]
    if len(head) != 4 or not head[0].isdigit():
        return False
    return all(ch.isalnum() for ch in head[1:])


def _extract_chain(structure_path: pathlib.Path) -> str:
    """
    Extract chain identifier from the structure filename when it is PDB-like.

    For stems whose first ``_``-separated segment is a four-character PDB entry
    id (digit + three alphanumerics), uses the **last character** after the
    final underscore (e.g. ``4v42_BI.pdb`` → ``I``, ``6sxu_BBB.pdb`` → ``B``,
    ``7qpl_A.pdb`` → ``A``). Multi-character suffixes use only the final
    character as the chain id (``id_AB`` → ``B``).

    If the segment after the last underscore is two or more digits only (e.g.
    batch or ordinal ids), returns ``"A"`` — those are not chain ids.

    For all other filenames (no underscore, or non-PDB-style prefix such as
    ``MGYG..._01599`` or ``..._model``), returns ``"A"``.

    Args:
        structure_path (pathlib.Path): Path to the structure file.

    Returns:
        str: Chain identifier (e.g. ``"A"``).
    """
    stem = structure_path.stem
    if not _stem_has_pdb_entry_id_prefix(stem):
        return "A"
    segment = stem.rsplit("_", 1)[1]
    if not segment:
        return "A"
    if len(segment) >= 2 and segment.isdigit():
        return "A"
    return segment[-1]


def _warn_length_mismatch(query_id: str, query_seq: str, struct_seq: str,
                          structure_ref: str) -> None:
    """
    Emit a warning if the query and structure sequences differ by >10% in length.

    Args:
        query_id (str): Identifier of the query sequence.
        query_seq (str): Query amino acid sequence.
        struct_seq (str): Structure-derived amino acid sequence.
        structure_ref (str): Structure reference identifier.
    """
    len_q = len(query_seq)
    len_s = len(struct_seq)
    shorter = min(len_q, len_s)
    longer = max(len_q, len_s)
    if shorter > 0 and longer / shorter > 1.1:
        logger.warning(
            "Length mismatch for %s <-> %s: "
            "query=%d aa, structure=%d aa (%.1fx difference).", query_id,
            structure_ref, len_q, len_s, longer / shorter)


def load_mapped_structures(
    query_file: QueryFile,
    mapping_csv: str,
    angstrom_contact_threshold: float = 6,
    generate_contacts: int = 2,
    alignment_gap_open: float = 10,
    alignment_gap_continuation: float = 1,
    scoring_matrix: str = "VTML80",
    threads: int = 1
) -> Tuple[List[Tuple[AlignmentResult, np.ndarray]], List[str]]:
    """
    Load user-supplied structure-to-query mappings and generate aligned contact maps.

    Reads a two-column CSV file (``query_id,structure_ref``) that maps each query
    protein to a local PDB or mmCIF structure file.  For every valid mapping the
    function:

    1. Resolves the structure file path (absolute path, filename with extension,
       or bare identifier with automatic ``.pdb`` / ``.cif`` extension lookup).
    2. Extracts residues and Cα coordinates from the structure, applying the same
       non-standard-residue handling used for PDB database structures.
    3. Aligns the query sequence to the structure-derived sequence and builds an
       aligned contact map suitable for GCN prediction.
    4. Emits a warning when the query and structure sequences differ by more
       than 10 % in length.

    Args:
        query_file (QueryFile): Object containing loaded query sequences.
        mapping_csv (str): Path to the two-column CSV file
            (``query_id,structure_ref``).
        angstrom_contact_threshold (float, optional): Distance threshold (Å) for
            contact map generation.  Defaults to 6.
        generate_contacts (int, optional): Gap-fill width for contact map alignment.
            Defaults to 2.
        alignment_gap_open (float, optional): Gap-open penalty for pairwise alignment.
            Defaults to 10.
        alignment_gap_continuation (float, optional): Gap-extension penalty.
            Defaults to 1.
        scoring_matrix (str, optional): Scoring matrix name for alignment.
            Defaults to ``"VTML80"``.
        threads (int, optional): Number of threads for contact map generation.
            Defaults to 1.

    Returns:
        Tuple[List[Tuple[AlignmentResult, np.ndarray]], List[str]]:
            A tuple of:
            - List of ``(AlignmentResult, aligned_contact_map)`` tuples for
              successfully processed mappings.
            - List of query IDs that were successfully mapped (used to exclude
              them from subsequent database searches).
    """
    mapping_csv = pathlib.Path(mapping_csv)
    mapping_dir = mapping_csv.parent

    alignments_with_coords = []
    mapped_query_ids = []

    with open(mapping_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue

            query_id = row[0].strip()
            structure_ref = row[1].strip()

            # ── resolve file path ──────────────────────────────────────
            structure_path = _resolve_structure_path(structure_ref, mapping_dir)
            if structure_path is None:
                logger.warning(
                    "Structure file not found for %s: %s", query_id,
                    structure_ref)
                continue

            # ── determine chain and filetype ───────────────────────────
            chain = _extract_chain(structure_path)
            suffix = structure_path.suffix.lower()
            filetype = "pdb" if suffix == ".pdb" else "mmcif"

            # ── load structure and extract coordinates ─────────────────
            with open(structure_path, "r", encoding="utf-8") as sf:
                structure_string = sf.read()

            try:
                sequence, coords = extract_residues_coordinates(
                    structure_string, chain=chain, filetype=filetype)
            except KeyError as e:
                logger.warning(
                    "Error extracting coordinates from %s: "
                    "non-standard residue %s; %s skipped.", structure_path,
                    str(e), query_id)
                continue
            except ValueError as e:
                logger.warning(
                    "Error processing %s: %s; %s skipped.", structure_path,
                    str(e), query_id)
                continue
            except TypeError as e:
                # e.g. biotite PDB/mmCIF parse yields invalid stack shape
                logger.warning(
                    "Structure load failed for query %s from %s: %s. "
                    "Falling back to hierarchical database search if databases "
                    "were configured; otherwise sequence-only (CNN) DeepFRI.",
                    query_id, structure_path, e)
                continue

            if sequence is None or coords is None:
                logger.warning(
                    "No coordinates found in %s; %s skipped.",
                    structure_path, query_id)
                continue

            # ── check that query exists ────────────────────────────────
            if query_id not in query_file.sequences:
                logger.warning(
                    "Query %s not found in query file; skipping.", query_id)
                continue

            query_sequence = query_file.sequences[query_id]

            # ── length mismatch warning ────────────────────────────────
            _warn_length_mismatch(query_id, query_sequence, sequence,
                                  structure_ref)

            # ── pairwise alignment ─────────────────────────────────────
            alignment_string, identity, query_coverage, target_coverage, aln_score = \
                align_pairwise(
                    query_sequence, sequence,
                    gap_open=int(alignment_gap_open),
                    gap_extend=int(alignment_gap_continuation),
                    scoring_matrix=scoring_matrix)

            aln = AlignmentResult(
                query_name=query_id,
                query_sequence=query_sequence,
                target_name=structure_path.stem,
                target_sequence=sequence,
                alignment=alignment_string,
                query_identity=identity,
                query_coverage=query_coverage,
                target_coverage=target_coverage,
                alignment_score=aln_score,
                db_name="user_structures",
                coords=coords,
                template_path=structure_path,
                template_chain=chain,
                template_filetype=filetype)

            alignments_with_coords.append(aln)
            mapped_query_ids.append(query_id)

    # ── build aligned contact maps (parallelised) ──────────────────────
    partial_map_align = partial(build_align_contact_map,
                                threshold=angstrom_contact_threshold,
                                generated_contacts=generate_contacts)

    with Pool(threads) as p:
        cmaps = list(p.map(partial_map_align, alignments_with_coords))

    # filter out failed contact maps
    aligned_cmaps = [cmap for cmap in cmaps if cmap[1] is not None]

    successfully_mapped = [aln.query_name for aln, _ in aligned_cmaps]
    logger.info(
        "User-supplied structures: %d/%d mappings produced valid contact maps.",
        len(aligned_cmaps), len(mapped_query_ids))

    return aligned_cmaps, successfully_mapped


def _initialize_processing_modes(modes: List[str],
                                 config: Dict[str, Any]) -> List[str]:
    """
    Filters processing modes based on the model config version.
    """
    filtered_modes = list(modes)
    # version 1.1 drops support for ec
    if config.get("version") == "1.1":
        if "ec" in filtered_modes:
            filtered_modes.remove("ec")
            logger.info(
                "EC number prediction is not supported in version 1.1.")

    if len(filtered_modes) == 0:
        raise ValueError("No processing modes selected.")
    return filtered_modes


def _run_prediction_loop(predictor, data_iterable: iter, data_len: int,
                         net_type: str, tsv_writer: csv.writer,
                         description: str):
    """
    A helper function to run a prediction loop for either GCN or CNN.
    """
    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}], {rate_fmt}{postfix}"
    # Assuming BAR_FORMAT and sys.stdout are available in scope
    # or passed as arguments if this is in a different module.
    for item in tqdm(data_iterable,
                     total=data_len,
                     desc=description,
                     bar_format=BAR_FORMAT,
                     file=sys.stdout,
                     mininterval=10):
        if net_type == "gcn":
            # item is (aln, aligned_cmap)
            aln, aligned_cmap = item
            query_id = aln.query_name
            pred_vector = predictor.forward_pass(seqres=aln.query_sequence,
                                                 cmap=aligned_cmap)
        else:  # net_type == "cnn"
            # item is (query_id, sequence)
            query_id, sequence = item
            pred_vector = predictor.forward_pass(seqres=sequence)

        out_row = [query_id, net_type] + pred_vector.tolist()
        tsv_writer.writerow(out_row)


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
        remove_intermediate=False,
        threads: int = 1,
        save_structures: bool = False,
        save_cmaps: bool = False,
        skip_matrix: bool = False,
        scoring_matrix: str = "VTML80",
        mapped_structures_csv: Optional[str] = None,
        save_aligned_structures: bool = False,
        save_raw_alignments: bool = False):
    """
    Predict protein function using DeepFRI.

    This function is the main entry point for the prediction pipeline. It aligns
    query sequences to databases, generates contact maps, and runs DeepFRI
    predictions for specified functional categories.

    Args:
        query_file (QueryFile): Object containing query sequences.
        databases (Tuple[Database]): Tuple of database objects to search against.
        weights (str): Path to folder containing DeepFRI model weights.
        output_path (str): Path to directory for saving results.
        deepfri_processing_modes (List[str], optional): List of modes to predict.
            Options: "ec", "bp", "mf", "cc".
            Defaults to ["ec", "bp", "mf", "cc"].
        angstrom_contact_threshold (float, optional): Distance threshold for contact maps.
            Defaults to 6.
        generate_contacts (int, optional): Gap for generating contact maps.
            Defaults to 2.
        alignment_gap_open (float, optional): Gap open penalty for alignment.
            Defaults to 10.
        alignment_gap_continuation (float, optional): Gap extension penalty.
            Defaults to 1.
        remove_intermediate (bool, optional): Remove intermediate files.
            Defaults to False.
        threads (int, optional): Number of threads for parallel processing.
            Defaults to 1.
        save_structures (bool, optional): Save aligned structures to disk.
            Defaults to False.
        save_cmaps (bool, optional): Save generated contact maps to disk.
            Defaults to False.
        skip_matrix (bool, optional): Skip writing full prediction matrices.
            Defaults to False.
        scoring_matrix (str, optional): Scoring matrix for alignment.
            Defaults to "VTML80".
        mapped_structures_csv (str, optional): Path to a two-column CSV file
            (``query_id,structure_ref``) that maps query sequences to local
            PDB/CIF structure files.  When provided, these mappings are
            processed before database searches and take priority.
            Defaults to None.
        save_aligned_structures (bool, optional): Write carved PDB fragments
            (template atoms mapped through PyOpal; query insertions omitted)
            under ``aligned_structures/``. Defaults to False.
        save_raw_alignments (bool, optional): If True, write per-database
            ``{db_name}_raw_alignments.fasta`` in the output directory with
            gapped query/target sequences and a ``#alignment_string`` line for
            each query that has a valid aligned contact map. Defaults to False.

    Returns:
        None: Results are written to files in output_path.

    See Also:
        hierarchical_database_search: For the initial search step.
        load_mapped_structures: For user-supplied structure mapping.
    """

    # load DeepFRI model
    deepfri_models_config = load_deepfri_config(weights)
    deepfri_processing_modes = _initialize_processing_modes(
        deepfri_processing_modes, deepfri_models_config)

    weights = pathlib.Path(weights)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    aligned_cmaps = []

    # ── user-supplied structure mappings (processed first) ─────────────
    if mapped_structures_csv is not None:
        user_cmaps, user_mapped_ids = load_mapped_structures(
            query_file=query_file,
            mapping_csv=mapped_structures_csv,
            angstrom_contact_threshold=angstrom_contact_threshold,
            generate_contacts=generate_contacts,
            alignment_gap_open=alignment_gap_open,
            alignment_gap_continuation=alignment_gap_continuation,
            scoring_matrix=scoring_matrix,
            threads=threads)
        aligned_cmaps.extend(user_cmaps)

    for db in databases:
        # SEQUENCE ALIGNMENT
        # calculate already aligned sequences
        alignments = align_mmseqs_results(
            best_matches_filepath=db.mmseqs_result,
            sequence_db=db.sequence_db,
            alignment_gap_open=alignment_gap_open,
            alignment_gap_extend=alignment_gap_continuation,
            threads=threads,
            scoring_matrix=scoring_matrix)

        try:
            # set a db name for alignments
            for aln in alignments:
                aln.db_name = db.name

            aligned_queries = [aln[0].query_name for aln in aligned_cmaps]
            new_alignments = {
                aln.query_name: aln
                for aln in alignments if aln.query_name not in aligned_queries
                and aln.query_name in query_file.sequences
            }

            # CONTACT MAP ALIGNMENT
            # initially designed as a separate step
            # some protein structures in PDB are not formatted correctly
            # so contact map alignment fails for them
            # for this cases we replace closest experimental structure with
            # closest predicted structure if available
            # if no alignments were found - report

            # remove broken structures
            if db.name == "highquality_clust30":
                data_path = pathlib.Path(__file__).parent / "assets"
                # convert to abspath
                data_path = data_path.resolve()
                with open(data_path / "highquality_clust30_error_ids.pkl",
                          "rb") as f:
                    error_ids = pickle.load(f)
                # filter out broken structures
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
            # in form of C-alpha coordinates
            if save_structures:
                save_dir = output_path / "structures" / db.name
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = None

            coord_bundles = extract_calpha_coords(db,
                                                  target_ids,
                                                  query_ids,
                                                  save_directory=save_dir,
                                                  threads=threads)

            for aln, bundle in zip(new_alignments.values(), coord_bundles):
                coord, tmpl, ch, ftype = bundle
                aln.coords = coord
                aln.template_structure_string = tmpl
                aln.template_chain = ch
                aln.template_filetype = ftype

        # troubleshoot cases where alignments are empty
        except IndexError:
            logger.info("No alignments found for %s.", db.name)
            new_alignments = {}
            continue

        # if new alignments are empty - result is empty as well
        partial_map_align = partial(build_align_contact_map,
                                    threshold=angstrom_contact_threshold,
                                    generated_contacts=generate_contacts)

        with Pool(threads) as p:
            cmaps = list(p.map(partial_map_align, new_alignments.values()))

        # filter errored contact maps
        # returned as Tuple[AlignmentResult, None] from `retrieve_align_contact_map`
        partial_cmaps = [cmap for cmap in cmaps if cmap[1] is not None]
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

    if save_aligned_structures:
        as_dir = output_path / "aligned_structures"
        as_dir.mkdir(parents=True, exist_ok=True)
        carved_tasks = [(aln, as_dir / f"{aln.query_name}.pdb")
                        for aln, _cmap in aligned_cmaps]
        if carved_tasks:
            if threads > 1:
                logger.info(
                    "Writing %d carved structures (%d parallel workers).",
                    len(carved_tasks), threads)
                with Pool(threads) as pool:
                    pool.map(_write_carved_structure_worker, carved_tasks)
            else:
                for task in carved_tasks:
                    _write_carved_structure_worker(task)

    aligned_queries = [aln[0].query_name for aln in aligned_cmaps]
    unaligned_queries = {
        query_id: seq
        for query_id, seq in query_file.sequences.items()
        if query_id not in aligned_queries
    }

    # WRITE ALIGNMENT RESULTS
    alignment_results_file = output_path / "alignment_summary.tsv"

    with open(alignment_results_file, "w", encoding="utf-8") as aln_output:
        tsv_writer = csv.writer(aln_output, delimiter="\t")
        tsv_writer.writerow(ALIGNMENT_HEADER)
        for aln, _ in aligned_cmaps:
            tsv_writer.writerow([
                aln.query_name, True, aln.target_name, aln.db_name,
                aln.query_identity, aln.query_coverage, aln.target_coverage
            ])
        for query_id in unaligned_queries:
            tsv_writer.writerow(
                [query_id, False, np.nan, np.nan, np.nan, np.nan, np.nan])

    if save_raw_alignments and aligned_cmaps:
        by_db: Dict[str, List[AlignmentResult]] = defaultdict(list)
        for aln, _ in aligned_cmaps:
            key = aln.db_name if aln.db_name is not None else "unknown"
            by_db[key].append(aln)
        for db_name, alns in by_db.items():
            safe = db_name.replace("/", "_").replace("\\", "_")
            fasta_path = output_path / f"{safe}_raw_alignments.fasta"
            blocks = [_raw_alignment_fasta_record(aln) for aln in alns]
            fasta_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    ### FUNCTION PREDICTION ###
    # sort cmaps by length of query sequence
    aligned_cmaps = sorted(aligned_cmaps,
                           key=lambda x: len(x[0].query_sequence))
    # sort unaligned queries by length
    unaligned_queries = dict(
        sorted(unaligned_queries.items(), key=lambda x: len(x[1])))

    # output_file_name = output_path / "results.tsv"
    # output_buffer = open(output_file_name, "w", encoding="utf-8")
    # csv_writer = csv.writer(output_buffer, delimiter="\t")
    # csv_writer.writerow(OUTPUT_HEADER)

    matrices = {}
    json_configs = {}
    for i, mode in enumerate(deepfri_processing_modes):
        # load model go terms
        model_path = deepfri_models_config["gcn"][mode]
        config_path = model_path.rsplit(".", 1)[0] + "_model_params.json"
        json_configs[mode] = config_path
        GOTERMS = get_json_values(config_path, "goterms")

        # create output file for each mode (or in-memory buffer if skipping)
        if skip_matrix:
            # Use in-memory buffer instead of file
            import io
            output_buffer = io.StringIO()
            matrices[mode] = output_buffer  # Store buffer for later reading
        else:
            output_matrix = output_path / f"prediction_matrix_{mode}.tsv"
            matrices[mode] = output_matrix
            output_buffer = open(output_matrix, "w", encoding="utf-8")

        tsv_writer = csv.writer(output_buffer, delimiter="\t")
        tsv_writer.writerow(["protein", "network_type"] + GOTERMS)

        logger.info("Processing mode: %s; %i/%i", DEEPFRI_MODES[mode], i + 1,
                    len(deepfri_processing_modes))

        # GCN prediction
        gcn_prots = len(aligned_cmaps)
        if gcn_prots > 0:
            net_type = "gcn"

            # GCN for queries with aligned contact map
            gcn_path = deepfri_models_config[net_type][mode]
            gcn = Predictor(gcn_path, threads=threads)
            _run_prediction_loop(
                predictor=gcn,
                data_iterable=aligned_cmaps,
                data_len=len(aligned_cmaps),
                net_type=net_type,
                tsv_writer=tsv_writer,
                description=f"Predicting with GCN ({DEEPFRI_MODES[mode]})")
            del gcn  # Explicitly free memory

        # CNN for queries without satisfying alignments
        cnn_prots = len(unaligned_queries)
        if cnn_prots > 0:
            net_type = "cnn"
            cnn_path = deepfri_models_config[net_type][mode]
            cnn = Predictor(cnn_path, threads=threads)
            _run_prediction_loop(
                predictor=cnn,
                data_iterable=unaligned_queries.items(),
                data_len=len(unaligned_queries),
                net_type=net_type,
                tsv_writer=tsv_writer,
                description=f"Predicting with CNN ({DEEPFRI_MODES[mode]})")
            del cnn  # Explicitly free memory

        # Close file buffer if writing to file (keep StringIO for reading later)
        if not skip_matrix:
            output_buffer.close()

    ### FORMAT AND CREATE FINAL OUTPUT FILES ###
    # combine mode-specific matrices into a single file
    # open and load alignment data
    with open(alignment_results_file, "r", encoding="utf-8") as aln_input:
        tsv_reader = csv.reader(aln_input, delimiter="\t")
        next(tsv_reader)  # skip header
        alignment_data = {row[0]: row[1:] for row in tsv_reader}

    final_output = output_path / "results.tsv"
    with open(final_output, "w", encoding="utf-8") as fout:
        fout.write("\t".join(FINAL_OUTPUT_HEADER) + "\n")
        for mode, matrix_source in matrices.items():
            json_path = json_configs[mode]
            GONAMES = get_json_values(json_path, "gonames")

            # Handle both file paths and StringIO buffers
            import io
            if isinstance(matrix_source, io.StringIO):
                # Read from in-memory buffer
                matrix_source.seek(0)  # Reset to beginning
                matrix_content = matrix_source.getvalue()
                matrix_lines = matrix_content.strip().split('\n')
                tsv_reader = csv.reader(matrix_lines, delimiter="\t")
            else:
                # Read from file
                with open(matrix_source, "r",
                          encoding="utf-8") as matrix_input:
                    tsv_reader = csv.reader(matrix_input, delimiter="\t")
                    # get term names from header
                    header = next(tsv_reader)
                    terms = header[
                        2:]  # skip first two columns (Protein and Type)
                    term_to_name = {
                        term: name
                        for term, name in zip(terms, GONAMES)
                    }
                    # get ids with scores > 0.1
                    for row in tsv_reader:
                        query_id = row[0]
                        net_type = row[1]
                        scores = row[2:]
                        term_score = {
                            terms[i]: float(scores[i])
                            for i in range(len(terms))
                            if float(scores[i]) >= 0.1
                        }
                        sorted_term_score = dict(
                            sorted(term_score.items(),
                                   key=lambda item: item[1],
                                   reverse=True))
                        # print results and add go names
                        for term, score in sorted_term_score.items():
                            go_name = term_to_name.get(term, "Unknown")
                            aln_info = alignment_data.get(
                                query_id, [np.nan] * 6)
                            aligned, target_id, database, target_identity, query_cov, target_cov = aln_info
                            fout.write(
                                f"{query_id}\t{net_type}\t{DEEPFRI_MODES[mode]}\t{term}\t{score:.4f}\t{go_name}\t"
                                f"\t{aligned}\t{target_id}\t{database}\t{target_identity}\t{query_cov}\t{target_cov}\n"
                            )
                continue  # Skip to next mode after processing file

            # Process in-memory buffer (same logic as file)
            header = next(tsv_reader)
            terms = header[2:]  # skip first two columns (Protein and Type)
            term_to_name = {term: name for term, name in zip(terms, GONAMES)}
            # get ids with scores > 0.1
            for row in tsv_reader:
                query_id = row[0]
                net_type = row[1]
                scores = row[2:]
                term_score = {
                    terms[i]: float(scores[i])
                    for i in range(len(terms)) if float(scores[i]) >= 0.1
                }
                sorted_term_score = dict(
                    sorted(term_score.items(),
                           key=lambda item: item[1],
                           reverse=True))
                # print results and add go names
                for term, score in sorted_term_score.items():
                    go_name = term_to_name.get(term, "Unknown")
                    aln_info = alignment_data.get(query_id, [np.nan] * 6)
                    aligned, target_id, database, target_identity, query_cov, target_cov = aln_info
                    fout.write(
                        f"{query_id}\t{net_type}\t{DEEPFRI_MODES[mode]}\t{term}\t{score:.4f}\t{go_name}\t"
                        f"\t{aligned}\t{target_id}\t{database}\t{target_identity}\t{query_cov}\t{target_cov}\n"
                    )

    if remove_intermediate:
        for db in databases:
            remove_intermediate_files([db.sequence_db, db.mmseqs_db])

    logger.info("meta-DeepFRI finished successfully.")
