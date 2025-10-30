import importlib.metadata
import logging
import sys
from pathlib import Path
import os
import numpy as np

import click
from click._compat import get_text_stderr
from click.exceptions import UsageError
from click.utils import echo

from mDeepFRI.pipeline import (hierarchical_database_search, load_query_file,
                               predict_protein_function)
from mDeepFRI.utils import download_model_weights, generate_config_json
from mDeepFRI.bio_utils import load_structure, get_residues_coordinates, calculate_contact_map

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app_version = importlib.metadata.version("mDeepFRI")


def _show_usage_error(self, file=None):
    if file is None:
        file = get_text_stderr()
    color = None
    if self.ctx is not None:
        color = self.ctx.color
        echo(self.ctx.get_help() + '\n', file=file, color=color)
    echo('Error: %s' % self.format_message(), file=file, color=color)


UsageError.show = _show_usage_error


def search_options(function):
    function = click.option(
        "-i",
        "--input",
        required=True,
        type=click.Path(exists=True),
        help="Path to an input protein fasta file (FASTA file, may be gzipped).",
    )(function)
    function = click.option(
        "-o",
        "--output",
        required=True,
        type=click.Path(exists=False),
        help="Path to output file.",
    )(function)
    function = click.option(
        "-d",
        "--db-path",
        required=False,
        type=click.Path(exists=True),
        multiple=True,
        help="Path to a structures database compressed with FoldComp.",
    )(function)
    function = click.option(
        "-s",
        "--sensitivity",
        required=False,
        default=5.7,
        type=click.FloatRange(1, 7.5),
        help="Sensitivity of the MMSeqs2 search. Default is 5.7.",
    )(function)
    function = click.option(
        "--min-length",
        required=False,
        default=None,
        type=int,
        help="Minimum length of the protein sequence.",
    )(function)
    function = click.option(
        "--max-length",
        required=False,
        default=None,
        type=int,
        help="Maximum length of the protein sequence.",
    )(function)
    function = click.option(
        "--min-bitscore",
        required=False,
        default=0,
        type=float,
        help="Minimum bitscore for MMseqs2 alignment.",
    )(function)
    function = click.option(
        "--max-eval",
        required=False,
        default=0.001,
        type=float,
        help="Maximum e-value for MMseqs2 alignment.",
    )(function)
    function = click.option(
        "--min_identity",
        required=False,
        default=0.5,
        type=float,
        help="Minimum identity for MMseqs2 alignment.",
    )(function)
    function = click.option(
        "--min-coverage",
        required=False,
        default=0.9,
        type=float,
        help=
        "Minimum coverage for MMseqs2 alignment for both query and target sequences.",
    )(function)
    function = click.option(
        "--top-k",
        required=False,
        default=1,
        type=int,
        help="Number of top MMSeqs2 hits to save. Default is 1.",
    )(function)
    function = click.option(
        "--overwrite",
        required=False,
        default=False,
        type=bool,
        is_flag=True,
        help="Overwrite existing files.",
    )(function)
    function = click.option(
        "-t",
        "--threads",
        required=False,
        default=1,
        type=int,
        help="Number of threads to use. Default is 1.",
    )(function)
    function = click.option(
        "--skip-pdb",
        required=False,
        default=False,
        type=bool,
        is_flag=True,
        help="Skip PDB100 database search.",
    )(function)
    function = click.option(
        "--tmpdir",
        required=False,
        default=None,
        type=click.Path(),
        help="Path to a temporary directory. Required for very large searches",
    )(function)
    return function


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.version_option(version=app_version)
def main(debug):
    """mDeepFRI"""

    loggers = [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]
    for log in loggers:
        if debug:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)


@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(exists=False),
    help="Path to folder where the weights will be stored.",
)
@click.option("-v",
              "--version",
              required=True,
              type=click.Choice(["1.0", "1.1"]),
              help="Version of the model.")
@main.command
def get_models(output, version):
    """Download model weights for mDeepFRI."""

    logger.info("Downloading DeepFRI models.")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    download_model_weights(output_path, version)
    generate_config_json(output_path, version)
    logger.info(f"DeepFRI models v{version} downloaded to {output_path}.")


@click.option(
    "-w",
    "--weights_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to a folder containing model weights.",
)
@click.option(
    "-v",
    "--version",
    required=True,
    type=click.Choice(["1.0", "1.1"]),
    help="Version of the model.",
)
@main.command
def generate_config(weights_path, version):
    """
    Generate a config file for mDeepFRI.
    This is used only when the model weights are downloaded manually.
    """

    logger.info("Generating config file for mDeepFRI.")
    weights_path = Path(weights_path)
    generate_config_json(weights_path, version)
    logger.info(f"Config file generated in {weights_path}.")


@main.command
@search_options
def search_databases(input, output, db_path, sensitivity, min_length,
                     max_length, min_bitscore, max_eval, min_ident, min_coverage,
                     top_k, overwrite, threads, skip_pdb, tmpdir):
    """
    Hierarchically search FoldComp databases for similar proteins with
    MMSeqs2. Based on the thresholds from https://doi.org/10.1038/s41586-023-06510-w.
    """

    # write command parameters to log
    logger.info("Command parameters:")
    logger.info("Input:                        %s", input)
    logger.info("Output:                       %s", output)
    logger.info("Database:                     %s", db_path)
    logger.info("Sensitivity:                  %s", sensitivity)
    logger.info("Minimum length:               %s", min_length)
    logger.info("Maximum length:               %s", max_length)
    logger.info("Minimum bitscore:             %s", min_bitscore)
    logger.info("Maximum e-value:              %s", max_eval)
    logger.info("Minimum identity:             %s", min_ident)
    logger.info("Minimum coverage:             %s", min_coverage)
    logger.info("Top k results:                %s", top_k)
    logger.info("Overwrite:                    %s", overwrite)
    logger.info("Threads:                      %s", threads)
    logger.info("Skip PDB:                     %s", skip_pdb)

    query_file = load_query_file(input)
    hierarchical_database_search(query_file=query_file,
                                 databases=db_path,
                                 output_path=output,
                                 sensitivity=sensitivity,
                                 min_bits=min_bitscore,
                                 max_eval=max_eval,
                                 min_ident=min_ident,
                                 min_coverage=min_coverage,
                                 top_k=top_k,
                                 skip_pdb=skip_pdb,
                                 overwrite=overwrite,
                                 tmpdir=tmpdir,
                                 threads=threads)


@main.command()
@search_options
@click.option(
    "-w",
    "--weights",
    required=True,
    type=click.Path(exists=True),
    help="Path to a folder containing model weights.",
)
@click.option(
    "-p",
    "--processing-modes",
    default=("bp", "cc", "ec", "mf"),
    type=click.Choice(["bp", "cc", "ec", "mf"]),
    multiple=True,
    help="Processing modes. Default is all"
    "(biological process, cellular component, enzyme commission, molecular function).",
)
@click.option(
    "-a",
    "--angstrom-contact-thresh",
    default=6,
    type=float,
    help="Angstrom contact threshold. Default is 6.",
)
@click.option(
    "--generate-contacts",
    default=2,
    type=int,
    help="Gap fill threshold during contact map alignment.",
)
@click.option(
    "--alignment-gap-open",
    default=10,
    type=int,
    help="Gap open penalty for contact map alignment.",
)
@click.option(
    "--alignment-gap-extend",
    default=1,
    type=int,
    help="Gap extend penalty for contact map alignment.",
)
@click.option(
    "--cmap-identity",
    default=0.5,
    type=float,
    help="Minimum identity for contact map alignment.",
)
@click.option(
    "--cmap-coverage",
    default=0.9,
    type=float,
    help="Minimum coverage for contact map alignment.",
)
@click.option(
    "--remove-intermediate",
    default=False,
    type=bool,
    is_flag=True,
    help="Remove intermediate files.",
)
@click.option(
    "--save-structures",
    default=False,
    type=bool,
    is_flag=True,
    help="Save structures of the top hits.",
)
@click.option(
    "--save-cmaps",
    default=False,
    type=bool,
    is_flag=True,
    help="Save contact maps of the top hits.",
)

def predict_function(input, db_path, weights, output, processing_modes,
                     angstrom_contact_thresh, generate_contacts,
                     sensitivity, min_bitscore,
                     max_eval, min_identity,
                     min_coverage, top_k, alignment_gap_open,
                     alignment_gap_extend, cmap_identity, tmpdir,
                     cmap_coverage, remove_intermediate, overwrite,
                     threads, skip_pdb, min_length, max_length,
                     save_structures, save_cmaps):
    """Predict protein function from sequence."""
    logger.info("Starting Metagenomic-DeepFRI.")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    # write command parameters to log
    logger.info("Command parameters:")
    logger.info("Input:                         %s", input)
    logger.info("Database:                      %s", db_path)
    logger.info("Weights:                       %s", weights)
    logger.info("Output:                        %s", output)
    logger.info("Processing modes:              %s", processing_modes)
    logger.info("Angstrom contact threshold:    %s", angstrom_contact_thresh)
    logger.info("Generate contacts:             %s", generate_contacts)
    logger.info("MMSeqs2 sensitivity:           %s", sensitivity)
    logger.info("MMSeqs2 minimum bitscore:      %s", min_bitscore)
    logger.info("MMSeqs2 maximum e-value:       %s", max_eval)
    logger.info("MMSeqs2 minimum identity:      %s", min_identity)
    logger.info("Top k results:                 %s", top_k)
    logger.info("Alignment gap open:            %s", alignment_gap_open)
    logger.info("Alignment gap extend:          %s", alignment_gap_extend)
    logger.info("Alignment minimum identity:    %s", cmap_identity)
    logger.info("Alignment minimum coverage:    %s", cmap_coverage)
    logger.info("Remove intermediate:           %s", remove_intermediate)
    logger.info("Overwrite:                     %s", overwrite)
    logger.info("Threads:                       %s", threads)
    logger.info("Temporary dir:                 %s", tmpdir)
    logger.info("Skip PDB:                      %s", skip_pdb)
    logger.info("Minimum length:                %s", min_length)
    logger.info("Maximum length:                %s", max_length)
    logger.info("Save structures:               %s", save_structures)
    logger.info("Save contact maps:             %s", save_cmaps)

    query_file = load_query_file(
        query_file=input, 
        min_length=min_length, 
        max_length=max_length)
    
    deepfri_dbs = hierarchical_database_search(
        query_file=query_file,
        output_path=output_path / "database_search",
        databases=db_path,
        sensitivity=sensitivity,
        min_bits=min_bitscore,
        max_eval=max_eval,
        min_ident=min_identity,
        min_coverage=min_coverage,
        top_k=top_k,
        skip_pdb=skip_pdb,
        overwrite=overwrite,
        tmpdir=tmpdir,
        threads=threads)

    predict_protein_function(
        query_file=query_file,
        databases=deepfri_dbs,
        weights=weights,
        output_path=output_path,
        deepfri_processing_modes=processing_modes,
        angstrom_contact_threshold=angstrom_contact_thresh,
        generate_contacts=generate_contacts,
        alignment_gap_open=alignment_gap_open,
        alignment_gap_continuation=alignment_gap_extend,
        identity_threshold=cmap_identity,
        coverage_threshold=cmap_coverage,
        remove_intermediate=remove_intermediate,
        save_structures=save_structures,
        save_cmaps=save_cmaps)


@main.command()
@click.option("--input_dir", "-i", type=click.Path(exists=True), required=True,
              help="Directory containing PDB or mmCIF files.")
@click.option("--output_dir", "-o", type=click.Path(), required=True,
              help="Directory to save computed contact maps.")
@click.option("--threshold", "-t", default=6.0, show_default=True,
              help="Distance threshold in Å for contact map.")
def make_cmaps(input_dir, output_dir, threshold):
    "Compute CA contact maps for all PDB/mmCIF files in a directory."
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith((".pdb", ".cif")):
            continue
        filetype = "pdb" if fname.endswith(".pdb") else "mmcif"
        with open(os.path.join(input_dir, fname)) as f:
            structure_str = f.read()
        residues, coords = get_residues_coordinates(load_structure(structure_str, filetype), chain="A")
        cmap = calculate_contact_map(coords, threshold)
        np.save(os.path.join(output_dir, fname.replace(".pdb", "_cmap.npy")), cmap)

if __name__ == "__main__":
    main()
