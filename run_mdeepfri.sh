#!/bin/bash

# =============================================================================
# mDeepFRI predict-function runner script
# =============================================================================
# This script runs mDeepFRI with multiple identity bins and generate_contacts
# values. Modify the variables below to change input files and databases.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION - Modify these variables as needed
# -----------------------------------------------------------------------------

# Input FASTA file
#INPUT_FILE="input_sequences/sample.fasta"
INPUT_FILE="input_sequences/landscape_afdb_50k.fasta"

# Path to model weights folder
WEIGHTS_PATH="weights_folder/"

# Output directory
OUTPUT_DIR="results/landscape_afdb_50k"
#OUTPUT_DIR="results/test_no_go_terms/"

# Timing log file (will be created in OUTPUT_DIR)
TIMING_LOG="${OUTPUT_DIR}/timing.log"

# Terminal output log file (will be created in OUTPUT_DIR)
# Set to empty string to disable saving terminal output
TERMINAL_LOG="${OUTPUT_DIR}/terminal_output.log"
# Set to true to save all terminal output (stdout + stderr) to TERMINAL_LOG
SAVE_TERMINAL_OUTPUT=true

# Database paths (can specify multiple databases)
# Example: DATABASES=("database1" "database2")
# Leave empty to use only PDB100: DATABASES=()
DATABASES=("/mnt/storage_10T/afdb_foldcomp/afdb_uniprot_v4")
#DATABASES=("/home/FilipS/software/Metagenomic-DeepFRI/afdb/afdb_rep_v4")

# Number of top MMSeqs2 hits to save
TOP_K=100

# Identity bins (format: "low,high")
# Example: IDENTITY_BINS=("0.9,1.0" "0.8,0.9" "0.7,0.8" "0.6,0.7")
IDENTITY_BINS=("0.90,1.00" "0.80,0.90" "0.70,0.80" "0.60,0.70" "0.50,0.60" "0.40,0.50" "0.30,0.40")
#IDENTITY_BINS=("0.90,1.00" "0.80,0.90" "0.70,0.80")

# Generate contacts values
# Example: GENERATE_CONTACTS=(0 1 2 3 4)
GENERATE_CONTACTS=(0 1 2 3 4)

# Processing modes (bp, cc, ec, mf)
PROCESSING_MODES=()

# Additional options
SAVE_CMAPS=true
SAVE_STRUCTURES=true
THREADS=14
SKIP_PDB=true

# -----------------------------------------------------------------------------
# ADVANCED OPTIONS (optional - uncomment and modify if needed)
# -----------------------------------------------------------------------------

# MMSeqs2 search parameters
# SENSITIVITY=5.7
# MIN_BITSCORE=0
# MAX_EVAL=0.001
MIN_IDENTITY=0.0
# MIN_COVERAGE=0.9

# Contact map alignment parameters
# ANGSTROM_CONTACT_THRESH=6
# ALIGNMENT_GAP_OPEN=10
# ALIGNMENT_GAP_EXTEND=1
CMAP_IDENTITY=0.0
CMAP_COVERAGE=0.0

# Other options
PER_QUERY="random"  # or "topbits"
DROP_SELF_HITS=true
SEED=42  # Random seed (important when PER_QUERY="random" for reproducibility)
# REMOVE_INTERMEDIATE=false

# =============================================================================
# SCRIPT EXECUTION - Do not modify below unless you know what you're doing
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Timing functions
# -----------------------------------------------------------------------------

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to get elapsed time in seconds
get_elapsed() {
    local start=$1
    local end=$2
    echo "$((end - start))"
}

# Function to format seconds into human-readable format
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# Function to log message with timestamp
log_timing() {
    local message="$1"
    local timestamp=$(get_timestamp)
    echo "[$timestamp] $message" | tee -a "$TIMING_LOG"
}

# Initialize timing log (create output directory and log file)
mkdir -p "$OUTPUT_DIR"
> "$TIMING_LOG"  # Clear/create log file

# Initialize terminal output log if enabled
if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
    > "$TERMINAL_LOG"  # Clear/create terminal log file
    log_timing "Terminal output will be saved to: $TERMINAL_LOG"
    # Function to echo and log to terminal log file
    echo_and_log() {
        echo "$@"
        echo "$@" >> "$TERMINAL_LOG"
    }
else
    # If not logging, just echo normally
    echo_and_log() {
        echo "$@"
    }
fi

# Record script start time
SCRIPT_START=$(date +%s)
SCRIPT_START_STR=$(get_timestamp)

log_timing "================================================================================"
log_timing "mDeepFRI predict-function - Execution Started"
log_timing "================================================================================"
log_timing "Start time:         $SCRIPT_START_STR"
log_timing "Input file:         $INPUT_FILE"
log_timing "Weights:            $WEIGHTS_PATH"
log_timing "Output directory:   $OUTPUT_DIR"
log_timing "Top-K:              $TOP_K"
log_timing "Databases:          ${DATABASES[@]:-PDB100 only}"
log_timing "Identity bins:      ${IDENTITY_BINS[@]}"
log_timing "Generate contacts:  ${GENERATE_CONTACTS[@]}"
log_timing "Processing modes:   ${PROCESSING_MODES[@]}"
log_timing "Threads:            $THREADS"
log_timing "Skip PDB:           $SKIP_PDB"
log_timing "Per query:          $PER_QUERY"
log_timing "Drop self hits:     $DROP_SELF_HITS"
log_timing "Seed:               ${SEED:-0}"
log_timing "================================================================================"
log_timing ""

echo_and_log "================================================================================"
echo_and_log "mDeepFRI predict-function"
echo_and_log "================================================================================"
echo_and_log "Input file:        $INPUT_FILE"
echo_and_log "Weights:           $WEIGHTS_PATH"
echo_and_log "Output directory:  $OUTPUT_DIR"
echo_and_log "Top-K:             $TOP_K"
echo_and_log "Databases:         ${DATABASES[@]:-PDB100 only}"
echo_and_log "Identity bins:     ${IDENTITY_BINS[@]}"
echo_and_log "Generate contacts: ${GENERATE_CONTACTS[@]}"
echo_and_log "Processing modes:  ${PROCESSING_MODES[@]}"
echo_and_log "Threads:           $THREADS"
echo_and_log "Timing log:        $TIMING_LOG"
if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
    echo_and_log "Terminal log:       $TERMINAL_LOG"
fi
echo_and_log "================================================================================"
echo_and_log ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo_and_log "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check if weights directory exists
if [ ! -d "$WEIGHTS_PATH" ]; then
    echo_and_log "ERROR: Weights directory not found: $WEIGHTS_PATH"
    exit 1
fi

# Build the command
CMD="mDeepFRI predict-function"
CMD="$CMD --input \"$INPUT_FILE\""
CMD="$CMD --weights \"$WEIGHTS_PATH\""
CMD="$CMD --output \"$OUTPUT_DIR\""
CMD="$CMD --top-k $TOP_K"

# Add database paths
if [ ${#DATABASES[@]} -gt 0 ]; then
    for db in "${DATABASES[@]}"; do
        CMD="$CMD --db-path \"$db\""
    done
fi

# Add skip-pdb flag if requested
if [ "$SKIP_PDB" = true ]; then
    CMD="$CMD --skip-pdb"
fi

# Add processing modes (skip if empty to disable GO-term prediction)
if [ ${#PROCESSING_MODES[@]} -gt 0 ]; then
    for mode in "${PROCESSING_MODES[@]}"; do
        CMD="$CMD --processing-modes $mode"
    done
else
    echo_and_log "PROCESSING_MODES is empty - GO-term prediction will be skipped"
    CMD="$CMD --skip-prediction"
fi

# Add identity bins
for bin in "${IDENTITY_BINS[@]}"; do
    CMD="$CMD --identity-bin \"$bin\""
done

# Add generate contacts
for gc in "${GENERATE_CONTACTS[@]}"; do
    CMD="$CMD --generate-contacts $gc"
done

# Add flags
if [ "$SAVE_CMAPS" = true ]; then
    CMD="$CMD --save-cmaps"
fi

if [ "$SAVE_STRUCTURES" = true ]; then
    CMD="$CMD --save-structures"
fi

CMD="$CMD --threads $THREADS"

# Add optional advanced parameters if they are set
if [ -n "${SENSITIVITY:-}" ]; then
    CMD="$CMD --sensitivity $SENSITIVITY"
fi

if [ -n "${MIN_BITSCORE:-}" ]; then
    CMD="$CMD --min-bitscore $MIN_BITSCORE"
fi

if [ -n "${MAX_EVAL:-}" ]; then
    CMD="$CMD --max-eval $MAX_EVAL"
fi

if [ -n "${MIN_IDENTITY:-}" ]; then
    CMD="$CMD --min_identity $MIN_IDENTITY"
fi

if [ -n "${MIN_COVERAGE:-}" ]; then
    CMD="$CMD --min-coverage $MIN_COVERAGE"
fi

if [ -n "${ANGSTROM_CONTACT_THRESH:-}" ]; then
    CMD="$CMD --angstrom-contact-thresh $ANGSTROM_CONTACT_THRESH"
fi

if [ -n "${ALIGNMENT_GAP_OPEN:-}" ]; then
    CMD="$CMD --alignment-gap-open $ALIGNMENT_GAP_OPEN"
fi

if [ -n "${ALIGNMENT_GAP_EXTEND:-}" ]; then
    CMD="$CMD --alignment-gap-extend $ALIGNMENT_GAP_EXTEND"
fi

if [ -n "${CMAP_IDENTITY:-}" ]; then
    CMD="$CMD --cmap-identity $CMAP_IDENTITY"
fi

if [ -n "${CMAP_COVERAGE:-}" ]; then
    CMD="$CMD --cmap-coverage $CMAP_COVERAGE"
fi

if [ -n "${PER_QUERY:-}" ]; then
    CMD="$CMD --per-query $PER_QUERY"
fi

if [ -n "${DROP_SELF_HITS:-}" ]; then
    if [ "$DROP_SELF_HITS" = true ]; then
        CMD="$CMD --drop-self-hits"
    else
        CMD="$CMD --keep-self-hits"
    fi
fi

if [ -n "${SEED:-}" ]; then
    CMD="$CMD --seed $SEED"
fi

if [ -n "${REMOVE_INTERMEDIATE:-}" ] && [ "$REMOVE_INTERMEDIATE" = true ]; then
    CMD="$CMD --remove-intermediate"
fi

# Print the command (for debugging)
echo_and_log "Running command:"
echo_and_log "$CMD"
echo_and_log ""
echo_and_log "================================================================================"
echo_and_log ""

# Record execution start time
EXEC_START=$(date +%s)
EXEC_START_STR=$(get_timestamp)
log_timing "Pipeline execution started at: $EXEC_START_STR"
log_timing "Command: $CMD"
log_timing ""

# Execute the command and capture exit status
set +e  # Temporarily disable exit on error to capture timing even if command fails

# Save terminal output to log file if enabled
if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
    # Use tee to save both stdout and stderr to log file while still displaying them
    # Note: This will also save the echo statements above, which is useful for context
    eval $CMD 2>&1 | tee -a "$TERMINAL_LOG"
    EXEC_EXIT_CODE=${PIPESTATUS[0]}  # Get exit code from eval, not tee
else
    eval $CMD
    EXEC_EXIT_CODE=$?
fi

set -e  # Re-enable exit on error

# Record execution end time
EXEC_END=$(date +%s)
EXEC_END_STR=$(get_timestamp)
EXEC_ELAPSED=$(get_elapsed $EXEC_START $EXEC_END)
EXEC_ELAPSED_STR=$(format_time $EXEC_ELAPSED)

log_timing ""
log_timing "Pipeline execution completed at: $EXEC_END_STR"
log_timing "Pipeline execution time: $EXEC_ELAPSED_STR ($EXEC_ELAPSED seconds)"
log_timing "Exit code: $EXEC_EXIT_CODE"

# Record total script time
SCRIPT_END=$(date +%s)
SCRIPT_END_STR=$(get_timestamp)
SCRIPT_ELAPSED=$(get_elapsed $SCRIPT_START $SCRIPT_END)
SCRIPT_ELAPSED_STR=$(format_time $SCRIPT_ELAPSED)

log_timing ""
log_timing "================================================================================"
log_timing "Total script execution time: $SCRIPT_ELAPSED_STR ($SCRIPT_ELAPSED seconds)"
log_timing "Script started:  $SCRIPT_START_STR"
log_timing "Script finished: $SCRIPT_END_STR"
log_timing "================================================================================"

echo_and_log ""
echo_and_log "================================================================================"
if [ $EXEC_EXIT_CODE -eq 0 ]; then
    echo_and_log "mDeepFRI completed successfully!"
else
    echo_and_log "mDeepFRI completed with exit code: $EXEC_EXIT_CODE"
fi
echo_and_log "Results saved to: $OUTPUT_DIR"
echo_and_log "Pipeline execution time: $EXEC_ELAPSED_STR"
echo_and_log "Total script time: $SCRIPT_ELAPSED_STR"
echo_and_log "Timing log saved to: $TIMING_LOG"
if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
    echo_and_log "Terminal output log saved to: $TERMINAL_LOG"
fi
echo_and_log "================================================================================"

# Exit with the same code as the mDeepFRI command
exit $EXEC_EXIT_CODE

