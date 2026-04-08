"""Write alignment-carved structure fragments as PDB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import biotite.structure as struc
import numpy as np

from mDeepFRI.alignment import AlignmentResult
from mDeepFRI.bio_utils import load_structure

logger = logging.getLogger(__name__)

AA_1_TO_3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    "U": "SEC", "O": "PYL", "X": "UNK", "*": "UNK",
}


def extract_pdb_preamble(pdb_text: str) -> str:
    lines: List[str] = []
    for line in pdb_text.splitlines():
        rec = line[:6].strip().upper()
        if rec in ("ATOM", "HETATM", "MODEL", "ENDMDL"):
            break
        lines.append(line)
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _iter_matched_columns(
        gapped_query: str, gapped_target: str) -> Iterator[Tuple[int, str]]:
    target_idx = 0
    for q, t in zip(gapped_query, gapped_target):
        if q == "-":
            if t != "-":
                target_idx += 1
        else:
            if t != "-":
                yield target_idx, q.upper()
                target_idx += 1


def _remark_lines(alignment: AlignmentResult, template_label: str) -> List[str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return [
        "REMARK 999 METAGENOMIC-DEEPFRI CARVED STRUCTURE FRAGMENT",
        "REMARK 999 TEMPLATE RESIDUES MAPPED TO QUERY VIA PYOPAL ALIGNMENT.",
        "REMARK 999 QUERY INSERTIONS (RELATIVE TO TEMPLATE) ARE OMITTED.",
        f"REMARK 999 QUERY_ID: {alignment.query_name}",
        f"REMARK 999 TARGET_ID: {alignment.target_name}",
        f"REMARK 999 TEMPLATE: {template_label}",
        f"REMARK 999 TEMPLATE_CHAIN: {alignment.template_chain or 'A'}",
        f"REMARK 999 EXPORT_TIME_UTC: {now}",
    ]


def _atom_line(serial: int, atom_name: str, res_name: str, chain_id: str,
               res_seq: int, x: float, y: float, z: float,
               occupancy: float, b_factor: float, element: str) -> str:
    # PDB cols 13-16 atom name, 17 altLoc, 18-20 resName, 21 blank, 22 chain,
    # 23-26 resSeq, 27-30 blank before coords.
    name = atom_name.strip()[:4]
    if len(name) == 1:
        name = f" {name}  "
    elif len(name) == 2:
        name = f" {name} "
    elif len(name) == 3:
        name = f" {name}"
    else:
        name = name[:4].ljust(4)
    res = (res_name.strip()[:3] + "   ")[:3]
    altloc = " "
    elem = (element or "").strip()[:2]
    if not elem:
        elem = name.strip()[:1] or "C"
    return (
        f"ATOM  {serial:5d} {name:4s}{altloc}{res:>3s} {chain_id:1s}"
        f"{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          "
        f"{elem:>2s}  \n"
    )


def _array_length(arr: struc.AtomArray) -> int:
    fn = getattr(arr, "array_length", None)
    if callable(fn):
        return int(fn())
    return len(arr)


def _write_atom_array_records(atom_array: struc.AtomArray) -> str:
    lines: List[str] = []
    serial = 1
    n = _array_length(atom_array)
    cats = atom_array.get_annotation_categories()
    occ_ann = (atom_array.get_annotation("occupancy")
               if "occupancy" in cats else None)
    bfac_ann = (atom_array.get_annotation("b_factor")
                if "b_factor" in cats else None)
    for i in range(n):
        x, y, z = atom_array.coord[i]
        occ = float(occ_ann[i]) if occ_ann is not None else 1.0
        bfac = float(bfac_ann[i]) if bfac_ann is not None else 0.0
        aname = str(atom_array.atom_name[i]).strip()
        rname = str(atom_array.res_name[i]).strip()
        cid = str(atom_array.chain_id[i]).strip()[:1] or "A"
        rid = int(atom_array.res_id[i])
        elem = str(atom_array.element[i]).strip() if atom_array.element[i] else ""
        lines.append(
            _atom_line(serial, aname, rname, cid, rid, x, y, z, occ, bfac, elem))
        serial += 1
    return "".join(lines)


def _write_ca_only(alignment: AlignmentResult, path: Path, preamble: str,
                   remarks: List[str]) -> None:
    coords = alignment.coords
    if coords is None:
        logger.warning("Cannot export carved PDB for %s: no CA coordinates.",
                       alignment.query_name)
        return
    parts: List[str] = []
    if preamble:
        parts.append(preamble)
    for line in remarks:
        parts.append(line + "\n")
    parts.append(
        "REMARK 999 CA-ONLY EXPORT (FULL TEMPLATE NOT AVAILABLE).\n")
    serial = 1
    out_res = 1
    for target_idx, q_aa in _iter_matched_columns(alignment.gapped_sequence,
                                                  alignment.gapped_target):
        if target_idx >= len(coords):
            logger.warning("Carved CA export: target_idx out of range for %s.",
                           alignment.query_name)
            break
        x, y, z = coords[target_idx]
        res3 = AA_1_TO_3.get(q_aa, "UNK")
        parts.append(
            _atom_line(serial, "CA", res3, "A", out_res, float(x), float(y),
                       float(z), 1.0, 0.0, "C"))
        serial += 1
        out_res += 1
    parts.append("END\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def write_carved_structure_pdb(alignment: AlignmentResult, path: Path) -> None:
    raw: Optional[str] = None
    filetype = "pdb"
    preamble = ""
    template_label = alignment.target_name

    if alignment.template_path is not None:
        raw = alignment.template_path.read_text(encoding="utf-8",
                                                errors="replace")
        template_label = str(alignment.template_path)
        if alignment.template_filetype:
            filetype = alignment.template_filetype
        elif alignment.template_path.suffix.lower() in (".cif", ".mmcif"):
            filetype = "mmcif"
        else:
            filetype = "pdb"
        if filetype == "pdb":
            preamble = extract_pdb_preamble(raw)
    elif alignment.template_structure_string:
        raw = alignment.template_structure_string
        template_label = alignment.target_name
        filetype = alignment.template_filetype or "pdb"
        if filetype == "pdb":
            preamble = extract_pdb_preamble(raw)

    remarks = _remark_lines(alignment, template_label)
    chain = alignment.template_chain or "A"

    if raw is None or not chain:
        _write_ca_only(alignment, path, preamble, remarks)
        return

    try:
        struct = load_structure(
            raw, filetype="mmcif" if filetype == "mmcif" else "pdb")
        protein = struct[struct.chain_id == chain]
        poly = protein[protein.hetero == False]  # noqa: E712
        ca_atoms = poly[poly.atom_name == "CA"]
        if len(ca_atoms) != len(alignment.target_sequence):
            logger.warning(
                "CA count mismatch for %s (%d CA vs target len %d); using CA "
                "fallback.", alignment.query_name, len(ca_atoms),
                len(alignment.target_sequence))
            _write_ca_only(alignment, path, preamble, remarks)
            return

        blocks: List[struc.AtomArray] = []
        out_res = 1
        for target_idx, q_aa in _iter_matched_columns(
                alignment.gapped_sequence, alignment.gapped_target):
            rid = ca_atoms.res_id[target_idx]
            ins = ca_atoms.ins_code[target_idx]
            cid = ca_atoms.chain_id[target_idx]
            mask = ((poly.chain_id == cid) & (poly.res_id == rid)
                    & (poly.ins_code == ins))
            block = poly[mask].copy()
            if _array_length(block) == 0:
                logger.warning("Empty residue block at target_idx %d for %s.",
                               target_idx, alignment.query_name)
                _write_ca_only(alignment, path, preamble, remarks)
                return
            res3 = AA_1_TO_3.get(q_aa, "UNK")
            block.res_name[:] = res3
            block.res_id[:] = out_res
            nb = _array_length(block)
            block.chain_id = np.array(["A"] * nb, dtype=block.chain_id.dtype)
            blocks.append(block)
            out_res += 1

        if not blocks:
            _write_ca_only(alignment, path, preamble, remarks)
            return

        merged = struc.concatenate(blocks)
        parts: List[str] = []
        if preamble:
            parts.append(preamble)
        for line in remarks:
            parts.append(line + "\n")
        parts.append(_write_atom_array_records(merged))
        parts.append("END\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(parts), encoding="utf-8")
    except Exception as exc:
        logger.warning(
            "Full-atom carved export failed for %s (%s); falling back to CA.",
            alignment.query_name, exc)
        _write_ca_only(alignment, path, preamble, remarks)
