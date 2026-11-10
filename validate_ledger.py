#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference timing ledger - schema and sanity validator (schema v1.0).

Validates CSV timing ledgers produced with the ``inference_timing_ledger.csv``
template. See README.md for the full column-by-column schema.

Usage:
    python3 validate_ledger.py <ledger.csv> [--strict] [--json]

Exit codes:
    0   ledger is valid (warnings are allowed unless --strict is given)
    1   validation failed
    2   the file could not be read or parsed

The validator is dependency-free and runs on Python 3.8+ (standard library only),
so it can be used locally or in CI without any installation step.

Author:      Jyotirmoy Saikia <jyotirmoysaikia05153@gmail.com>
Attribution: "Jyotirmoy Saikia: inference timing ledger"
License:     MIT
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"

# Schema definition

COLUMNS = [
    "schema_version", "run_id", "recorded_at_utc", "recorded_by", "contact_email",
    "model_id", "model_revision",
    "device_type", "device_name", "device_count", "device_memory_total_gb",
    "precision", "quantization_method",
    "input_len_tokens", "output_len_tokens", "batch_size",
    "warmup_runs", "timed_runs", "sync_method",
    "latency_ms_min", "latency_ms_p50", "latency_ms_mean", "latency_ms_p90",
    "latency_ms_p95", "latency_ms_p99", "latency_ms_max", "latency_ms_stddev",
    "throughput_output_tokens_per_sec", "peak_memory_mb",
    "inference_engine", "python_version", "torch_version", "transformers_version",
    "cuda_version", "os_version", "notes",
]

OPTIONAL_COLUMNS = {
    "notes", "cuda_version", "torch_version", "transformers_version",
    "peak_memory_mb", "device_memory_total_gb",
}

ENUMS: Dict[str, set] = {
    "device_type": {"cpu", "gpu", "mps", "npu", "tpu", "other"},
    "precision": {"fp32", "fp16", "bf16", "fp8", "int8", "int4"},
    "quantization_method": {"none", "bitsandbytes", "gptq", "awq", "other"},
    "sync_method": {"device_sync", "cuda_events", "host_timer_only", "not_applicable"},
}

INT_COLUMNS = {
    "device_count", "input_len_tokens", "output_len_tokens",
    "batch_size", "warmup_runs", "timed_runs",
}

LATENCY_CHAIN = [
    "latency_ms_min", "latency_ms_p50", "latency_ms_p90",
    "latency_ms_p95", "latency_ms_p99", "latency_ms_max",
]

FLOAT_COLUMNS = {
    "latency_ms_min", "latency_ms_p50", "latency_ms_mean", "latency_ms_p90",
    "latency_ms_p95", "latency_ms_p99", "latency_ms_max", "latency_ms_stddev",
    "throughput_output_tokens_per_sec", "peak_memory_mb", "device_memory_total_gb",
}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

# column -> (predicate, human-readable constraint)
_GT_ZERO: Tuple[Callable[[float], bool], str] = (lambda v: v > 0, "> 0")
_GE_ZERO: Tuple[Callable[[float], bool], str] = (lambda v: v >= 0, ">= 0")
_GE_ONE: Tuple[Callable[[float], bool], str] = (lambda v: v >= 1, ">= 1")

NUMERIC_RULES: Dict[str, Tuple[Callable[[float], bool], str]] = {
    "device_count": _GE_ONE,
    "input_len_tokens": _GE_ONE,
    "output_len_tokens": _GE_ZERO,
    "batch_size": _GE_ONE,
    "warmup_runs": _GE_ZERO,
    "timed_runs": _GE_ONE,
    "latency_ms_stddev": _GE_ZERO,
    "throughput_output_tokens_per_sec": _GE_ZERO,
    "peak_memory_mb": _GT_ZERO,
    "device_memory_total_gb": _GT_ZERO,
}
for _col in LATENCY_CHAIN + ["latency_ms_mean"]:
    NUMERIC_RULES[_col] = _GT_ZERO

FILE_FATAL_CODES = {"E_UNREADABLE", "E_EMPTY_FILE", "E_CSV_PARSE"}

# Findings

@dataclass
class Finding:
    severity: str           
    code: str
    message: str
    row: Optional[int] = None   
    run_id: Optional[str] = None
    column: Optional[str] = None

# Validator

class LedgerValidator:
    def __init__(self) -> None:
        self.findings: List[Finding] = []
        self.rows_checked = 0
        self._seen_run_ids: Dict[str, int] = {}

    # -- finding helpers ---------------------------------------------------

    def error(self, code: str, message: str, row: Optional[int] = None,
              run_id: Optional[str] = None, column: Optional[str] = None) -> None:
        self.findings.append(Finding("error", code, message, row, run_id, column))

    def warning(self, code: str, message: str, row: Optional[int] = None,
                run_id: Optional[str] = None, column: Optional[str] = None) -> None:
        self.findings.append(Finding("warning", code, message, row, run_id, column))

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    # -- file / header -----------------------------------------------------

    def validate_path(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                try:
                    header = next(reader)
                    raw_rows = list(reader)
                except csv.Error as exc:
                    self.error("E_CSV_PARSE", f"could not parse CSV: {exc}")
                    return
        except OSError as exc:
            self.error("E_UNREADABLE", f"could not read file: {exc}")
            return

        if not header or all(h.strip() == "" for h in header):
            self.error("E_EMPTY_FILE", "file is empty (no header row)")
            return

        self._validate_header([h.strip() for h in header])
        if self.errors:
            return  # header problems make row checks meaningless

        for row_num, raw in enumerate(raw_rows, start=1):
            if not raw or all(cell.strip() == "" for cell in raw):
                continue  # blank line / fully empty row
            self.rows_checked += 1
            self._validate_row(row_num, raw, header)

    def _validate_header(self, header: List[str]) -> None:
        duplicates = sorted({h for h in header if header.count(h) > 1})
        if duplicates:
            self.error("E_DUPLICATE_COLUMN",
                       f"duplicate column name(s): {', '.join(duplicates)}")
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            self.error("E_MISSING_COLUMN",
                       f"missing required column(s): {', '.join(missing)}")
        unknown = [h for h in header if h not in COLUMNS]
        if unknown:
            self.error("E_UNKNOWN_COLUMN",
                       f"unknown column(s): {', '.join(unknown)}")

    # -- rows ---------------------------------------------------------------

    def _validate_row(self, row_num: int, raw: List[str], header: List[str]) -> None:
        if len(raw) != len(header):
            self.error("E_FIELD_COUNT",
                       f"row has {len(raw)} fields, expected {len(header)}",
                       row=row_num)
            return

        values: Dict[str, str] = {}
        for col, cell in zip(header, raw):
            if cell != cell.strip():
                self.warning("W_WHITESPACE",
                             f"'{col}' has surrounding whitespace (stripped before validation)",
                             row=row_num, column=col)
            values[col] = cell.strip()

        run_id = values.get("run_id", "") or None

        # required columns are non-empty
        for col in COLUMNS:
            if col in OPTIONAL_COLUMNS:
                continue
            if values.get(col, "") == "":
                self.error("E_EMPTY_REQUIRED", f"required column '{col}' is empty",
                           row=row_num, run_id=run_id, column=col)

        # schema version
        sv = values.get("schema_version", "")
        if sv and sv != SCHEMA_VERSION:
            self.error("E_SCHEMA_VERSION",
                       f"schema_version must be '{SCHEMA_VERSION}', found {sv!r}",
                       row=row_num, run_id=run_id, column="schema_version")

        # run id
        rid = values.get("run_id", "")
        if rid:
            if any(ch.isspace() for ch in rid):
                self.error("E_RUN_ID_WHITESPACE", "run_id contains whitespace",
                           row=row_num, run_id=run_id, column="run_id")
            elif not RUN_ID_RE.match(rid):
                self.warning("W_RUN_ID_PATTERN",
                             "run_id does not match the recommended pattern "
                             "'[A-Za-z0-9][A-Za-z0-9._-]*'",
                             row=row_num, run_id=run_id, column="run_id")
            if rid in self._seen_run_ids:
                self.error("E_DUPLICATE_RUN_ID",
                           f"run_id {rid!r} already used on row {self._seen_run_ids[rid]}",
                           row=row_num, run_id=run_id, column="run_id")
            else:
                self._seen_run_ids[rid] = row_num

        # enums
        for col, allowed in ENUMS.items():
            v = values.get(col, "")
            if v and v not in allowed:
                self.error("E_BAD_ENUM",
                           f"'{v}' is not an allowed value for {col} "
                           f"(allowed: {', '.join(sorted(allowed))})",
                           row=row_num, run_id=run_id, column=col)

        # timestamp
        ts = values.get("recorded_at_utc", "")
        if ts:
            if not TIMESTAMP_RE.match(ts):
                self.error("E_BAD_TIMESTAMP",
                           f"expected ISO-8601 UTC 'YYYY-MM-DDTHH:MM:SSZ', found {ts!r}",
                           row=row_num, run_id=run_id, column="recorded_at_utc")
            elif not self._parses_as_timestamp(ts):
                self.error("E_BAD_TIMESTAMP", f"{ts!r} is not a valid date/time",
                           row=row_num, run_id=run_id, column="recorded_at_utc")

        # light format warnings
        email = values.get("contact_email", "")
        if email and not EMAIL_RE.match(email):
            self.warning("W_EMAIL_FORMAT",
                         f"contact_email {email!r} does not look like an email address",
                         row=row_num, run_id=run_id, column="contact_email")
        model_id = values.get("model_id", "")
        if model_id and "/" not in model_id:
            self.warning("W_MODEL_ID_FORMAT",
                         f"model_id {model_id!r} usually looks like 'org/name'",
                         row=row_num, run_id=run_id, column="model_id")

        # numeric parsing
        parsed: Dict[str, float] = {}
        for col in INT_COLUMNS:
            v = values.get(col, "")
            if v == "":
                continue
            n = self._to_int(v, row_num, run_id, col)
            if n is not None:
                parsed[col] = n
        for col in FLOAT_COLUMNS:
            v = values.get(col, "")
            if v == "":
                continue
            n = self._to_float(v, row_num, run_id, col)
            if n is not None:
                parsed[col] = n

        # numeric ranges
        for col, (ok, human) in NUMERIC_RULES.items():
            if col in parsed and not ok(parsed[col]):
                self.error("E_RANGE",
                           f"{col} is {parsed[col]} but must be {human}",
                           row=row_num, run_id=run_id, column=col)

        # latency distribution
        if all(c in parsed for c in LATENCY_CHAIN):
            for a, b in zip(LATENCY_CHAIN, LATENCY_CHAIN[1:]):
                if parsed[a] > parsed[b]:
                    self.error("E_LATENCY_ORDER",
                               f"{a} ({parsed[a]}) must be <= {b} ({parsed[b]})",
                               row=row_num, run_id=run_id, column=b)
            mean = parsed.get("latency_ms_mean")
            if mean is not None and not (
                    parsed["latency_ms_min"] <= mean <= parsed["latency_ms_max"]):
                self.error("E_MEAN_OUT_OF_RANGE",
                           f"latency_ms_mean ({mean}) must be within "
                           f"[latency_ms_min, latency_ms_max] = "
                           f"[{parsed['latency_ms_min']}, {parsed['latency_ms_max']}]",
                           row=row_num, run_id=run_id, column="latency_ms_mean")

        # distribution sanity
        timed = parsed.get("timed_runs")
        sd = parsed.get("latency_ms_stddev")
        if timed is not None:
            if timed < 10:
                self.warning("W_FEW_TIMED_RUNS",
                             f"only {int(timed)} timed run(s); >= 10 recommended or "
                             "the tail percentiles are unreliable",
                             row=row_num, run_id=run_id, column="timed_runs")
            if sd is not None:
                if timed >= 2 and sd == 0:
                    self.warning("W_ZERO_STDDEV",
                                 f"latency_ms_stddev is 0 across {int(timed)} timed runs; "
                                 "check timer resolution / that iterations are actually re-timed",
                                 row=row_num, run_id=run_id, column="latency_ms_stddev")
                if timed == 1 and sd > 0:
                    self.warning("W_SPREAD_WITH_SINGLE_RUN",
                                 "latency_ms_stddev > 0 but timed_runs is 1; "
                                 "a single sample has no spread",
                                 row=row_num, run_id=run_id, column="latency_ms_stddev")

        # device-conditional checks
        dev = values.get("device_type", "")
        sync = values.get("sync_method", "")
        if dev in ENUMS["device_type"] and sync in ENUMS["sync_method"]:
            if dev == "gpu" and sync in ("host_timer_only", "not_applicable"):
                self.warning("W_SYNC_MISMATCH",
                             f"sync_method '{sync}' on a gpu: host-side timing under-reports "
                             "asynchronous kernels; use 'device_sync' or 'cuda_events'",
                             row=row_num, run_id=run_id, column="sync_method")
            if dev == "cpu" and sync in ("device_sync", "cuda_events"):
                self.warning("W_SYNC_MISMATCH",
                             f"sync_method '{sync}' has no effect on a cpu; "
                             "use 'not_applicable'",
                             row=row_num, run_id=run_id, column="sync_method")
            if dev == "mps" and sync == "cuda_events":
                self.warning("W_SYNC_MISMATCH",
                             "cuda_events requires a cuda gpu; use 'device_sync' "
                             "(torch.mps.synchronize) on mps",
                             row=row_num, run_id=run_id, column="sync_method")
            if dev in ("gpu", "mps"):
                if parsed.get("warmup_runs") == 0:
                    self.warning("W_NO_WARMUP",
                                 "warmup_runs is 0; first-iteration effects (lazy init, "
                                 "kernel autotuning) will pollute the distribution",
                                 row=row_num, run_id=run_id, column="warmup_runs")
                if values.get("peak_memory_mb", "") == "":
                    self.warning("W_NO_PEAK_MEMORY",
                                 "peak_memory_mb is empty; expected for gpu/mps runs "
                                 "(e.g. torch.cuda.max_memory_allocated()/1024**2 after warmup)",
                                 row=row_num, run_id=run_id, column="peak_memory_mb")
            if dev == "gpu" and values.get("cuda_version", "") == "":
                self.warning("W_CUDA_MISSING",
                             "cuda_version is empty for a gpu run",
                             row=row_num, run_id=run_id, column="cuda_version")
            if dev != "gpu" and values.get("cuda_version", "") != "":
                self.warning("W_CUDA_UNEXPECTED",
                             f"cuda_version is set but device_type is '{dev}'",
                             row=row_num, run_id=run_id, column="cuda_version")

        # quantization cross-check
        prec = values.get("precision", "")
        quant = values.get("quantization_method", "")
        if prec in ("int8", "int4", "fp8") and quant == "none":
            self.warning("W_QUANT_METHOD",
                         f"precision '{prec}' usually implies a quantization_method "
                         "other than 'none'",
                         row=row_num, run_id=run_id, column="quantization_method")

        # throughput consistency
        if all(k in parsed for k in ("batch_size", "output_len_tokens",
                                     "throughput_output_tokens_per_sec",
                                     "latency_ms_mean")):
            out = parsed["output_len_tokens"]
            thr = parsed["throughput_output_tokens_per_sec"]
            mean = parsed["latency_ms_mean"]
            if out == 0:
                if thr != 0:
                    self.error("E_THROUGHPUT_WITHOUT_OUTPUT",
                               f"throughput is {thr} but output_len_tokens is 0; "
                               "throughput must be 0 when no tokens are generated",
                               row=row_num, run_id=run_id,
                               column="throughput_output_tokens_per_sec")
            elif thr == 0:
                self.warning("W_THROUGHPUT_MISSING",
                             "throughput is 0 but output_len_tokens > 0",
                             row=row_num, run_id=run_id,
                             column="throughput_output_tokens_per_sec")
            elif mean > 0:
                expected = parsed["batch_size"] * out * 1000.0 / mean
                if abs(thr - expected) / expected > 0.05:
                    self.warning("W_THROUGHPUT_MISMATCH",
                                 f"throughput is {thr} but "
                                 "batch_size x output_len_tokens / latency_ms_mean "
                                 f"recomputes to {expected:.1f} (more than 5% off)",
                                 row=row_num, run_id=run_id,
                                 column="throughput_output_tokens_per_sec")

        # memory checks
        peak = parsed.get("peak_memory_mb")
        total = parsed.get("device_memory_total_gb")
        if values.get("device_memory_total_gb", "") == "":
            self.warning("W_MEMORY_TOTAL_MISSING",
                         "device_memory_total_gb is empty; recommended so that "
                         "peak_memory_mb can be interpreted",
                         row=row_num, run_id=run_id, column="device_memory_total_gb")
        if peak is not None and total is not None and peak > total * 1024.0:
            self.error("E_MEMORY_EXCEEDS_TOTAL",
                       f"peak_memory_mb ({peak} MB) exceeds device_memory_total_gb "
                       f"({total} GB = {total * 1024.0:.0f} MB)",
                       row=row_num, run_id=run_id, column="peak_memory_mb")

        # recommended software columns
        for col in ("torch_version", "transformers_version"):
            if values.get(col, "") == "":
                self.warning("W_RECOMMENDED_EMPTY",
                             f"'{col}' is empty; recommended for reproducibility "
                             "(leave empty only when not applicable)",
                             row=row_num, run_id=run_id, column=col)

    # -- parsing helpers ----------------------------------------------------

    @staticmethod
    def _parses_as_timestamp(value: str) -> bool:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False

    def _to_int(self, v: str, row: int, run_id: Optional[str], col: str) -> Optional[int]:
        if "_" in v:
            self.error("E_BAD_INT", f"'{col}': expected an integer, found {v!r}",
                       row=row, run_id=run_id, column=col)
            return None
        try:
            return int(v)
        except ValueError:
            self.error("E_BAD_INT", f"'{col}': expected an integer, found {v!r}",
                       row=row, run_id=run_id, column=col)
            return None

    def _to_float(self, v: str, row: int, run_id: Optional[str], col: str) -> Optional[float]:
        if "_" in v:
            self.error("E_BAD_FLOAT", f"'{col}': expected a number, found {v!r}",
                       row=row, run_id=run_id, column=col)
            return None
        try:
            n = float(v)
        except ValueError:
            self.error("E_BAD_FLOAT", f"'{col}': expected a number, found {v!r}",
                       row=row, run_id=run_id, column=col)
            return None
        if not math.isfinite(n):
            self.error("E_BAD_FLOAT", f"'{col}': {v!r} is not a finite number",
                       row=row, run_id=run_id, column=col)
            return None
        return n

# Reporting / CLI

def sorted_findings(findings: List[Finding]) -> List[Finding]:
    return sorted(findings, key=lambda f: (f.row if f.row is not None else 0,
                                           f.column or "", f.code))


def build_report(path: Path, validator: LedgerValidator, strict: bool) -> dict:
    has_file_fatal = any(f.code in FILE_FATAL_CODES for f in validator.errors)
    valid = (not validator.errors) and (not strict or not validator.warnings)
    return {
        "file": str(path),
        "schema_version": SCHEMA_VERSION,
        "rows_checked": validator.rows_checked,
        "valid": valid,
        "error_count": len(validator.errors),
        "warning_count": len(validator.warnings),
        "findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "row": f.row,
                "run_id": f.run_id,
                "column": f.column,
                "message": f.message,
            }
            for f in sorted_findings(validator.findings)
        ],
        "file_unreadable": has_file_fatal,
    }


def print_human(report: dict, validator: LedgerValidator, strict: bool) -> None:
    print(f"inference timing ledger validator (schema v{SCHEMA_VERSION})")
    print(f"file: {report['file']}")
    print(f"data rows checked: {report['rows_checked']}")
    print()
    if not validator.findings:
        print("no issues found")
    for f in sorted_findings(validator.findings):
        marker = "ERROR" if f.severity == "error" else "WARN "
        where = f"file" if f.row is None else f"row {f.row}"
        if f.run_id:
            where += f" ({f.run_id})"
        col = f".{f.column}" if f.column else ""
        print(f"{marker}  {where}{col}: {f.message}  [{f.code}]")
    print()
    n_err = report["error_count"]
    n_warn = report["warning_count"]
    if report["file_unreadable"]:
        verdict = "FAILED - file could not be read/parsed"
    elif not report["valid"] and n_err:
        verdict = "INVALID"
    elif not report["valid"] and strict:
        verdict = f"INVALID (--strict; {n_warn} warning(s) treated as errors)"
    elif n_warn:
        verdict = f"VALID (with warnings)"
    else:
        verdict = "VALID"
    print(f"result: {verdict} - {n_err} error(s), {n_warn} warning(s)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_ledger.py",
        description="Validate an inference timing ledger CSV against schema v%s." % SCHEMA_VERSION,
        epilog="examples: python3 validate_ledger.py inference_timing_ledger.csv\n"
               "          python3 validate_ledger.py inference_timing_ledger.csv --strict --json",
    )
    parser.add_argument("ledger", type=Path, help="path to the ledger CSV to validate")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures (recommended in CI)")
    parser.add_argument("--json", action="store_true",
                        help="print a machine-readable JSON report instead of text")
    args = parser.parse_args(argv)

    validator = LedgerValidator()
    if not args.ledger.exists():
        validator.error("E_UNREADABLE", f"file not found: {args.ledger}")
    else:
        validator.validate_path(args.ledger)

    report = build_report(args.ledger, validator, args.strict)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report, validator, args.strict)

    if report["file_unreadable"]:
        return 2
    if not report["valid"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
