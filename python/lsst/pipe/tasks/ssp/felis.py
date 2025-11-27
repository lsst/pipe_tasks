import numpy as np
from typing import Mapping, Any
from textwrap import wrap

import argparse
import sys
import yaml

# Default tables to process if none specified
DEFAULT_TABLES = ["SSObject", "SSSource", "mpc_orbits", "current_identifications", "numbered_identifications"]


# ----------------------------------------------------------------------
# Helper: timestamp precision → numpy time unit
# ----------------------------------------------------------------------
def _timestamp_precision_to_unit(prec: int) -> str:
    """
    Map a Felis timestamp precision to a numpy datetime64 unit.

    prec = number of decimal places of seconds to retain.
    """
    if prec <= 0:
        return "s"
    elif prec <= 2:
        return "ms"
    elif prec <= 5:
        return "us"
    else:
        return "ns"  # max precision numpy supports


# ----------------------------------------------------------------------
# Column → NumPy dtype
# ----------------------------------------------------------------------
def _felis_column_to_numpy_dtype(col: Mapping[str, Any]) -> tuple[str, Any]:
    name = col["name"]

    dt = col.get("datatype")
    if dt is None:
        raise ValueError(f"Column {name!r} has no datatype")

    dt = dt.lower()

    # ---------- numeric ------------------------------------------------------
    if dt == "int8":
        return name, np.int8
    if dt in ("int16", "short"):
        return name, np.int16
    if dt in ("int32", "int"):
        return name, np.int32
    if dt in ("int64", "long", "bigint"):
        return name, np.int64

    if dt == "uint8":
        return name, np.uint8
    if dt == "uint16":
        return name, np.uint16
    if dt == "uint32":
        return name, np.uint32
    if dt == "uint64":
        return name, np.uint64

    if dt in ("float32", "float"):
        return name, np.float32
    if dt in ("float64", "double"):
        return name, np.float64

    if dt in ("bool", "boolean"):
        return name, np.bool_

    # ---------- timestamps ---------------------------------------------------
    if dt == "timestamp":
        prec = col.get("precision", 0)
        if not isinstance(prec, int):
            raise ValueError(f"Timestamp field {name!r} has non-integer precision")
        unit = _timestamp_precision_to_unit(prec)
        return name, np.dtype(f"datetime64[{unit}]")

    # ---------- fixed-size binary via length? --------------------------------
    # If you want to support `datatype: binary` later, we can add that here.

    # ---------- strings ------------------------------------------------------
    if dt in ("string", "unicode", "str", "char"):
        L = col.get("length")
        if isinstance(L, int):
            return name, np.dtype(f"U{L}")
        return name, np.dtype("U")

    # ---------- lists / arrays / unknown → object ----------------------------
    return name, object


# ----------------------------------------------------------------------
# Table → NumPy dtype with metadata
# ----------------------------------------------------------------------
def felis_table_to_numpy_dtype(table: Mapping[str, Any]) -> np.dtype:
    """
    Convert a Felis table definition (YAML → dict) into a NumPy dtype.
    Metadata stored:

        dtype.metadata["description"] = table description
        dtype.metadata["columns"]     = {name: "[unit] description"}
    """
    cols = table.get("columns")
    if cols is None:
        raise ValueError("Table definition has no 'columns' key")

    # Field dtypes
    fields = [_felis_column_to_numpy_dtype(c) for c in cols]

    # Table-level description
    table_desc = table.get("description")

    # Column metadata with optional unit prepended
    colmeta = {}
    for c in cols:
        name = c["name"]
        desc = c.get("description")
        unit = c.get("ivoa:unit")

        if unit is not None:
            if desc:
                full = f"[{unit}] {desc}"
            else:
                full = f"[{unit}]"
        else:
            full = desc

        if full is not None:
            colmeta[name] = full

    metadata = {}
    if table_desc is not None:
        metadata["description"] = table_desc
    if colmeta:
        metadata["columns"] = colmeta

    if metadata:
        return np.dtype(fields, metadata=metadata)
    return np.dtype(fields)


def pretty_print_dtype(
    dtype: np.dtype,
    table_name: str,
    target_comment_col: int = 36,
    max_line_length: int = 110,
) -> str:
    """
    Pretty-print a structured NumPy dtype (with Felis-derived metadata)
    as valid, readable Python code:

        # Wrapped table description...
        #
        <table_name>Dtype = np.dtype([
            ('field1', '<i8'),         # comment...
            ('very_long_field', ...    # comment juts, next long field aligned
                                       # to same jutter column...
        ])

    Parameters
    ----------
    dtype : np.dtype
        Structured dtype with metadata fields:
            metadata["description"]  : table-level description (optional)
            metadata["columns"]      : {col_name: per-column description}
    table_name : str
        Name used for the assignment, e.g. <table_name>Dtype.
    target_comment_col : int, default=36
        Preferred starting column for comments when the field fits before it.
        If the field text is longer than this, a "juttering group" alignment
        logic kicks in to prevent jagged right edges.
    max_line_length : int, default=110
        Maximum line length for wrapping table descriptions and comments.
        Lines will be wrapped to be strictly less than this length.

    Behavior
    --------
    * Table description is wrapped to < max_line_length chars, placed above
      dtype assignment, followed by a blank line.
    * Field comments:
        - If the field length <= target_comment_col - 1 → comment starts at
          target_comment_col, and the "juttering group" resets.
        - If the field length >= target_comment_col → the comment "juts out".
            + First such field sets the group's jutter column.
            + Next juttering fields use max(previous_jut_col, natural_jut_col).
            + This avoids jaggedness.
    * Final lines never exceed max_line_length - 1 chars.
    * Per-field comments wrap into at most 2 lines, with "..." if needed.
    * dtype.metadata is NOT emitted; only used for comments.

    Returns
    -------
    str
        Pretty Python code string.
    """
    if not isinstance(dtype, np.dtype) or not dtype.fields:
        raise TypeError("Expected a structured numpy.dtype with fields")

    md = dtype.metadata or {}
    table_desc = md.get("description")
    col_descs = md.get("columns", {})

    lines: list[str] = []

    # ---- Table description: wrap to < max_line_length chars -----------
    if table_desc:
        txt = f"{table_name}: {table_desc}"
        for w in wrap(txt, width=max_line_length - 3):
            lines.append(f"# {w}")

    # ---- Begin dtype assignment ---------------------------------------------
    lines.append(f"{table_name}Dtype = np.dtype([")

    # Build base field specs
    field_entries = []
    for name, (ftype, _) in dtype.fields.items():
        base = f"    ({name!r}, {ftype.str!r}),"
        comment = col_descs.get(name)
        if comment:
            comment = " ".join(str(comment).split())  # normalize whitespace
        field_entries.append((base, comment))

    last_jut_comment_col = None  # track juttering group alignment

    # ---- Process each field with smoothed jutter alignment ------------------
    for base, comment in field_entries:
        base_len = len(base)

        if not comment:
            lines.append(base)
            # Reset jutter group if this field doesn't jut
            if base_len <= target_comment_col - 1:
                last_jut_comment_col = None
            continue

        # Determine comment_col for this field
        if base_len <= target_comment_col - 1:
            # field fits → align to target column, reset jutter group
            comment_col = target_comment_col
            last_jut_comment_col = None
        else:
            # field juts
            natural_col = base_len + 1  # one space after field

            if last_jut_comment_col is None:
                # first jutter field
                comment_col = natural_col
                last_jut_comment_col = comment_col
            else:
                # subsequent jutter fields (apply smoothing)
                if natural_col <= last_jut_comment_col:
                    # would jut less; align with previous jutter column
                    comment_col = last_jut_comment_col
                else:
                    # juts further; update group
                    comment_col = natural_col
                    last_jut_comment_col = comment_col

        # Compute max allowed comment length for < max_line_length total chars
        max_comment_width = max(10, (max_line_length - 1) - (comment_col + 2))  # 2 for "# "

        # Wrap comment into segments
        words = comment.split()
        segments = []
        cur = ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_comment_width:
                cur += " " + w
            else:
                segments.append(cur)
                cur = w
        if cur:
            segments.append(cur)

        # Ellipsize to 2 lines max
        if len(segments) > 2:
            segments = segments[:2]
            if len(segments[-1]) + 3 > max_comment_width:
                segments[-1] = segments[-1][: max_comment_width - 3].rstrip()
            segments[-1] += "..."

        # Emit first line
        pad = " " * (comment_col - base_len)
        lines.append(f"{base}{pad}# {segments[0]}")

        # Continuations
        cont_prefix = " " * comment_col + "# "
        for seg in segments[1:]:
            lines.append(f"{cont_prefix}{seg}")

    lines.append("])")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate NumPy dtypes from Felis YAML schema")
    parser.add_argument("felis_yaml_file", help="Path to the YAML schema file")
    parser.add_argument(
        "table_names",
        nargs="*",
        default=DEFAULT_TABLES,
        help=(f"Names of tables to process (default: {', '.join(DEFAULT_TABLES)})"),
    )
    args = parser.parse_args()

    with open(args.felis_yaml_file) as fp:
        schema = yaml.safe_load(fp)

    table_schemas = {t["name"]: t for t in schema["tables"]}

    # Print header
    print("# ***** GENERATED FILE, DO NOT EDIT BY HAND *****")
    print("# ruff: noqa: W505")
    print(f"# generated with {' '.join(sys.argv)} # noqa: E501")
    print()
    print("import numpy as np")
    print()

    for i, table in enumerate(args.table_names):
        dtype = felis_table_to_numpy_dtype(table_schemas[table])
        print(pretty_print_dtype(dtype, table))
        if i < len(args.table_names) - 1:
            print()


if __name__ == "__main__":
    main()
