import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml

class Role(str, Enum):
    """What a column is semantically"""
    IDENTIFIER = "identifier"      # addresses / stream ids; never a feature
    NUMERIC = "numeric"            # continuous or count-valued
    BINARY = "binary"              # 0/1 indicator
    CATEGORICAL = "categorical"    # string-valued nominal
    LABEL = "label"                # target
    GROUP = "group"                # CV grouping key (capture id)


class SchemaError(ValueError):
    """Raised when a file does not match the schema it was loaded under."""


@dataclass(frozen=True)
class Schema:
    name: str
    identifiers: tuple[str, ...]
    leakage_suspects: tuple[str, ...]
    binary: tuple[str, ...]
    categorical: tuple[str, ...]
    numeric_sentinels: dict[str, list[float]]
    string_sentinels: tuple[str, ...]
    validity_flags: dict[str, str]
    groups: dict[str, list[str]]
    range_constraints: tuple[tuple[str, float, float], ...]
    range_exempt: tuple[str, ...]
    label_patterns: tuple[tuple[str, str], ...]
    facet_patterns: dict[str, list[tuple[str, str]]]
    benign_label: str = "benign"
    binary_anomalous: str = "anomalous"
    binary_col: str = ""
    label_col: str = "label"
    source_col: str = "capture_id"
    known_columns: tuple[str, ...] = field(default=())
    structural_missing: tuple[str, ...] = ()


    @property
    def meta_columns(self) -> tuple[str, ...]:
        cols = [self.label_col, self.source_col, *self.facet_patterns.keys()]
        if self.binary_col:
            cols.append(self.binary_col)
        return tuple(cols)

    @property
    def target_columns(self) -> tuple[str, ...]:
        cols = [self.label_col]
        if self.binary_col:
            cols.append(self.binary_col)
        return tuple(cols)

    @property
    def is_resolved(self) -> bool:
        """Schemas are constructed unresolved (known_columns=()). io.resolve_schema
        returns a resolved copy via dataclasses.replace -- construction lives
        outside this class; a Schema only answers questions."""
        return bool(self.known_columns)

    def role(self, column: str) -> Role:
            if column == self.label_col or column == self.binary_col:
                return Role.LABEL
            if column == self.source_col or column in self.facet_patterns:
                return Role.GROUP
            if column in self.identifiers:
                return Role.IDENTIFIER
            if column in self.categorical:
                return Role.CATEGORICAL
            if column in self.binary:
                return Role.BINARY
            return Role.NUMERIC



    # ---- get column properties ------------------------

    def is_target(self, column: str) -> bool:
        return column in self.target_columns
 
    def is_structural(self, column: str) -> bool:
        return column in self.structural_missing

    def sentinels_for(self, column: str) -> list[float]:
        return list(self.numeric_sentinels.get(column, []))

    def validity_flag_for(self, column: str) -> str | None:
        return self.validity_flags.get(column)

    def group_of(self, column: str) -> str | None:
        for name, cols in self.groups.items():
            if column in cols:
                return name
        return None

    def ungrouped_columns(self) -> list[str]:
        """Known columns that belong to no group. Informational only -- they
        are loaded, cleaned and ranked exactly like any other column."""
        grouped = {c for cols in self.groups.values() for c in cols}
        return [c for c in self.known_columns if c not in grouped]
 
    # --- parsing --------
    def label_from_filename(self, filename: str) -> str:
        low = filename.lower()
        for pattern, label in self.label_patterns:
            if re.search(pattern, low):
                return label
        raise SchemaError(
            f"no label pattern matched {filename!r}"
        )

    def facets_from_filename(self, filename: str) -> dict[str, str]:
        low = filename.lower()
        out: dict[str, str] = {}
        for facet, patterns in self.facet_patterns.items():
            for pattern, value in patterns:
                if re.search(pattern, low):
                    out[facet] = value
                    break
        return out


    def binary_from_label(self, label: str) -> str:
        return (self.benign_label if label == self.benign_label else self.binary_anomalous)

    
    #--------------validation---------------

    def self_check(self) -> None:
        """
        1. Checks if any feature is not part of group
        2. Checks if any feature is part of two groups
        """
        grouped = [c for cols in self.groups.values() for c in cols]
        dupes = {c for c in grouped if grouped.count(c) > 1}
        if dupes:
            raise SchemaError(f"{self.name}: column(s) in >1 group: {sorted(dupes)}")
        if self.binary_col:
            taken = {self.label_col, self.source_col, *self.facet_patterns}
            if self.binary_col in taken:
                raise SchemaError(
                    f"{self.name}: columns.binary_state {self.binary_col!r} collides "
                    f"with another generated column ({sorted(taken)})"
                )
            if self.is_resolved and self.binary_col in self.known_columns:
                raise SchemaError(
                    f"{self.name}: columns.binary_state {self.binary_col!r} is "
                    f"already a column in the data"
                )
            if self.benign_label == self.binary_anomalous:
                raise SchemaError(
                    f"{self.name}: labels.benign and labels.anomalous are both "
                    f"{self.benign_label!r} -- the binary column would be constant"
                )

    # -- construction from config -----------------------------------------
 
    @classmethod
    def from_dict(cls, cfg: dict, name: str | None = None) -> "Schema":
        """
        Build a Schema from a parsed config mapping.

        """
        allowed = {"name", "roles", "sentinels", "groups", "constraints", "missingness",
                   "labels", "columns"}
        unknown = set(cfg) - allowed
        if unknown:
            raise SchemaError(f"unknown section(s) in config: {sorted(unknown)}; "
                              f"expected any of {sorted(allowed)}")
 
        def section(key: str, keys: set[str]) -> dict:
            sec = cfg.get(key) or {}
            if not isinstance(sec, dict):
                raise SchemaError(f"section {key!r} must be a mapping")
            bad = set(sec) - keys
            if bad:
                raise SchemaError(f"unknown key(s) in {key!r}: {sorted(bad)}; "
                                  f"expected any of {sorted(keys)}")
            return sec
 
        roles = section("roles", {"identifiers", "leakage_suspects", "binary",
                                  "categorical"})
        sent = section("sentinels", {"numeric", "string",
                                     "validity_flags"})
        cons = section("constraints", {"ranges", "exempt"})
        labels = section("labels", {"benign", "anomalous", "patterns", "facets"})
        cols = section("columns", {"label", "source", "binary_state"})
        missing = section("missingness", {"structural"})
 
        def tup(d, k):
            return tuple(d.get(k) or ())
 
        ranges = tuple(
            (str(r[0]), float(r[1]), float(r[2]))
            for r in (cons.get("ranges") or [])
        )
        patterns = tuple((str(p[0]), str(p[1]))
                         for p in (labels.get("patterns") or []))
        facets = {
            facet: [(str(p[0]), str(p[1])) for p in plist]
            for facet, plist in (labels.get("facets") or {}).items()
        }
 
        schema = cls(
            name=name or cfg.get("name") or "unnamed",
            identifiers=tup(roles, "identifiers"),
            leakage_suspects=tup(roles, "leakage_suspects"),
            binary=tup(roles, "binary"),
            categorical=tup(roles, "categorical"),
            numeric_sentinels={k: [float(x) for x in v]
                               for k, v in (sent.get("numeric") or {}).items()},
            string_sentinels=tuple(sent.get("string") or ("none", "na", "n/a", "")),
            validity_flags=dict(sent.get("validity_flags") or {}),
            groups={k: list(v) for k, v in (cfg.get("groups") or {}).items()},
            range_constraints=ranges,
            range_exempt=tup(cons, "exempt"),
            label_patterns=patterns,
            facet_patterns=facets,
            benign_label=labels.get("benign", "benign"),
            binary_anomalous=labels.get("anomalous", "anomalous"),
            label_col=cols.get("label", "label"),
            binary_col=cols.get("binary_state", "") or "",
            source_col=cols.get("source", "capture_id"),
            structural_missing=missing.get("structural") or (),
        )
        schema.self_check() # checks if feature is in multiple groups, and if binary is fine
        return schema
 
    @classmethod
    def from_yaml(cls, path) -> "Schema":
        """Load a format description from a YAML config file."""
        path = Path(path)
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise SchemaError(f"{path}: invalid YAML -- {exc}") from exc
        if not isinstance(cfg, dict):
            raise SchemaError(f"{path}: top level must be a mapping")
        return cls.from_dict(cfg, name=cfg.get("name") or path.stem)
 

# A schema that claims nothing. It is the default everywhere, so a corpus loaded
# without an explicit config is typed conservatively rather than having some
# other format's policy applied to it by accident.
GENERIC_SCHEMA = Schema(
    name="generic",
    identifiers=(),
    leakage_suspects=(),
    binary=(),
    categorical=(),
    numeric_sentinels={},
    string_sentinels=("none", "na", "n/a", ""),
    validity_flags={},
    groups={},
    range_constraints=(),
    range_exempt=(),
    label_patterns=((r"benign|normal|clean", "benign"),),
    facet_patterns={},
)