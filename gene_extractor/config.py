from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set


@dataclass
class PipelineConfig:
    min_frequency: int = 2
    min_token_len: int = 3
    max_token_len: int = 10
    min_context_score: float = 3.0
    min_final_score: float = 6.0
    min_relation_hits: int = 1
    min_section_diversity: int = 1
    strict_mode: bool = False
    use_biomedical_ensemble: bool = False
    use_web_validation: bool = False
    web_validation_timeout_seconds: int = 12
    web_cache_path: Path = Path(".cache/web_validation_cache.json")
    aliases_path: Path = Path("gene_extractor/data/gene_aliases.json")
    scispacy_model: str = "en_core_sci_sm"
    biobert_model: str = "d4data/biomedical-ner-all"

    section_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "results": 1.7,
            "discussion": 1.6,
            "abstract": 1.4,
            "introduction": 1.0,
            "methods": 0.9,
            "other": 1.0,
        }
    )

    context_keywords: Set[str] = field(
        default_factory=lambda: {
            "gene",
            "protein",
            "expression",
            "mutation",
            "pathway",
            "regulation",
            "transcription",
            "oncogene",
            "tumor",
            "dna",
            "rna",
            "cell",
            "knockdown",
            "overexpression",
            "signaling",
        }
    )

    action_keywords: Set[str] = field(
        default_factory=lambda: {
            "regulates",
            "inhibits",
            "activates",
            "suppresses",
            "induces",
            "binds",
            "encodes",
            "interacts",
            "upregulated",
            "downregulated",
            "mutated",
        }
    )

    negative_context_keywords: Set[str] = field(
        default_factory=lambda: {
            "algorithm",
            "software",
            "dataset",
            "benchmark",
            "parameter",
            "pipeline",
            "table",
            "figure",
            "supplementary",
            "appendix",
            "buffer",
            "kit",
        }
    )

    ambiguous_tokens: Set[str] = field(
        default_factory=lambda: {
            "MAP",
            "SET",
            "CAT",
            "ACT",
            "RUN",
            "TOP",
            "MIN",
            "MAX",
            "SUM",
            "AGE",
            "WAS",
            "ARE",
            "THE",
            "AND",
            "FOR",
            "NOT",
            "YES",
        }
    )
