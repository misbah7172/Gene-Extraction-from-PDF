from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BIO_ACTION_WORDS = {
    "activate",
    "activates",
    "activation",
    "bind",
    "binds",
    "binding",
    "encode",
    "encodes",
    "expression",
    "inhibit",
    "inhibits",
    "knockdown",
    "mutation",
    "overexpression",
    "pathway",
    "protein",
    "regulate",
    "regulates",
    "regulation",
    "signaling",
    "transcription",
}

NEGATIVE_NEIGHBOR_WORDS = {
    "algorithm",
    "benchmark",
    "dataset",
    "figure",
    "implementation",
    "parameter",
    "pipeline",
    "software",
    "table",
}


@dataclass
class BioEvidence:
    section: str
    sentence: str
    score: float
    source: str


class AliasTable:
    def __init__(self, aliases_path: Optional[Path] = None):
        self.aliases_path = aliases_path or Path(__file__).with_name("data").joinpath("gene_aliases.json")
        self.alias_to_canonical: Dict[str, str] = {}
        self.canonical_to_aliases: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.aliases_path.exists():
            self.alias_to_canonical = {}
            self.canonical_to_aliases = {}
            return

        payload = json.loads(self.aliases_path.read_text(encoding="utf-8"))
        alias_to_canonical: Dict[str, str] = {}
        canonical_to_aliases: Dict[str, List[str]] = {}

        for canonical, aliases in payload.items():
            canonical_norm = self._normalize(canonical)
            canonical_to_aliases[canonical_norm] = [self._normalize(alias) for alias in aliases]
            alias_to_canonical[canonical_norm] = canonical_norm
            for alias in aliases:
                alias_to_canonical[self._normalize(alias)] = canonical_norm

        self.alias_to_canonical = alias_to_canonical
        self.canonical_to_aliases = canonical_to_aliases

    def canonicalize(self, token: str) -> str:
        normalized = self._normalize(token)
        return self.alias_to_canonical.get(normalized, normalized)

    def _normalize(self, token: str) -> str:
        token = token.strip().upper()
        token = re.sub(r"[^A-Z0-9-]", "", token)
        return token


class BiomedicalConsensusValidator:
    def __init__(
        self,
        scispacy_model: str = "en_core_sci_sm",
        biobert_model: str = "d4data/biomedical-ner-all",
        enable_transformers: bool = True,
    ):
        self.scispacy_model = scispacy_model
        self.biobert_model = biobert_model
        self.enable_transformers = enable_transformers
        self._spacy_nlp = self._load_spacy_model()
        self._ner_pipeline = self._load_ner_pipeline() if enable_transformers else None

    def _load_spacy_model(self):
        try:
            import spacy
        except Exception:
            return None

        model_candidates = [self.scispacy_model, "en_core_sci_sm", "en_core_web_sm"]
        for model_name in model_candidates:
            try:
                return spacy.load(model_name)
            except Exception:
                continue
        return None

    def _load_ner_pipeline(self):
        try:
            from transformers import pipeline
        except Exception:
            return None

        try:
            return pipeline(
                "token-classification",
                model=self.biobert_model,
                aggregation_strategy="simple",
            )
        except Exception:
            return None

    def score_sentence(self, sentence: str) -> Tuple[float, List[str]]:
        scores: List[float] = []
        evidence_sources: List[str] = []

        spacy_score = self._score_with_spacy(sentence)
        if spacy_score > 0:
            scores.append(spacy_score)
            evidence_sources.append("scispacy")

        biobert_score = self._score_with_biobert(sentence)
        if biobert_score > 0:
            scores.append(biobert_score)
            evidence_sources.append("biobert")

        if not scores:
            return 0.0, evidence_sources

        consensus_boost = 0.8 if len(scores) >= 2 else 0.0
        return sum(scores) / len(scores) + consensus_boost, evidence_sources

    def token_supported(self, token: str, sentence: str) -> bool:
        sentence_score, evidence_sources = self.score_sentence(sentence)
        token_lower = token.lower()
        return sentence_score >= 1.0 and any(token_lower in source_sentence.lower() for source_sentence in [sentence]) and bool(evidence_sources)

    def _score_with_spacy(self, sentence: str) -> float:
        if self._spacy_nlp is None:
            return 0.0

        doc = self._spacy_nlp(sentence)
        if not doc:
            return 0.0

        ent_texts = {ent.text.lower() for ent in getattr(doc, "ents", [])}
        token_texts = {token.text.lower() for token in doc}
        bio_word_hits = sum(1 for word in BIO_ACTION_WORDS if word in sentence.lower())
        neg_hits = sum(1 for word in NEGATIVE_NEIGHBOR_WORDS if word in sentence.lower())

        score = 0.0
        if ent_texts:
            score += 0.8
        if bio_word_hits >= 2:
            score += 1.0
        elif bio_word_hits == 1:
            score += 0.6
        if any(word in token_texts for word in {"gene", "protein", "mrna", "dna", "rna"}):
            score += 0.6
        if neg_hits and bio_word_hits == 0:
            score -= 0.8

        dep_hits = 0
        for token in doc:
            lemma = token.lemma_.lower() if token.lemma_ else token.text.lower()
            if lemma in {"regulate", "activate", "inhibit", "encode", "bind", "associate"}:
                if token.head is not None and token.head != token:
                    dep_hits += 1
                if any(child.dep_ in {"nsubj", "dobj", "pobj", "nsubjpass"} for child in token.children):
                    dep_hits += 1

        if dep_hits >= 2:
            score += 1.2
        elif dep_hits == 1:
            score += 0.7

        return max(score, 0.0)

    def _score_with_biobert(self, sentence: str) -> float:
        if self._ner_pipeline is None:
            return 0.0

        try:
            entities = self._ner_pipeline(sentence)
        except Exception:
            return 0.0

        if not entities:
            return 0.0

        entity_labels = [entity.get("entity_group", entity.get("entity", "")) for entity in entities]
        entity_texts = [str(entity.get("word", "")).lower() for entity in entities]
        bio_word_hits = sum(1 for word in BIO_ACTION_WORDS if word in sentence.lower())

        score = 0.0
        if entity_labels:
            score += 0.7
        if any(label.upper().startswith(("GENE", "PRO", "CHEM", "DIS", "DNA", "RNA")) for label in entity_labels):
            score += 1.0
        if any(word in sentence.lower() for word in ["gene", "protein", "expression", "mutation"]):
            score += 0.7
        if bio_word_hits >= 2:
            score += 0.6
        if entity_texts and len({text for text in entity_texts if text}) >= 2:
            score += 0.2

        return min(score, 2.0)
