from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from .config import PipelineConfig
from .biomed import AliasTable, BiomedicalConsensusValidator


SECTION_PATTERNS = {
    "abstract": re.compile(r"^\s*(abstract)\s*$", re.IGNORECASE),
    "introduction": re.compile(r"^\s*(introduction|background)\s*$", re.IGNORECASE),
    "methods": re.compile(
        r"^\s*(materials\s+and\s+methods|methods?|methodology)\s*$", re.IGNORECASE
    ),
    "results": re.compile(r"^\s*(results?)\s*$", re.IGNORECASE),
    "discussion": re.compile(r"^\s*(discussion|conclusion|conclusions)\s*$", re.IGNORECASE),
    "references": re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE),
    "acknowledgements": re.compile(r"^\s*(acknowledg(e)?ments?)\s*$", re.IGNORECASE),
}

GENE_CANDIDATE_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,14}\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])")
CITATION_RE = re.compile(r"\((?:[A-Z][A-Za-z\-]+(?:\set\sal\.)?,?\s*\d{4}[a-z]?(?:;\s*)?)+\)")
AFFILIATION_LINE_RE = re.compile(
    r"^\s*(\d+\s+)?([A-Za-z].*)?(university|department|institute|hospital|college)\b",
    re.IGNORECASE,
)


@dataclass
class SentenceRecord:
    section: str
    sentence: str


@dataclass
class CandidateEvidence:
    token: str
    canonical_token: str
    frequency: int
    context_score: float
    relation_hits: int
    section_diversity: int
    biomedical_score: float
    biomedical_sources: List[str]
    pattern_score: float
    ambiguity_penalty: float
    web_score: float
    final_score: float
    accepted: bool
    section_mentions: Dict[str, int]
    example_sentence: str


@dataclass
class PipelineResult:
    clean_text: str
    sections: Dict[str, str]
    sentence_count: int
    candidates_considered: int
    accepted_genes: List[CandidateEvidence]
    rejected_candidates: List[CandidateEvidence]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "clean_text_length": len(self.clean_text),
            "sections": {k: len(v) for k, v in self.sections.items()},
            "sentence_count": self.sentence_count,
            "candidates_considered": self.candidates_considered,
            "accepted_genes": [asdict(x) for x in self.accepted_genes],
            "rejected_candidates": [asdict(x) for x in self.rejected_candidates],
        }


class GeneExtractionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.alias_table = AliasTable(self.config.aliases_path)
        self.biomedical_validator = (
            BiomedicalConsensusValidator(
                scispacy_model=self.config.scispacy_model,
                biobert_model=self.config.biobert_model,
            )
            if self.config.use_biomedical_ensemble
            else None
        )
        self._web_cache: Dict[str, float] = {}
        self._load_web_cache()

    def run(self, pdf_path: Path) -> PipelineResult:
        raw_text = self._extract_pdf_text(pdf_path)
        clean_text = self._clean_text(raw_text)
        sections = self._segment_sections(clean_text)
        sentence_records = self._segment_sentences(sections)
        alias_map = self._build_alias_map(sentence_records)

        candidates = self._extract_candidates(sentence_records)
        candidates = self._canonicalize_candidates(candidates, alias_map)
        candidates = self._frequency_filter(candidates, sentence_records)

        accepted, rejected = self._validate_candidates(candidates, sentence_records)
        accepted_sorted = sorted(accepted, key=lambda x: x.final_score, reverse=True)
        rejected_sorted = sorted(rejected, key=lambda x: x.final_score, reverse=True)

        self._save_web_cache()

        return PipelineResult(
            clean_text=clean_text,
            sections=sections,
            sentence_count=len(sentence_records),
            candidates_considered=len(candidates),
            accepted_genes=accepted_sorted,
            rejected_candidates=rejected_sorted,
        )

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        text_chunks: List[str] = []

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                text_chunks.append(page.extract_text() or "")
        except Exception as exc:
            raise RuntimeError(
                "Failed to parse PDF using pypdf. Install dependencies and verify the file."
            ) from exc

        combined_text = "\n".join(text_chunks).strip()

        # OCR fallback for scanned PDFs with little or no extracted text.
        if len(combined_text) < 300:
            combined_text = self._ocr_pdf(pdf_path)

        return combined_text

    def _ocr_pdf(self, pdf_path: Path) -> str:
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except Exception as exc:
            raise RuntimeError(
                "OCR requested but pdf2image/pytesseract are unavailable."
            ) from exc

        images = convert_from_path(str(pdf_path))
        ocr_chunks = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(ocr_chunks)

    def _clean_text(self, text: str) -> str:
        lines = text.splitlines()
        cleaned_lines: List[str] = []
        cut_mode = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue

            if SECTION_PATTERNS["references"].match(stripped) or SECTION_PATTERNS[
                "acknowledgements"
            ].match(stripped):
                cut_mode = True

            if cut_mode:
                continue

            if AFFILIATION_LINE_RE.search(stripped) and len(stripped) < 140:
                continue

            without_citations = CITATION_RE.sub("", stripped)
            without_noise = re.sub(r"[^\w\s\-\.,;:()/%]", " ", without_citations)
            normalized = re.sub(r"\s+", " ", without_noise).strip()
            if normalized:
                cleaned_lines.append(normalized)

        cleaned_text = "\n".join(cleaned_lines)
        return re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    def _segment_sections(self, text: str) -> Dict[str, str]:
        sections = defaultdict(list)
        current_section = "other"

        for line in text.splitlines():
            header_hit = None
            for section_name, pattern in SECTION_PATTERNS.items():
                if pattern.match(line.strip()):
                    header_hit = section_name
                    break

            if header_hit:
                current_section = header_hit
                continue

            if current_section in {"references", "acknowledgements"}:
                continue

            sections[current_section].append(line)

        if not sections:
            return {"other": text}

        return {sec: "\n".join(lines).strip() for sec, lines in sections.items() if lines}

    def _segment_sentences(self, sections: Dict[str, str]) -> List[SentenceRecord]:
        sentence_records: List[SentenceRecord] = []

        for section, content in sections.items():
            if not content.strip():
                continue

            normalized = content.replace("\n", " ")
            pieces = SENTENCE_SPLIT_RE.split(normalized)
            for piece in pieces:
                sentence = piece.strip()
                if len(sentence) < 8:
                    continue
                sentence_records.append(SentenceRecord(section=section, sentence=sentence))

        return sentence_records

    def _extract_candidates(self, sentence_records: List[SentenceRecord]) -> Counter:
        candidate_counts: Counter = Counter()
        for rec in sentence_records:
            for token in GENE_CANDIDATE_RE.findall(rec.sentence):
                candidate_counts[token] += 1
        return candidate_counts

    def _build_alias_map(self, sentence_records: List[SentenceRecord]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        alias_pattern = re.compile(
            r"\b([A-Za-z][A-Za-z0-9\- ]{2,60}?)\s*\(([^()]{2,40})\)"
        )
        alias_stop_tokens = {
            "GENE",
            "GENES",
            "PROTEIN",
            "PROTEINS",
            "HUB",
            "MODULE",
            "DATA",
            "MODEL",
            "NETWORK",
            "PATHWAY",
        }

        for rec in sentence_records:
            for match in alias_pattern.finditer(rec.sentence):
                left = match.group(1).strip()
                right = match.group(2).strip()
                left_candidates = GENE_CANDIDATE_RE.findall(left.upper())
                right_candidates = GENE_CANDIDATE_RE.findall(right.upper())

                # Alias mapping should only happen for one-to-one mentions, not list declarations.
                if len(left_candidates) != 1 or len(right_candidates) != 1:
                    continue

                left_gene = next(
                    (
                        token
                        for token in left_candidates
                        if self._looks_gene_like(token) and token not in alias_stop_tokens
                    ),
                    None,
                )
                right_gene = next(
                    (
                        token
                        for token in right_candidates
                        if self._looks_gene_like(token) and token not in alias_stop_tokens
                    ),
                    None,
                )

                if left_gene and right_gene:
                    # Prefer canonical forms with more structural gene signal.
                    left_signal = int(any(ch.isdigit() for ch in left_gene)) + int("-" in left_gene)
                    right_signal = int(any(ch.isdigit() for ch in right_gene)) + int("-" in right_gene)

                    if left_signal >= right_signal:
                        alias_map[right_gene] = left_gene
                    else:
                        alias_map[left_gene] = right_gene

        return alias_map

    def _canonicalize_candidates(self, candidates: Counter, alias_map: Dict[str, str]) -> Counter:
        canonical_counts: Counter = Counter()
        for token, count in candidates.items():
            canonical = alias_map.get(token, token)
            canonical_counts[canonical] += count
        return canonical_counts

    def _frequency_filter(
        self, candidates: Counter, sentence_records: List[SentenceRecord]
    ) -> Counter:
        token_to_sentences: Dict[str, List[SentenceRecord]] = defaultdict(list)
        for rec in sentence_records:
            rec_tokens = set(GENE_CANDIDATE_RE.findall(rec.sentence))
            for token in rec_tokens:
                canonical = self._canonicalize_token(token)
                if canonical in candidates:
                    token_to_sentences[canonical].append(rec)

        filtered: Counter = Counter()
        for token, count in candidates.items():
            if count >= self.config.min_frequency:
                filtered[token] = count
                continue

            # Keep rare candidates if they appear in explicit gene-declaration contexts.
            if self._has_gene_declaration_mention(token, token_to_sentences.get(token, [])):
                filtered[token] = count

        return filtered

    def _has_gene_declaration_mention(
        self, token: str, token_sentences: Iterable[SentenceRecord]
    ) -> bool:
        token_lower = token.lower()
        for rec in token_sentences:
            sentence = rec.sentence.lower()
            if token_lower not in sentence:
                continue

            if re.search(rf"\bgenes?\b[^.]*\b{re.escape(token_lower)}\b", sentence):
                return True
            if re.search(rf"\b{re.escape(token_lower)}\b[^.]*\bgenes?\b", sentence):
                return True
            if re.search(
                rf"\b(hub\s+genes?|biomarkers?|differentially\s+expressed\s+genes?)\b[^.]*\b{re.escape(token_lower)}\b",
                sentence,
            ):
                return True

        return False

    def _validate_candidates(
        self, candidates: Counter, sentence_records: List[SentenceRecord]
    ) -> Tuple[List[CandidateEvidence], List[CandidateEvidence]]:
        token_to_sentences: Dict[str, List[SentenceRecord]] = defaultdict(list)
        high_value_sections = {"results", "discussion", "abstract"}
        available_high_value_sections = {
            rec.section for rec in sentence_records if rec.section in high_value_sections
        }
        effective_min_section_diversity = min(
            self.config.min_section_diversity,
            max(1, len(available_high_value_sections)),
        )

        for rec in sentence_records:
            rec_tokens = set(GENE_CANDIDATE_RE.findall(rec.sentence))
            for token in rec_tokens:
                canonical_token = self._canonicalize_token(token)
                if canonical_token in candidates:
                    token_to_sentences[canonical_token].append(rec)

        accepted: List[CandidateEvidence] = []
        rejected: List[CandidateEvidence] = []

        for token, freq in candidates.items():
            canonical_token = self._canonicalize_token(token)
            (
                context_score,
                relation_hits,
                section_mentions,
                example_sentence,
            ) = self._context_score(canonical_token, token_to_sentences[token])
            biomedical_score, biomedical_sources = self._biomedical_consensus_score(
                canonical_token, token_to_sentences[token]
            )
            pattern_score = self._pattern_score(canonical_token)
            ambiguity_penalty = self._ambiguity_penalty(canonical_token, context_score)
            web_score = (
                self._web_validation_score(canonical_token)
                if self.config.use_web_validation
                else 0.0
            )
            section_diversity = len([sec for sec, hits in section_mentions.items() if hits > 0])
            has_declaration_context = self._has_gene_declaration_mention(
                canonical_token, token_to_sentences[token]
            )
            effective_min_relation_hits = self.config.min_relation_hits
            if has_declaration_context:
                effective_min_relation_hits = min(effective_min_relation_hits, 1)

            final_score = (
                (2.0 if freq >= self.config.min_frequency else 0.0)
                + context_score
                + biomedical_score
                + pattern_score
                + web_score
                + ambiguity_penalty
                + min(section_diversity - 1, 2) * 0.8
            )

            if self.config.use_biomedical_ensemble:
                final_score += min(len(biomedical_sources), 2) * 0.5

            # Fast-fail non-gene detection
            is_nongene = self._is_likely_nongene(canonical_token, context_score)
            
            accepted_flag = (
                not is_nongene  # Hard fail on non-gene detection
                and context_score >= self.config.min_context_score
                and relation_hits >= effective_min_relation_hits
                and section_diversity >= effective_min_section_diversity
                and (not self.config.use_biomedical_ensemble or biomedical_score >= 1.0)
                and final_score >= self.config.min_final_score
                and pattern_score > 0
                and self._passes_section_presence(section_mentions)
            )

            evidence = CandidateEvidence(
                token=token,
                canonical_token=canonical_token,
                frequency=freq,
                context_score=round(context_score, 3),
                relation_hits=relation_hits,
                section_diversity=section_diversity,
                biomedical_score=round(biomedical_score, 3),
                biomedical_sources=biomedical_sources,
                pattern_score=round(pattern_score, 3),
                ambiguity_penalty=round(ambiguity_penalty, 3),
                web_score=round(web_score, 3),
                final_score=round(final_score, 3),
                accepted=accepted_flag,
                section_mentions=section_mentions,
                example_sentence=example_sentence,
            )

            if accepted_flag:
                accepted.append(evidence)
            else:
                rejected.append(evidence)

        return accepted, rejected

    def _biomedical_consensus_score(
        self, token: str, token_sentences: Iterable[SentenceRecord]
    ) -> Tuple[float, List[str]]:
        if self.biomedical_validator is None:
            return 0.0, []

        best_score = 0.0
        best_sources: List[str] = []
        token_lower = token.lower()

        for rec in token_sentences:
            if token_lower not in rec.sentence.lower():
                continue

            sentence_score, sources = self.biomedical_validator.score_sentence(rec.sentence)
            if sentence_score > best_score:
                best_score = sentence_score
                best_sources = sources

        return best_score, best_sources

    def _canonicalize_token(self, token: str) -> str:
        canonical = token.strip().upper()
        canonical = re.sub(r"[^A-Z0-9-]", "", canonical)
        canonical = self.alias_table.canonicalize(canonical)
        return canonical

    def _looks_gene_like(self, token: str) -> bool:
        if not token:
            return False
        if len(token) < self.config.min_token_len or len(token) > self.config.max_token_len:
            return False
        if not re.fullmatch(r"[A-Z][A-Z0-9-]*", token):
            return False
        return True

    def _context_score(
        self, token: str, token_sentences: Iterable[SentenceRecord]
    ) -> Tuple[float, int, Dict[str, int], str]:
        score = 0.0
        relation_hits = 0
        section_mentions: Dict[str, int] = defaultdict(int)
        best_sentence = ""
        best_sentence_score = -1.0

        for rec in token_sentences:
            sentence_lower = rec.sentence.lower()
            section_weight = self.config.section_weights.get(
                rec.section, self.config.section_weights["other"]
            )

            keyword_hits = sum(1 for kw in self.config.context_keywords if kw in sentence_lower)
            action_hits = sum(1 for kw in self.config.action_keywords if kw in sentence_lower)
            negative_hits = sum(
                1 for kw in self.config.negative_context_keywords if kw in sentence_lower
            )

            relation_match_hits = self._relation_pattern_hits(token, sentence_lower)
            relation_hits += relation_match_hits

            local_score = (
                keyword_hits * 1.0 + action_hits * 1.5 + relation_match_hits * 2.0
            ) * section_weight

            if negative_hits and keyword_hits == 0 and action_hits == 0:
                local_score -= min(negative_hits, 2) * 1.2 * section_weight

            if token in rec.sentence and re.search(rf"\b{re.escape(token)}\b", rec.sentence):
                local_score += 0.5

            score += local_score
            section_mentions[rec.section] += 1

            if local_score > best_sentence_score:
                best_sentence_score = local_score
                best_sentence = rec.sentence

        # Temporal consistency bonus across high-value sections.
        if section_mentions.get("results", 0) > 0 and section_mentions.get("discussion", 0) > 0:
            score += 1.2

        return score, relation_hits, dict(section_mentions), best_sentence

    def _relation_pattern_hits(self, token: str, sentence_lower: str) -> int:
        token_lower = token.lower()
        patterns = [
            rf"\b{re.escape(token_lower)}\s+(gene|protein)\b",
            rf"(expression|mutation|knockdown|overexpression)\s+of\s+{re.escape(token_lower)}\b",
            rf"\b{re.escape(token_lower)}\s+(regulates|activates|inhibits|suppresses|encodes|binds)\b",
            rf"\b{re.escape(token_lower)}-?(mediated|dependent)\b",
            rf"\bgenes?\b[^.]*\b{re.escape(token_lower)}\b",
            rf"\b(hub\s+genes?|biomarkers?|differentially\s+expressed\s+genes?)\b[^.]*\b{re.escape(token_lower)}\b",
        ]
        return sum(1 for pattern in patterns if re.search(pattern, sentence_lower))

    def _passes_section_presence(self, section_mentions: Dict[str, int]) -> bool:
        high_value_hits = (
            section_mentions.get("results", 0)
            + section_mentions.get("discussion", 0)
            + section_mentions.get("abstract", 0)
        )
        # Allow from "other" section if no high-value sections exist
        if high_value_hits > 0:
            return True
        # Fallback: accept if any section mentions exist (for poorly structured PDFs)
        return len(section_mentions) > 0

    def _pattern_score(self, token: str) -> float:
        if len(token) < self.config.min_token_len or len(token) > self.config.max_token_len:
            return -2.0

        if not token.isupper():
            return -1.0

        has_digit = any(ch.isdigit() for ch in token)
        has_hyphen = "-" in token

        score = 1.0
        if has_digit:
            score += 0.7
        if has_hyphen:
            score += 0.2

        # Penalize tokens that are mostly digits or malformed.
        alpha_count = sum(1 for ch in token if ch.isalpha())
        digit_count = sum(1 for ch in token if ch.isdigit())
        if alpha_count == 0 or digit_count > alpha_count * 2:
            score -= 1.5

        return score

    def _ambiguity_penalty(self, token: str, context_score: float) -> float:
        # Hard block: tokens in explicit non-gene blacklist
        if token in self.config.non_gene_blacklist:
            return -10.0  # Unrecoverable penalty
        
        if token in self.config.ambiguous_tokens:
            # Permit rare recovery if context is very strong.
            return -1.0 if context_score >= 8.0 else -3.0

        if token in {"DNA", "RNA"}:
            return -2.0

        return 0.0

    def _is_likely_nongene(self, token: str, context_score: float) -> bool:
        """Detect if token is likely NOT a gene based on heuristics."""
        # Hard block: explicit blacklist
        if token in self.config.non_gene_blacklist:
            return True
        
        # Database name heuristic: all-caps multi-letter acronyms ending in common patterns
        if self._is_database_acronym(token):
            return True
        
        # Repository/tool name heuristic: specific naming patterns
        if self._is_software_name(token):
            return True
        
        # Clinical score heuristic: numbers after capital letter(s)
        if self._looks_like_clinical_score(token):
            return True
        
        # Parameter/file format heuristic
        if self._is_parameter_or_format(token):
            return True
        
        return False

    def _is_database_acronym(self, token: str) -> bool:
        """Detect database/repository name patterns (e.g., EMBL-EBI, NCBI, PDB)."""
        if len(token) < 2 or len(token) > 12:
            return False
        
        # All caps with hyphens in structured patterns (e.g., EMBL-EBI)
        if re.match(r"^[A-Z]{2,}(-[A-Z]{2,})?$", token):
            # If it's a well-known database abbreviation
            if token in {"NCBI", "PDB", "KEGG", "OMIM", "UNIPROT", "EMBL", "ENSEMBL", 
                        "UCSC", "GENBANK", "PUBMED", "EBI"}:
                return True
            # Generic patterns for databases: 3-4 letters only, mostly consonants
            if 2 <= len(token) <= 4 and token.count('A') + token.count('E') + token.count('I') + token.count('O') + token.count('U') <= 1:
                return True
        
        return False

    def _is_software_name(self, token: str) -> bool:
        """Detect software/tool names (e.g., SAM Tools, GATK, PLINK)."""
        # Known tools
        if token in {"BLAST", "BLAT", "BWA", "GATK", "PLINK", "SAMTOOLS", "BEDTOOLS", 
                     "VCFTOOLS", "STAR", "BOWTIE", "TOPHAT", "CUFFLINKS"}:
            return True
        
        # Pattern: repeated double-letter (rare in genes): KKKK, LLLL, etc. often in tools
        if re.search(r"([A-Z])\1{2,}", token):
            if token not in {"TTTTT"}:  # TTTTT could be a weird gene, but unlikely
                return True
        
        return False

    def _looks_like_clinical_score(self, token: str) -> bool:
        """Detect clinical scores (e.g., PASI, APACHE, SOFA)."""
        # Clinical score heuristic: typically 4-5 caps, often acronyms
        clinical_terms = {
            "PASI",    # Psoriasis Area and Severity Index
            "APACHE",  # Acute Physiology and Chronic Health Evaluation
            "SOFA",    # Sequential Organ Failure Assessment
            "NEWS",    # National Early Warning Score
            "CURB",    # Confusion, Urea, Respiratory rate, Blood pressure
            "SAPS",    # Simplified Acute Physiology Score
            "QSOFA",   # Quick SOFA
        }
        
        if token in clinical_terms:
            return True
        
        # Pattern: 4-5 letter acronym ending in common endings
        if 4 <= len(token) <= 5 and token[-1] in {"I", "C"}:
            consonant_ratio = sum(1 for c in token if c not in "AEIOU") / len(token)
            if consonant_ratio >= 0.6:  # >60% consonants suggests acronym, not gene
                return True
        
        return False

    def _is_parameter_or_format(self, token: str) -> bool:
        """Detect parameter names and file formats."""
        formats_and_params = {
            "BAM", "VCF", "SAM", "FASTA", "FASTQ", "GFF", "BED", "MAF",  # File formats
            "NODE", "EDGE", "VERTEX",  # Graph terms
            "ROC", "AUC", "KNN", "SVM", "PCA", "LDA",  # ML terms
            "PVALUE", "ZSCORE", "FDR", "QVALUE",  # Stats
        }
        
        if token in formats_and_params:
            return True
        
        return False

    def _web_validation_score(self, token: str) -> float:
        if token in self._web_cache:
            return self._web_cache[token]

        sources_found = 0
        confidence = 0.0

        wiki_score = self._check_wikipedia(token)
        if wiki_score > 0:
            sources_found += 1
            confidence += wiki_score

        ncbi_score = self._check_ncbi_gene_search(token)
        if ncbi_score > 0:
            sources_found += 1
            confidence += ncbi_score

        # Require at least two supporting sources for strong trust.
        if sources_found >= 2:
            confidence += 1.5

        self._web_cache[token] = confidence
        return confidence

    def _check_wikipedia(self, token: str) -> float:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{token} gene protein",
            "format": "json",
            "utf8": 1,
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config.web_validation_timeout_seconds,
                headers={"User-Agent": "gene-extractor/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return 0.0

        snippets = [hit.get("snippet", "") for hit in payload.get("query", {}).get("search", [])[:3]]
        if not snippets:
            return 0.0

        text = " ".join(snippets).lower()
        keyword_hits = sum(1 for kw in ["gene", "protein", "dna", "expression"] if kw in text)
        if keyword_hits >= 2:
            return 2.0
        if keyword_hits == 1:
            return 1.0
        return 0.0

    def _check_ncbi_gene_search(self, token: str) -> float:
        url = "https://www.ncbi.nlm.nih.gov/gene/"
        params = {"term": token}

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config.web_validation_timeout_seconds,
                headers={"User-Agent": "gene-extractor/1.0"},
            )
            response.raise_for_status()
            text = response.text.lower()
        except Exception:
            return 0.0

        hit_count = sum(1 for kw in ["gene", "protein", "expression", "homo sapiens"] if kw in text)
        if hit_count >= 3:
            return 2.0
        if hit_count >= 1:
            return 1.0
        return 0.0

    def _load_web_cache(self) -> None:
        path = self.config.web_cache_path
        if not path.exists():
            self._web_cache = {}
            return

        try:
            self._web_cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._web_cache = {}

    def _save_web_cache(self) -> None:
        path = self.config.web_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._web_cache, indent=2), encoding="utf-8")
