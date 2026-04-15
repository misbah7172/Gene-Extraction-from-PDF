import argparse
import csv
import json
from pathlib import Path

from gene_extractor import GeneExtractionPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-precision gene extractor from research PDFs."
    )
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/gene_extraction_result.json"),
        help="Path to JSON output",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/final_genes.csv"),
        help="Path to CSV output with accepted genes",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum candidate frequency",
    )
    parser.add_argument(
        "--min-context-score",
        type=float,
        default=3.0,
        help="Minimum context score",
    )
    parser.add_argument(
        "--min-final-score",
        type=float,
        default=6.0,
        help="Minimum final score",
    )
    parser.add_argument(
        "--web-validate",
        action="store_true",
        help="Enable optional web consensus validation",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable highest-precision preset with stricter thresholds and web validation",
    )
    parser.add_argument(
        "--bio-ensemble",
        action="store_true",
        help="Enable SciSpacy/BioBERT-style consensus validation without strict thresholds",
    )
    parser.add_argument(
        "--scispacy-model",
        type=str,
        default="en_core_sci_sm",
        help="spaCy/SciSpacy model name to use for dependency parsing",
    )
    parser.add_argument(
        "--biobert-model",
        type=str,
        default="d4data/biomedical-ner-all",
        help="Hugging Face biomedical NER model name to use for token classification",
    )
    return parser


def write_csv(path: Path, accepted_genes: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "token",
                "frequency",
                "context_score",
                "pattern_score",
                "web_score",
                "ambiguity_penalty",
                "final_score",
                "example_sentence",
            ]
        )

        for gene in accepted_genes:
            writer.writerow(
                [
                    gene.token,
                    gene.frequency,
                    gene.context_score,
                    gene.pattern_score,
                    gene.web_score,
                    gene.ambiguity_penalty,
                    gene.final_score,
                    gene.example_sentence,
                ]
            )


def main() -> None:
    args = build_parser().parse_args()

    use_web_validation = args.web_validate or args.strict
    use_biomedical_ensemble = args.bio_ensemble or args.strict
    cfg = PipelineConfig(
        min_frequency=2 if args.strict else args.min_frequency,
        min_context_score=5.0 if args.strict else args.min_context_score,
        min_final_score=10.0 if args.strict else args.min_final_score,
        min_relation_hits=1 if args.strict else 1,
        min_section_diversity=2 if args.strict else 1,
        strict_mode=args.strict,
        use_biomedical_ensemble=use_biomedical_ensemble,
        use_web_validation=use_web_validation,
        scispacy_model=args.scispacy_model,
        biobert_model=args.biobert_model,
    )

    pipeline = GeneExtractionPipeline(cfg)
    result = pipeline.run(args.pdf)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result.to_json_dict(), indent=2), encoding="utf-8"
    )

    write_csv(args.output_csv, result.accepted_genes)

    print(f"Accepted genes: {len(result.accepted_genes)}")
    print(f"Rejected candidates: {len(result.rejected_candidates)}")
    print(f"JSON output: {args.output_json}")
    print(f"CSV output: {args.output_csv}")


if __name__ == "__main__":
    main()
