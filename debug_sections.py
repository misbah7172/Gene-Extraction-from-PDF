from pathlib import Path
from gene_extractor.pipeline import GeneExtractionPipeline
from gene_extractor.config import PipelineConfig

cfg = PipelineConfig()
p = GeneExtractionPipeline(cfg)

# Extract sections without running full pipeline
pdf_path = Path('nature11412.pdf')
text = p._extract_pdf_text(pdf_path)
clean = p._clean_text(text)
sections = p._segment_sections(clean)

print("Sections found:")
for sec, content in sections.items():
    print(f"  {sec}: {len(content)} chars")
