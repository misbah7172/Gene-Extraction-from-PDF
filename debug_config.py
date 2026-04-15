from pathlib import Path
from gene_extractor import PipelineConfig
from gene_extractor.pipeline import GeneExtractionPipeline

# Check what config was used
cfg = PipelineConfig(strict_mode=True, use_biomedical_ensemble=True, use_web_validation=True)
print("Config values:")
print(f"  min_section_diversity: {cfg.min_section_diversity}")
print(f"  use_biomedical_ensemble: {cfg.use_biomedical_ensemble}")
print(f"  use_web_validation: {cfg.use_web_validation}")
print(f"  min_final_score: {cfg.min_final_score}")
print(f"  min_context_score: {cfg.min_context_score}")

# Now check what main.py sets
import sys
sys.path.insert(0, '/d:/CODE/GENE')
from main import build_config_from_args
# Mock args
class Args:
    strict = True
    bio_ensemble = False
    scispacy_model = None
    biobert_model = None
    min_context_score = None
    min_final_score = None
    min_frequency = None

args = Args()
cfg2 = build_config_from_args(args)
print("\nStrict mode config:")
print(f"  min_section_diversity: {cfg2.min_section_diversity}")
print(f"  min_final_score: {cfg2.min_final_score}")
print(f"  min_context_score: {cfg2.min_context_score}")
