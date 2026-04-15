import json

data = json.load(open('output/gene_extraction_result.json'))
rejected = data['rejected_candidates']
accepted = data['accepted_genes']

print(f"Accepted: {len(accepted)}")
print(f"Rejected: {len(rejected)}")
print("\nTop 15 rejected candidates (sorted by final_score):")
for g in sorted(rejected, key=lambda x: x['final_score'], reverse=True)[:15]:
    print(f"  {g['token']}: score={g['final_score']}, context={g['context_score']}, biomedical={g['biomedical_score']}, pattern={g['pattern_score']}, rel_hits={g['relation_hits']}, section_div={g['section_diversity']}")

if accepted:
    print("\nAccepted genes:")
    for g in accepted:
        print(f"  {g['token']}: score={g['final_score']}")
else:
    print("\n⚠️  NO GENES ACCEPTED - Checking why:")
    # Check a real gene
    pik3ca = next((g for g in rejected if g['token'] == 'PIK3CA'), None)
    if pik3ca:
        print(f"\nPIK3CA analysis: {pik3ca}")

