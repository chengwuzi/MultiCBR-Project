Diffusion Top-K=3 mechanism analysis outputs
================================================

Representative user id: 4830

Generated files:
- Adjacent epoch overlap line data: module1_ablation\analysis\diffusion_topk3\adjacent_epoch_overlap.csv
- Representative user candidates: module1_ablation\analysis\diffusion_topk3\representative_user_candidates.csv
- Representative user heatmap rank matrix: module1_ablation\analysis\diffusion_topk3\representative_user_top3_rank_matrix.csv
- Representative user heatmap long table: module1_ablation\analysis\diffusion_topk3\representative_user_top3_rank_long.csv
- Representative user heatmap label matrix: module1_ablation\analysis\diffusion_topk3\representative_user_top3_label_matrix.csv

Chart 1:
Use adjacent_epoch_overlap.csv. X-axis can be epoch or epoch_pair; Y-axis is avg_jaccard_overlap.

Chart 2:
Use representative_user_top3_rank_matrix.csv for a 30x3 heatmap. Lower pretrained_rank means higher pretrained user-bundle dot-product score.
Use representative_user_top3_rank_long.csv if the chart tool supports cell labels or richer annotations.