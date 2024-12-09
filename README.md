# Assignment 05 - ATMT 24 fall

## Running the Experiments

### 1. Beam Search Experiments
```bash
# Standard beam search with varying sizes
bash beam_size_run_constant.sh

# Generate visualization of results
python BLEU_BP_plot_generate.py
```

### 2. Advanced Beam Search
```bash
# Constant beam size search
bash beam_size_run_constant.sh

# Beam search with pruning
bash beam_size_run_constant_prune.sh
```

## Output Files
Each experiment generates the following files:
- `translations.txt` - Raw model outputs
- `translations.p.txt` - Post-processed translations
- `bleu_results.txt` - BLEU score evaluation results
