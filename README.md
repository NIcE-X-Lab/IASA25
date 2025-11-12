# IASA25

## Pause Type Detection

This module contains training and evaluation code for pause-type detection models (classification, regression, and two-stage variants).

### Directory structure

```
gt_label/                 # manual pause-type annotations per audio clip (see note below)
pause_type_detection/
  data/                    # dataloaders and metadata helpers
  models/                  # model architectures and losses
  preprocessing/           # feature extraction & label conversion utilities
  postprocessing/          # event-level metrics utilities
  run_experiment/          # training loops for single-stage models
  scripts/                 # entry points that parse config and run experiments
  utils/                   # generic utilities (e.g., random seed)
```

Note: The corresponding audio files are available in the repository [data_after_cardio](https://github.com/Columbia-ICSL/data_after_cardio). Files in `gt_label` are manual annotations of pause types for each audio clip, where: `s` = semantic pause, `b` = breathing pause, `bs` = breathing and semantic pause, `o` = other (e.g., reading/talking), `p` = plosives, `l` = laughing. In our paper, we only use `s`, `b`, and `bs`, and map all other frames to `o`.

### Citation

If you use this code, please cite our paper:

- DOI: [https://doi.org/10.1145/3737901.3768369](https://doi.org/10.1145/3737901.3768369)

BibTeX: you can obtain the official BibTeX from the DOI using a service like [doi2bib](https://doi2bib.org/bib/10.1145/3737901.3768369) or via the ACM Digital Library “Cite this” tool on the paper page.