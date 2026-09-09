# MinimumDataset

This project-provided minimal dataset is included solely to illustrate the expected UWF image and label formats. It contains 30 example JPG images and one CSV label file; it is not a training, validation, benchmark, or clinical evaluation dataset.

```text
MinimumDataset/
├── Pretraining/
│   ├── Self-supervised/                 # Two unlabelled UWF examples
│   └── Knowledge distillation/           # Two examples and labels.csv
└── Fine-tuning/                          # One positive/negative example per task
    ├── Glaucoma Suspect/
    ├── Age-related Macular Degeneration/
    ├── Retinal Artery Occlusion/
    └── ...
```

`Pretraining/Knowledge distillation/labels.csv` uses one row per image and one binary column per clinical label. The Fine-tuning folders show the intended class semantics for the downstream tasks, but their two images are intentionally not arranged as the `train/<class>/...` and `test/<class>/...` layout required by the training scripts.

Use this dataset only under the applicable ethics, governance, and redistribution permissions. Do not treat it as de-identified by default, merge it with private data in Git, or use it to report model performance.
