# Model And Data Artifacts

Keep these artifacts outside Git:

- object detector weights, for example `yolo11x.pt`
- action/relevance models, for example `best_multi_task_model.pth` and `*.joblib`
- RealSense recordings
- generated videos, reports, and slide decks
- training JSON files

Recommended local layout:

```text
D:\nassim\qxg_artifacts\
  models\
    yolo11x.pt
    rf_model_relevant_objects.joblib
  recordings\
    clip1\
      config.json
      color\
      depth\
```

Point configs to those files with absolute paths or environment-specific overrides.
