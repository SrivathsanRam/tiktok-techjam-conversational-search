# CP4 cross-encoder fine-tuning

This directory contains the reproducible, target-disjoint fine-tuning pipeline
for the optional CP4 TinyBERT reranker. Generated JSONL pairs, downloaded base
weights, checkpoints, and caches are ignored by Git. The final quantized ONNX
runtime asset is exported separately under `models/` only if it passes the
ablation gates.

The training examples are produced from catalog-derived synthetic sessions
whose targets do not overlap the 200 public targets. Complete target products
are assigned to either training or validation by a stable SHA-256 split. Each
group contains one target product and five hard plus two tail negatives mined
from the actual exact+BM25+RRF+linear candidate funnel. The loss is pairwise
logistic ranking loss. See `data/manifest.json` for the frozen counts and hashes.

```powershell
python -m venv venv-cp4
.\venv-cp4\Scripts\python.exe -m pip install -r fine-tune\requirements.txt

.\venv\Scripts\python.exe -m scripts.create_cp3_synthetic_set `
  --count 1000 --seed 20260831 `
  --output data\releases\cp3\synthetic-1000.jsonl

.\venv-cp4\Scripts\python.exe fine-tune\build_pairs.py `
  --dataset data\releases\cp3\synthetic-1000.jsonl `
  --train-output fine-tune\data\train.jsonl `
  --validation-output fine-tune\data\validation.jsonl `
  --seed 20260901

.\venv-cp4\Scripts\python.exe fine-tune\train_pairwise.py `
  --train fine-tune\data\train.jsonl `
  --validation fine-tune\data\validation.jsonl `
  --output fine-tune\artifacts\model `
  --epochs 3 --batch-size 32 --learning-rate 3e-5 --max-length 160 `
  --device auto

.\venv-cp4\Scripts\python.exe fine-tune\export_onnx.py `
  --model fine-tune\artifacts\model `
  --output models\cp4-tinybert-reranker
```

`--device auto` selects CUDA, then Apple MPS, then CPU. The default base model
and immutable revision are pinned in the script; after the first download,
training can use the local cache. Pass `--base-model <local-directory>` for a
fully offline run.
