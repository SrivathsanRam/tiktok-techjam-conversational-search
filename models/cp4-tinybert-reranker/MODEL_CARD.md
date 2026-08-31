# CP4 TinyBERT pairwise reranker

This is a locally fine-tuned derivative of
[`cross-encoder/ms-marco-TinyBERT-L2-v2`](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2),
revision `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`, licensed Apache-2.0.

The model has 4,386,049 parameters. It was trained for three CPU epochs with a
pairwise logistic ranking loss, learning rate `3e-5`, batch size 32, and maximum
sequence length 160. Training used 1,740 query groups and validation used 465
groups from complete, target-disjoint synthetic sessions. Each training group
contains the target plus five mined hard negatives and two tail negatives.

Validation group MRR changed from `0.698262` before task fine-tuning to
`0.774857` at the selected third epoch. The shipped model is dynamically
quantized to unsigned 8-bit ONNX weights (4.49 MB). It reranks only the first 20
candidates classified as specific-buying intent. Its score is fused with the
CP3 rank using reciprocal-rank fusion at neural weight `0.15`.

The model is an offline ranking component, not a general-purpose language
model. It should not be used for factual generation or outside this catalog
retrieval task without separate evaluation. If the model, NumPy, or ONNX
Runtime cannot load, the agent continues with the deterministic CP3 path.
