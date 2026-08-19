# Prompt strategy comparison

Winner (default): **few_shot**

| Strategy | Accuracy | Groundedness | Combined | Acc hits | Ground hits | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| plain | 100.00% | 100.00% | 100.00% | 7/7 | 7/7 | 1.752 |
| few_shot | 100.00% | 100.00% | 100.00% | 7/7 | 7/7 | 1.213 |
| chain_of_thought | 100.00% | 100.00% | 100.00% | 7/7 | 7/7 | 2.443 |

Accuracy: expected keyword appears in the final answer (refusals count as misses).  Groundedness: final-answer content words are supported by retrieved context, or the model refused instead of hallucinating.

_Scored 7/18 questions where all three strategies returned a real model answer (API errors excluded)._
