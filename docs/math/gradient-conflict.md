# Gradient conflict geometry

Two gradients conflict in the selected parameter space when their dot product,
or equivalently their cosine similarity, is negative. The cancellation measure

```text
1 - norm(sum_i g_i) / sum_i norm(g_i)
```

quantifies component magnitude lost when gradients combine. It is zero for
perfectly aligned vectors and one for complete cancellation. Zero vectors make
cosine undefined. These quantities depend on the chosen parameters and their
scaling; they describe geometry and cannot establish which loss or sample is
causally wrong.
