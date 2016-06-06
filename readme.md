This small project explored two approaches to improve the SAGA incremental
gradient algorithm:

1. Take gradients over mini-batches to reduce the memory requirement.
2. Compute gradients in parallel on multiple CPU cores to speed it up.

See our [small report](report.pdf) for an exposition of the methods and some
experimental results.

You'll also find a [Python implementation](./mini_batch) of our mini-batch
approach as well as a [MATLAB implementation](./distributed) of our distributed
approach.
