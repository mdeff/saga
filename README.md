# Mini-batch and distributed SAGA
[Michaël Defferrard](https://deff.ch), [Soroosh Shafiee](https://sorooshafiee.github.io)

This small project explored two approaches to improve the SAGA incremental
gradient algorithm:

1. Take gradients over mini-batches to reduce the memory requirement.
2. Compute gradients in parallel on multiple CPU cores to speed it up.

## Content

See our [proposal](proposal.pdf), [report](report.pdf), and
[presentation](presentation.pdf) for an exposition of the methods and some
experimental results.

You'll also find a [Python implementation](./mini_batch) of our mini-batch
approach as well as a [MATLAB implementation](./distributed) of our distributed
approach.
