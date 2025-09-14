# Geometry-aware Edge Pooling for Graph Neural Networks

This repository implements two hierarchical pooling layers, MagEdgePool and SpreadEdgePool, as introduced in [Geometry-aware Edge Pooling for Graph Neural Networks](https://arxiv.org/abs/2506.11700). These pooling layers preserve graphs' geometry by preserving their structural diversity during edge contraction pooling. 

## Main Functionalities

This repository implements:
- The computation of various **graph metrics** based on a graph's adjacencies.
- The computation of the graph's structural diversity using **metric space magnitude or metric space spread**.
- The computation of **edge importance scores** that assess an edge's importance for the overall diversity.
- The edge contraction pooling methods, **MagEdgePool and SpreadEdgePool**, which stepwise contract the least important edges and are computed prior to GNN training.
- A **PyTorch** implementation of the affermentioned pooling methods that implements edge contraction pooling based on the precomputed pooling assignment.


## Examples

An example notebook demonstrating our pooling methods can be found under the `examples` folder.

## Dependencies

Dependencies can be managed using the [`poetry`](https://python-poetry.org) package manager. Using your activated virtual environment, run the following to install `poetry`:

```python
$ pip install poetry
```

With `poetry` installed, run the following command from the main directory to download the necessary dependencies:

```python
$ poetry install
```

## Citation
Please consider citing our work!

```bibtex
@inproceedings{limbeck2025geometry,
  title         = {Geometry-aware Edge Pooling for Graph Neural Networks}, 
  author        = {Katharina Limbeck and Lydia Mezrag and Guy Wolf and Bastian Rieck},
  booktitle     = {ECML PKDD Workshop on Mining and Learning with Graphs},
  year          = {2025}
}

@inproceedings{limbeck2024metric,
  title         = {Metric Space Magnitude for Evaluating the Diversity of Latent Representations}, 
  author        = {Katharina Limbeck and Rayna Andreeva and Rik Sarkar and Bastian Rieck},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {37},
  year          = {2024}
}
```