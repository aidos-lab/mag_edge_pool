from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import coalesce, scatter 
import numpy as np
import networkx as nx
import random
import pandas as pd
from mag_edge_pool.src.magnitude import compute_magnitude_subgraphs_with_dist, compute_magnitude_subgraphs
from magnipy.magnitude.compute import compute_magnitude_from_distances
from torch_geometric.utils import to_networkx
from mag_edge_pool.src.metric_choice import choose_graph_metric

def get_scores_edge(this_graph, edges, dist_fn, ts, original_magni, subgraphs, Ds, mags, method="cholesky"):
    """
    Compute edge importance scores for a given graph based on magnitude difference after edge contraction.
    
    Parameters
    ----------
    this_graph : nx.Graph
        The input graph.
    edges : list
        List of edges in the graph.
    dist_fn : Callable
        A function used to compute distance metric between nodes.
    ts : list
        The scale parameters for magnitude computations.
    original_magni : float
        The original magnitude of the graph.
    subgraphs : list
        List of subgraphs at different scales.
    Ds : list
        List of distance matrices corresponding to the subgraphs.
    mags : list
        List of magnitudes corresponding to the subgraphs.
    method : str, optional
        Method for magnitude computation. Default is "cholesky".
        Computes the spread if "spread" is in the method name.

    Returns
    -------
    scores: list
        List of edge importance scores based on magnitude difference.   
    """

    scores = []

    for edge in edges:
        node_a, node_b = edge
        mags_steps = []
        for i, subgraph in enumerate(subgraphs):
            if edge[0] not in subgraph.nodes() or edge[1] not in subgraph.nodes():
                #continue
                mags_steps.append(mags[i])
            else:
                step_graph = subgraph.copy()
                nodes = [m for m in step_graph.nodes()]
                node_a, node_b = edge

                # Merge node_a and node_b into node_a
                neighbors_a = set(step_graph.neighbors(node_a)) - {node_b}
                neighbors_b = set(step_graph.neighbors(node_b)) - {node_a}
                merged_neighbors = neighbors_b.difference(neighbors_a)

                # Connect node_a to the neighbors of both nodes
                for neighbor in merged_neighbors:
                    step_graph.add_edge(node_a, neighbor)

                # Remove node_b and its edges
                step_graph.remove_node(node_b)

                # Compute the magnitude difference
                step_magni, _ = compute_magnitude_subgraphs(step_graph, subgraphs=[step_graph], dist_fn=dist_fn, ts=ts, get_weights=False, method=method)
                mags_steps.append(step_magni[0])
        
        mag_diff_this = abs(original_magni[0] - sum(mags_steps))
        scores.append(mag_diff_this)
    return scores

def get_scores_edge_approx_dist(this_graph, edges, dist_fn, ts, original_magni, subgraphs, Ds, mags, method="cholesky"):
    """
    Compute edge importance scores for a given graph based on magnitude difference after edge contraction.
    Approximate method that updates the distance matrix instead of recomputing it.
    
    Parameters
    ----------
    this_graph : nx.Graph
        The input graph.
    edges : list
        List of edges in the graph.
    dist_fn : Callable
        A function used to compute distance metric between nodes.
    ts : list
        The scale parameters for magnitude computations.
    original_magni : float
        The original magnitude of the graph.
    subgraphs : list
        List of subgraphs at different scales.
    Ds : list
        List of distance matrices corresponding to the subgraphs.
    mags : list
        List of magnitudes corresponding to the subgraphs.
    method : str, optional
        Method for magnitude computation. Default is "cholesky".
        Computes the spread if "spread" is in the method name.

    Returns
    -------
    scores: list
        List of edge importance scores based on magnitude difference.   
    """
    
    scores = []

    for edge in edges:
        node_a, node_b = edge
        mags_steps = []
        for i, subgraph in enumerate(subgraphs):
            if edge[0] not in subgraph.nodes() or edge[1] not in subgraph.nodes():
                #continue
                mags_steps.append(mags[i])
            else:
                step_graph = subgraph.copy()
                nodes = [m for m in step_graph.nodes()]
                
                node_a, node_b = edge

                # Merge node_a and node_b into node_a
                neighbors_a = set(step_graph.neighbors(node_a)) - {node_b}
                neighbors_b = set(step_graph.neighbors(node_b)) - {node_a}
                merged_neighbors = neighbors_b.difference(neighbors_a)

                D =  Ds[i].copy()
                D_sub = Ds[i].copy()

                for neighbor in merged_neighbors:
                    weight = min(D[nodes.index(node_a),:][nodes.index(neighbor)],
                                D[nodes.index(node_b),:][nodes.index(neighbor)])
                    D_sub[nodes.index(node_a), nodes.index(neighbor)] = weight

                index_b = nodes.index(node_b)

                D_sub = np.delete(D_sub, index_b, axis=0)
                D_sub = np.delete(D_sub, index_b, axis=1)

                mag_step = compute_magnitude_from_distances(
                    D_sub,
                    ts=ts,
                    method=method,
                    get_weights=False,
                    one_point_property=True,
                    perturb_singularities=True,
                    positive_magnitude=False,
                    input_distances=True
                )
                mags_steps.append(mag_step[0])
        step_magni = sum(mags_steps)
        mag_diff_this = abs(original_magni[0] - step_magni)
        scores.append(mag_diff_this)
    return scores

def mag_edge_pool(g, ts, dist_fn, original_magni=None, n_steps=None, method="cholesky", scores_method ="full", k=0):
    """
    The algorithm for MagEdgePool and SpreadEdgePool. Uses the magnitude difference after 
    edge contraction to stepwise contract the least important edges. Returns the pooled graph 
    and the cluster assignment that can be used for pooling in a GNN architecture.

    Parameters
    ----------
    g : nx.Graph
        The input graph.
    ts : list
        List of scale parameters for magnitude computations.
    dist_fn : Callable
        A function used to compute distance metric between nodes.
    original_magni : float, optional
        The original magnitude of the graph. If None, it will be computed.
    n_steps : int, optional
        Number of edges to contract. If None, defaults to number of nodes - 1.
    method : str, optional
        Method for magnitude computation. Default is "cholesky". 
        Computes the spread if "spread" is in the method name.
    scores_method : str, optional
        Method to compute scores: "full" or "approx_all". Default is "full".
    k : int, optional
        An index used for labelling the node clusters of nodes that will be merged. 
        Clusters will be labelled from k to k+n.

    Returns
    -------
    this_graph : nx.Graph
        The pooled graph after edge contractions.
    S : np.ndarray
        The assignment matrix that can be used for pooling in a GNN.
    cluster : np.ndarray
        The cluster assignment vector indicating the super-node each original node belongs to.
    """
    if original_magni is None:
        original_magni, _, subgraphs, Ds, mags = compute_magnitude_subgraphs_with_dist(g, dist_fn=dist_fn, ts=ts, get_weights=False, method=method)
    
    this_graph = g.copy()
    n = g.number_of_nodes()

    if n_steps is None:
        n_steps = n-1

    edges = list(this_graph.edges()) 

    if scores_method == "full":
        scores = get_scores_edge(this_graph, edges, dist_fn, ts, original_magni, subgraphs, Ds, mags, method=method)
    elif scores_method == "approx_all":
        scores = get_scores_edge_approx_dist(this_graph, edges, dist_fn, ts, original_magni, subgraphs, Ds, mags, method=method)
    else:
        raise ValueError(f"Unknown scores_method: {scores_method}")

    n_steps = min(n_steps, len(scores))
    S = np.eye(n)
    order_this = np.argsort(scores).flatten().astype(int).tolist()
    scores = [scores[i] for i in order_this]
    edges = [edges[i] for i in order_this]

    i = 0
    for l in range(0, n_steps):

        min_score_indices = np.flatnonzero(np.isclose(scores, scores[0], rtol=1e-8, atol=1e-12))
        chosen_index = random.choice(min_score_indices)  # Randomly select one of the indices
        chosen_edge = edges[chosen_index]
        node_a, node_b = chosen_edge

        this_nodes = list(this_graph.nodes())
        indx_a = this_nodes.index(node_a)
        indx_b = this_nodes.index(node_b)

        if 0 <= indx_b < S.shape[0]: 
            S[indx_a, :] += S[indx_b, :]
            S = np.delete(S, (indx_b), axis=0)
        else:
            raise IndexError(f"Index indx_b={indx_b} is out of bounds for array S with shape {S.shape}")

        scores.pop(chosen_index)
        edges.pop(chosen_index)

        this_graph = nx.contracted_nodes(this_graph, node_a, node_b, self_loops=False)

        if this_graph.number_of_edges() == 0:
            break

        # Rename node_b to node_a for all edges
        new_edges = []
        new_scores = []
        for i, edge in enumerate(edges):
            if (node_b in edge) | (node_a in edge):
                continue
            else:
                new_edge = edge
                new_edges.append(new_edge)
                new_scores.append(scores[i])

        edges = new_edges
        scores = new_scores

        if (len(edges) == 0) & (l < n_steps - 1):
            break
    
    # Normalize each row in S by its sum
    row_sums = S.sum(axis=1, keepdims=True)
    S = S / row_sums

    cluster = np.where(S != 0)[0] + k

    return this_graph, S, cluster


def mag_edge_pool_repeated(g, ts, dist_fn, original_magni=None, n_steps=None, method="cholesky", scores_method ="full", k=0):
    """
    Repeatedly applies the mag_edge_pool function until the desired number of steps is reached.
    This is useful when a large number of edge contractions is needed.
    
    Parameters
    ----------
    g : nx.Graph
        The input graph.    
    ts : list
        List of scale parameters for magnitude computations.
    dist_fn : Callable
        A function used to compute distance metric between nodes.
    original_magni : float, optional
        The original magnitude of the graph. If None, it will be computed.
    n_steps : int, optional
        Total number of edges to contract. If None, defaults to number of nodes - 1
    method : str, optional
        Method for magnitude computation. Default is "cholesky".
        Computes the spread if "spread" is in the method name.
    scores_method : str, optional
        Method to compute scores: "full" or "approx_all". Default is "full".
    k : int, optional
        An index used for labelling the node clusters of nodes that will be merged. 
        Clusters will be labelled from k to k+n.    
    
    Returns
    -------
    this_graph : nx.Graph
        The pooled graph after edge contractions.
    S : np.ndarray
        The assignment matrix that can be used for pooling in a GNN.
    cluster : np.ndarray
        The cluster assignment vector indicating the super-node each original node belongs to.
    """

    n_nodes = g.number_of_nodes()

    if n_steps is None:
        n_steps = n_nodes - 1

    S = np.eye(n_nodes)
    all_nodes = [m for m in g.nodes()]
    cluster = [k+j for j, m in enumerate(all_nodes)]
    this_graph = g.copy()

    while (this_graph.number_of_nodes() > (g.number_of_nodes()+1-n_steps)) and (this_graph.number_of_edges() > 0):
        to_do = n_steps - (n_nodes- this_graph.number_of_nodes())
        this_graph, S_new, this_cluster = mag_edge_pool(this_graph, ts, dist_fn, original_magni=original_magni, n_steps=to_do, method=method, scores_method=scores_method, k=k)

        S = np.dot(S_new, S)

        cluster = np.where(S != 0)[0] + k

    return this_graph, S, cluster


def mag_edge_pool_transform(dataset_sparse, pooling_method, ratio, metric, mag_method):
    """
    Applies the MagEdgePool or SpreadEdgePool transformation to a dataset of graphs.
    Computes the cluster assignments assigning each node to a unique super-node across 
    the dataset.
    
    Parameters
    ----------
    dataset_sparse : torch_geometric.data.Dataset
        A PyTorch Geometric dataset containing sparse graph data.
    pooling_method : str
        The pooling method to use: "MagEdgePool", "SpreadEdgePool", or their approximate variants.
    ratio : float
        The ratio of nodes to retain after pooling.
    metric : str
        The distance metric to use for computing edge importance scores.
    mag_method : str    
        Method for magnitude computation. Default is "cholesky".
        Computes the spread if "spread" is in the method name.
    
    Returns
    -------
    data_list : list
        List of graphs with added cluster assignments.
    """

    if "Approx" in pooling_method:
        scores_method ="approx_all"
    else:
        scores_method = "full"
    
    cluster_index = 0
    data_list = []
    dist_fn = choose_graph_metric(metric, mode="structure")
    for j, data in enumerate(dataset_sparse):

        g = to_networkx(data, to_undirected=True)
        n_steps = int(round((1-ratio) * data.num_nodes))
        if ratio > 0.5:
            this_graph, S, cluster = mag_edge_pool_repeated(g, ts=[1], dist_fn=dist_fn, original_magni=None, n_steps=n_steps, method=mag_method, scores_method = scores_method, k=cluster_index)
        else:
            this_graph, S, cluster = mag_edge_pool(g, ts=[1], dist_fn=dist_fn, original_magni=None, n_steps=n_steps, method=mag_method, scores_method = scores_method, k=cluster_index)
        cluster_index += data.num_nodes
        cluster = torch.tensor(cluster, dtype=torch.int64)
        data.cluster = cluster
        data_list.append(data)
    return data_list


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor


class MagEdgePooling(torch.nn.Module):
    r"""The edge pooling based on a pre-computed cluster assignment.
    In short, edges are contracted based on the edge pooling learnt prior to GNN training.
    See the `"Geometry-aware Edge Pooling for Graph Neural Networks" for more details.
    This implementation is adapted from `torch_geometric.nn.pool.edge_pool.EdgePooling`.

    Args:
        in_channels (int): Size of each input sample.
    """
    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        cluster: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
            cluster (torch.Tensor): The cluster vector assigning each node 
                to a unique super-node across the batch.

        Return types:
            * **x** *(torch.Tensor)* - The pooled node features.
            * **edge_index** *(torch.Tensor)* - The coarsened edge indices.
            * **batch** *(torch.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        x, edge_index, batch, unpool_info = self._pool_graph(
            x, edge_index, batch, cluster)

        return x, edge_index, batch, unpool_info

    def _pool_graph(self, x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        cluster: Tensor,
        ):

        i = cluster.unique().numel()

        # Renumber clusters to be consecutive
        unique_clusters, cluster = cluster.unique(return_inverse=True)
        new_x = scatter(x, cluster, dim=0, dim_size=unique_clusters.numel(), reduce='mean')

        new_edge_index = coalesce(cluster[edge_index], num_nodes=new_x.size(0))
        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
                                 batch=batch) 

        return new_x, new_edge_index, new_batch, unpool_info
        
    def unpool(
        self,
        x: Tensor,
        unpool_info: UnpoolInfo,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (torch.Tensor): The node features.
            unpool_info (UnpoolInfo): Information that has been produced by
                :func:`EdgePooling.forward`.

        Return types:
            * **x** *(torch.Tensor)* - The unpooled node features.
            * **edge_index** *(torch.Tensor)* - The new edge indices.
            * **batch** *(torch.Tensor)* - The new batch vector.
        """
        new_x = x 
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'