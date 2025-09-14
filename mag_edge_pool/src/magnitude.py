from magnipy.magnitude.scales import (
    get_scales,
)
from magnipy.magnitude.convergence import guess_convergence_scale
from magnipy.magnitude.compute import (
    compute_magnitude_from_distances,
)
import numpy as np
import copy
import networkx as nx
from magnipy.magnitude.function_operations import mag_diff

#  ╭──────────────────────────────────────────────────────────╮
#  │ Computing Magnitude on (Sub)graphs using Graph Metrics   │
#  ╰──────────────────────────────────────────────────────────╯

def compute_magnitude_subgraphs(
    G,
    ts,
    dist_fn,
    subgraphs=None,
    method="cholesky",
    get_weights=False
):
    """ 
    Compute the magnitude of a graph using a specified distance function.
    The magnitude is computed across a fixed choice of scales.
    This function computes the magnitude of each connected component
    of the graph separately and sums them up to obtain the total magnitude.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    dist_fn : function
        A function that takes a graph as input and returns a distance matrix.
    subgraphs : list of networkx.Graph, optional
        A list of subgraphs. If provided, the magnitude will be computed
        on each subgraph and summed up. If None, the connected components
        of the graph will be used as subgraphs.
    method : str
        The method used to compute magnitude. If 'cholesky' is chosen, the Cholesky decomposition
        will be used to compute magnitude. If 'spread' is chosen, the spread of a metric space will be computed.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    ts : array_like, shape (`n_ts`, )
        The scales at which magnitude has been evaluated.
    """
    original_G = copy.deepcopy(G)
    if subgraphs is None:
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    mags = []

    for s in subgraphs:
        D = dist_fn(s)

        mag = compute_magnitude_from_distances(
            D,
            ts=ts,
            method=method,
            get_weights=get_weights,
            one_point_property=True,
            perturb_singularities=True,
            positive_magnitude=False,
            input_distances=True
        )
        mags.append(mag)

    if get_weights:
        weights = np.zeros((original_G.number_of_nodes(), len(ts)))
        node_idx = {node: idx for idx, node in enumerate(original_G.nodes)}

        for subgraph, (mag) in zip(subgraphs, mags):
            for nn, node in enumerate(subgraph.nodes):
                weights[node_idx[node], :] = mag[nn] 
        return weights, ts
    else:
        total_magnitude = np.sum([mag for mag in mags], axis=0)
        return total_magnitude, ts

def compute_magnitude_subgraphs_with_dist(
    G,
    ts,
    dist_fn,
    subgraphs=None,
    method="cholesky",
    get_weights=False
):
    """ 
    Compute the magnitude of a graph using a specified distance function.
    The magnitude is computed across a fixed choice of scales.
    This function computes the magnitude of each connected component
    of the graph separately and sums them up to obtain the total magnitude.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    dist_fn : function
        A function that takes a graph as input and returns a distance matrix.
    subgraphs : list of networkx.Graph, optional
        A list of subgraphs. If provided, the magnitude will be computed
        on each subgraph and summed up. If None, the connected components
        of the graph will be used as subgraphs.
    method : str
        The method used to compute magnitude. If 'cholesky' is chosen, the Cholesky decomposition
        will be used to compute magnitude. If 'spread' is chosen, the spread of a metric space will be computed.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    ts : array_like, shape (`n_ts`, )
        The scales at which magnitude has been evaluated.
    subgraphs : list of networkx.Graph
        The subgraphs on which the magnitude has been computed.
    Ds : list of np.array
        The distance matrices of the subgraphs.
    mags : list of array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        The magnitudes of the subgraphs.
    """
    if subgraphs is None:
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    mags = []
    Ds = []

    for s in subgraphs:
        D = dist_fn(s)
        Ds.append(D)

        mag = compute_magnitude_from_distances(
            D,
            ts=ts,
            method=method,
            get_weights=get_weights,
            one_point_property=True,
            perturb_singularities=True,
            positive_magnitude=False,
            input_distances=True
        )
        mags.append(mag)

    total_magnitude = np.sum([mag for mag in mags], axis=0)
    
    return total_magnitude, ts, subgraphs, Ds, mags
    
def compute_magnitude_graph(
    G,
    dist_fn,
    ts=None,
    target_value=None,
    n_ts=10,
    log_scale=False,
    scale_finding = "convergence",
    method = "cholesky",
    get_weights=False
):
    """
    Compute the magnitude of a graph using a specified distance function.
    The magnitude is computed either across a fixed choice of scales
    or until the magnitude function has reached a certain target value.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    dist_fn : function
        A function that takes a graph as input and returns a distance matrix.
    ts : None or array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
        Alternativally, if ts is None, the evaluation scales will be choosen automatically.
    scale_finding : str
        The method used to find a convergence scale. Must be one of 'convergence' or 'median_heuristic'.
        If 'convergence' is chosen, the scale will be found by evaluating magnitude at increasing
        scales until the magnitude function reaches the target value.
        If 'median_heuristic' is chosen, the scale will be found using the median heuristic.
        Only used if ts is None.
    target_value : float
        The value of margnitude that should be reached. Only used if ts is None and scale_finding is 'convergence'.
    n_ts : int
        The number of evaluation scales that should be sampled. Only used if ts is None.
    log_scale : bool
        If True sample evaluation scales on a logarithmic scale instead of evenly. Only used if ts is None.
    method : str
        The method used to compute magnitude. If 'cholesky' is chosen, the Cholesky decomposition
        will be used to compute magnitude. If 'spread' is chosen, the spread of a metric space will be computed.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.
   
    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    ts : array_like, shape (`n_ts`, )
        The scales at which magnitude has been evaluated.

    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024.
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations.
    """

    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    if ts is None:
        if G.number_of_nodes() == 1:
            raise Exception(
                "We cannot find the convergence scale for a one point space!"
            )

        if scale_finding == "convergence":
            def comp_mag(G, ts):
                return compute_magnitude_subgraphs(
                    G=G,
                    ts=ts,
                    dist_fn=dist_fn,
                    subgraphs=subgraphs,
                    method=method,
                    one_point_property=True,
                    perturb_singularities=True,
                    positive_magnitude=False,
                )[0]

            if target_value is None:
                target_value = 0.95 * G.number_of_nodes()
            else:
                if target_value >= G.number_of_nodes():
                    raise Exception(
                        "The target value needs to be smaller than the cardinality!"
                    )
                if 0 >= target_value:
                    raise Exception("The target value needs to be larger than 0!")

            t_conv = guess_convergence_scale(
                G, comp_mag=comp_mag, target_value=target_value, guess=10
            )
        else:
            t_conv = median_heuristic(dist_fn, G=None, subgraphs=subgraphs)
        
        if n_ts == 1:
            ts = [t_conv]
        else:
            ts = get_scales(
                t_conv,
                n_ts,
                log_scale=log_scale,
                one_point_property=True,
            )

    magnitude, ts = compute_magnitude_subgraphs(
        G,
        ts,
        dist_fn,
        subgraphs=subgraphs,
        method=method,
        get_weights=get_weights,
        one_point_property=True,
        perturb_singularities=True,
        positive_magnitude=False,
    )

    return magnitude, ts

#  ╭──────────────────────────────────────────────────────────╮
#  │ Alternative Scale Selection Heuristics                   │
#  ╰──────────────────────────────────────────────────────────╯

def median_heuristic_from_distances(D):
    """
    Compute the median heuristic for the scale selection.

    Parameters
    ----------
    D : np.array
        Distance matrix.

    Returns
    -------
    float
        The scale selected by the median heuristic.
    """
    
    d_flat = D[np.triu_indices(D.shape[0], k=1)]
    median = np.median(d_flat)
    return 1 / np.sqrt(median/2)

def median_heuristic(dist_fn, G=None, subgraphs=None):
    """
    Compute the median heuristic for the scale selection.

    
    Parameters
    ----------
    dist_fn : function
        A function that takes a graph as input and returns a distance matrix.
    G : networkx.Graph, optional
        A graph. If provided, the median heuristic will be computed on this graph.
    subgraphs : list of networkx.Graph, optional
        A list of subgraphs. If provided, the median heuristic will be computed
        on each subgraph and the median of the medians will be returned.

    Returns
    -------
    float
        The scale selected by the median heuristic.
    """
    if subgraphs is None:
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    distances = []
    for s in subgraphs:
        D = dist_fn(s)
        d_flat = D[np.triu_indices(D.shape[0], k=1)]
        distances = distances + list(d_flat)
    
    median = np.median(distances)
    return 1 / np.sqrt(median/2)

def get_magdiff(mag1, mag2, ts):
    """
    Compute the difference between the magnitude (functions) of two spaces.
    """
    if len(ts) == 1:
        return mag2[0] - mag1[0]
    else:
        return mag_diff(
            magnitude=mag1,
            D=None,
            D2=None,
            magnitude2=mag2,
            ts=ts,
            ts2=ts,
            scale=False,
            absolute_area=False
        )
