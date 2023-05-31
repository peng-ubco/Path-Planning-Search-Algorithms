import networkx as nx
# dependencies for our dijkstra's implementation
from queue import PriorityQueue
from math import inf


"""Dijkstra's lowest-cost path algorithm"""
def dijkstra(graph: 'networkx.classes.graph.Graph', start: str, end: str) -> 'List':
    """Get the shortest path of nodes by going backwards through prev list
    credits: https://github.com/blkrt/dijkstra-python/blob/3dfeaa789e013567cd1d55c9a4db659309dea7a5/dijkstra.py#L5-L10"""

    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node)
        path.reverse()
        return path

    """get the cost of edges from node -> node
    cost(u,v) = edge_weight(u,v)"""

    def cost(u, v):
        return graph.get_edge_data(u, v).get('weight')

    """main algorithm"""
    # predecessor of current node on shortest path
    prev = {}
    # initialize distances from start -> given node i.e. dist[node] = dist(start, node)
    dist = {v: inf for v in list(nx.nodes(graph))}
    # nodes we've visited
    visited = set()
    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node)
    pq = PriorityQueue()

    dist[start] = 0  # dist from start -> start is zero
    pq.put((dist[start], start))

    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        visited.add(curr)
#         print(f'visiting {curr}')
        # look at curr's adjacent nodes
        for neighbor in dict(graph.adjacency()).get(curr):
            # if we found a shorter path
            path_len = dist[curr] + cost(curr, neighbor)
            if path_len < dist[neighbor]:
                # if we haven't visited the neighbor
                if neighbor not in visited:
                    # insert into priority queue and mark as visited
                    visited.add(neighbor)
                    pq.put((path_len, neighbor))
                # otherwise update the entry in the priority queue
                else:
                    # remove old
                    pq.queue.remove((dist[neighbor], neighbor))
                    # update new
                    pq.put((path_len, neighbor))
                # update the distance, we found a shorter one!
                dist[neighbor] = path_len
                # update the previous node to be prev on new shortest path
                prev[neighbor] = curr
#         print("=== Dijkstra's Algo Output ===")
#         print("Distances")
#         print(dist)
#         print("Visited")
#         print(visited)
#         print("Previous")
#         print(prev)
    # we are done after every possible path has been checked
    return backtrace(prev, start, end), dist[end]


"""Shortest paths and path lengths using the A* ("A star") algorithm.
"""
from heapq import heappush, heappop
from itertools import count
from math import sin, cos, sqrt, atan2, radians
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function

__all__ = ["astar_path", "astar_path_length"]

def dist(start_node, end_node, reverse_nodes_dict, KelownaG):
    start_node = reverse_nodes_dict[start_node]
    end_node = reverse_nodes_dict[end_node]
    lat1 = radians(KelownaG.nodes[start_node]['y'])
    lon1 = radians(KelownaG.nodes[start_node]['x'])
    lat2 = radians(KelownaG.nodes[end_node]['y'])
    lon2 = radians(KelownaG.nodes[end_node]['x'])

    # Approximate radius of earth in km
    R = 6373.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def     astar_path(G, source, target, reverse_nodes_dict, KelownaG, heuristic=dist, weight="weight"):
    """Returns a list of nodes in a shortest path between source and target
    using the A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.astar_path(G, 0, 4))
    [0, 1, 2, 3, 4]
    >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
    >>> nx.set_edge_attributes(G, {e: e[1][0] * 2 for e in G.edges()}, "cost")
    >>> def dist(a, b):
    ...     (x1, y1) = a
    ...     (x2, y2) = b
    ...     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight="cost"))
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]


    See Also
    --------
    shortest_path, dijkstra_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(start_node, end_node, reverse_nodes_dict, KelownaG):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}
    visited = set()
    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)
        visited.add(curnode)
        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path, visited

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            ncost = dist + weight(curnode, neighbor, w)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target, reverse_nodes_dict, KelownaG)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    print(queue)
    print(curnode)


def astar_path_length(G, source, target, reverse_nodes_dict, kelowna_G, heuristic=dist, weight="weight"):
    """Returns the length of the shortest path between source and target using
    the A* ("A-star") algorithm.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.
    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    See Also
    --------
    astar_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    weight = _weight_function(G, weight)
    path, _ = astar_path(G, source, target, reverse_nodes_dict, kelowna_G, heuristic, weight)
    return sum(weight(u, v, G[u][v]) for u, v in zip(path[:-1], path[1:]))