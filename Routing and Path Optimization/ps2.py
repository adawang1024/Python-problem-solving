# 6.0002 Problem Set 2 Spring 2022
# Graph Optimization
# Name:
# Collaborators:
# Time:

#
# Finding shortest paths to drive from home to work on a road network
#

from graph import DirectedRoad, Node, RoadMap


# PROBLEM 2: Building the Road Network
#
# PROBLEM 2.1: Designing your Graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the times
# represented?
#
# Write your answer below as a comment:
#
#
#
#

# PROBLEM 2.2: Implementing create_graph
def create_graph(map_filename):
    """
    Parses the map file and constructs a road map (graph).

    Travel time and traffic multiplier should be each cast to a float.

    Parameters:
        map_filename : str
            Name of the map file.

    Assumes:
        Each entry in the map file consists of the following format, separated by spaces:
            source_node destination_node travel_time road_type traffic_multiplier

        Note: hill road types always are uphill in the source to destination direction and
              downhill in the destination to the source direction. Downhill travel takes
              half as long as uphill travel. The travel_time represents the time to travel
              from source to destination (uphill).

        e.g.
            N0 N1 10 highway 1
        This entry would become two directed roads; one from 'N0' to 'N1' on a highway with
        a weight of 10.0, and another road from 'N1' to 'N0' on a highway using the same weight.

        e.g.
            N2 N3 7 uphill 2
        This entry would become two directed roads; one from 'N2' to 'N3' on a hill road with
        a weight of 7.0, and another road from 'N3' to 'N2' on a hill road with a weight of 3.5.
        Note that the directed roads created should have both type 'hill', not 'uphill'!

    Returns:
        RoadMap
            A directed road map representing the given map.
    """

    raise NotImplementedError  # Remove once you are done

# PROBLEM 2.3: Testing create_graph
# Go to the bottom of this file, look for the section under FOR PROBLEM 2.3,
# and follow the instructions in the handout.


# PROBLEM 3: Finding the Shortest Path using Optimized Search Method



# Problem 3.1: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer:
#
#
#

# PROBLEM 3.2: Implement find_shortest_path
def find_shortest_path(roadmap, start, end, restricted_roads=None, has_traffic=False):
    """
    Finds the shortest path between start and end nodes on the road map,
    without using any restricted roads, following traffic conditions.
    If restricted_roads is None, assume there are no restricted roads.
    Use Dijkstra's algorithm.

    Parameters:
        roadmap: RoadMap
            The graph on which to carry out the search.
        start: Node
            Node at which to start.
        end: Node
            Node at which to end.
        restricted_roads: list of str or None
            Road Types not allowed on path. If None, all are roads allowed
        has_traffic: bool
            Flag to indicate whether to get shortest path during traffic or not.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Node, the shortest path from start to end.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """

    raise NotImplementedError  # Remove once you are done

# PROBLEM 4.1: Implement optimal_path_no_traffic
def find_shortest_path_no_traffic(filename, start, end):
    """
    Finds the shortest path from start to end during conditions of no traffic.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end in normal traffic.
        If there exists no path, then return None.
    """

    raise NotImplementedError  # Remove once you are done

# PROBLEM 4.2: Implement optimal_path_restricted
def find_shortest_path_restricted(filename, start, end):
    """
    Finds the shortest path from start to end when local roads and hill roads cannot be used.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end given the aforementioned conditions.
        If there exists no path that satisfies constraints, then return None.
    """

    raise NotImplementedError  # Remove once you are done


# PROBLEM 4.3: Implement optimal_path_heavy_traffic
def find_shortest_path_in_traffic_no_toll(filename, start, end):
    """
    Finds the shortest path from start to end when toll roads cannot be used and in traffic,
    i.e. when all roads' travel times are multiplied by their traffic multipliers.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end given the aforementioned conditions.
        If there exists no path that satisfies the constraints, then return None.
    """

    raise NotImplementedError  # Remove once you are done


if __name__ == '__main__':

    # UNCOMMENT THE LINES BELOW TO DEBUG OR TO EXECUTE PROBLEM 2.3
    pass

    # small_map = create_graph('./maps/small_map.txt')

    # # ------------------------------------------------------------------------
    # # FOR PROBLEM 2.3
    # road_map = create_graph("maps/test_create_graph.txt")
    # print(road_map)
    # # ------------------------------------------------------------------------

    # start = Node('N0')
    # end = Node('N4')
    # restricted_roads = []
    # print(find_shortest_path(small_map, start, end, restricted_roads))
