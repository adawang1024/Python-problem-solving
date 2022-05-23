# 6.0002 Problem Set 2 Spring 2022
# Graph Optimization
# Name: Yiduo Wang
# Collaborators: None
# Time: 3:30

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
#The nodes represent the points where roads end/meet
#The edges represent weighted roads between points(two directed roads to represent one undirected road)
#Times are represented in weights

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
    
    roadmap = RoadMap()
    with open(map_filename) as f:
          l = f.readlines()
          # print(l)
          for i in range(len(l)):
              ls = l[i].split()        
              # add nodes
              if not roadmap.contains_node(Node(ls[0])):
                  roadmap.insert_node(Node(ls[0]))
              if not roadmap.contains_node(Node(ls[1])):
                  roadmap.insert_node(Node(ls[1]))
            # special situation: uphill
              if ls[3] == "uphill":
                  
                  roadmap.insert_road((DirectedRoad(Node(ls[0]), Node(ls[1]), float(ls[2]), "hill", float(ls[4]))))
              
                  roadmap.insert_road((DirectedRoad(Node(ls[1]), Node(ls[0]), float(ls[2])/2, "hill", float(ls[4]))))
            # undirected road represented as two directed roads
              else:
                  roadmap.insert_road((DirectedRoad(Node(ls[0]), Node(ls[1]), float(ls[2]), ls[3], float(ls[4]))))
                  roadmap.insert_road((DirectedRoad(Node(ls[1]), Node(ls[0]), float(ls[2]),ls[3],float(ls[4]))))
                  
              
    return roadmap
    
    
    

# PROBLEM 2.3: Testing create_graph
# Go to the bottom of this file, look for the section under FOR PROBLEM 2.3,
# and follow the instructions in the handout.


# PROBLEM 3: Finding the Shortest Path using Optimized Search Method



# Problem 3.1: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer: find the shortest path by minimizing the time spent travelling from the start node(home) to the end node(work place)
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

   #   Note: Since this is pseudocode, you will have to choose some concrete implementation details (e.g. data structures used, base cases) for yourself. We also did not include how to handle has_traffic=True - make sure to implement it as described above the pseudocode.
   # if either start or end is not contained in roadmap:return None
    if restricted_roads == None:
        restricted_roads = [] #did not default to None because of the potnetial issue of aliasing
        
    if not roadmap.contains_node(start):
            return None
    if not roadmap.contains_node(end):
            return None  
    # if start and end are the same node:
    #     return ([start], 0) # Empty path with 0 travel time
    else:
        if start == end: 
            return ([start], 0)
        # Label every node as unvisited.
        unvisited = roadmap.get_all_nodes()
        # Label every node with a shortest time value from the start
        # node, with the start node being assigned a travel time of 0 and
        # every other node assigned a travel time of ​∞​.
        time_to = {node: float('inf')for node in roadmap.get_all_nodes()}
        time_to[start] = 0
        predecessor = {node: None for node in roadmap.get_all_nodes()}
        while unvisited:
            # Set unvisited node with least travel time as current node.
            current = min(unvisited, key = lambda node: time_to[node])
            
   # while there are unvisited nodes:
   #     if least travel time to an unvisited node is ​∞​, break.
            if time_to[current] == float("inf"):
                break
      #     If current node is end node, break.
            if current == end: 
                break
    
            for road in roadmap.get_reachable_roads_from_node(current, restricted_roads):
                
                alternative_path_time = time_to[current] + road.get_travel_time(has_traffic)
                #     For each neighbor of the current node:
                neighbor = road.get_destination_node() 
                if alternative_path_time < time_to[neighbor]:
    #     Update the current node's best path and best time.
                    time_to[neighbor] = alternative_path_time
                    predecessor[neighbor] = current
 #     Mark the current node as visited.
            unvisited.remove(current)
        path = []
        current = end
        while predecessor[current] != None:
            path.insert(0,current)
            current = predecessor[current]
        if path != []:
            path.insert(0,current)
           #     No path exists between start and end. Return None.
        else: 
            return None
        return (path, time_to[end])


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

    roadmap = create_graph(filename)
    if find_shortest_path(roadmap, start, end, restricted_roads=None, has_traffic=False):
        return find_shortest_path(roadmap, start, end, restricted_roads=None, has_traffic=False)[0]
    else: 
        return None
    

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
    roadmap = create_graph(filename)
    restricted_roads = []
    restricted_roads.append("local")
    restricted_roads.append("hill")
    if find_shortest_path(roadmap, start, end, restricted_roads, has_traffic=False):
        return find_shortest_path(roadmap, start, end, restricted_roads, has_traffic=False)[0]
    else: 
        return None


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
    roadmap = create_graph(filename)
    restricted_roads = []
    restricted_roads.append("toll")
    if find_shortest_path(roadmap, start, end, restricted_roads, has_traffic=True):
        return find_shortest_path(roadmap, start, end, restricted_roads, has_traffic=True)[0]
    else:
        return None
    
   


if __name__ == '__main__':

    # UNCOMMENT THE LINES BELOW TO DEBUG OR TO EXECUTE PROBLEM 2.3
    pass

    small_map = create_graph('./maps/small_map.txt')

    # ------------------------------------------------------------------------
    # FOR PROBLEM 2.3
    road_map = create_graph("maps/test_create_graph.txt")
    print(road_map)
    # ------------------------------------------------------------------------

    start = Node('N0')
    end = Node('N4')
    restricted_roads = []
    print(find_shortest_path(small_map, start, end, restricted_roads))
