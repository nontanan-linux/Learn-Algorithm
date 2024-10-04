import heapq
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, position, g=0, h=0, parent=None):
        self.position = position
        self.g = g  # Cost from start to this node
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.parent = parent  # Keep track of the parent node for path reconstruction

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # Using Manhattan distance as heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    open_list = []
    closed_list = set()
    
    start_node = Node(start, 0, heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # Check if we reached the goal
        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        # Get neighbors (4 possible directions: up, down, left, right)
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for new_position in neighbors:
            node_position = (current_node.position[0] + new_position[0], 
                             current_node.position[1] + new_position[1])

            # Check if the new position is within the grid bounds
            if (0 <= node_position[0] < len(grid) and
                0 <= node_position[1] < len(grid[0])):
                # Check if walkable
                if grid[node_position[0]][node_position[1]] == 1:
                    continue
                if node_position in closed_list:
                    continue

                g_cost = current_node.g + 1  # Cost from start to neighbor
                h_cost = heuristic(node_position, goal)
                neighbor_node = Node(node_position, g_cost, h_cost, current_node)

                # Check if this path to neighbor is better
                if all(neighbor_node.f < open_node.f for open_node in open_list if open_node.position == node_position):
                    heapq.heappush(open_list, neighbor_node)

    return None  # Path not found

def visualize_grid(grid, path, start, goal):
    grid_array = np.array(grid)
    
    # Create a colormap
    cmap = plt.cm.get_cmap('Greys', 2)
    
    # Plot grid
    plt.imshow(grid_array, cmap=cmap, origin='upper')

    # Mark the path on the grid
    if path is not None:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', color='red')  # Red line for the path

    # Highlight start and goal positions
    plt.plot(start[1], start[0], marker='o', color='green', label='Start')  # Start in green
    plt.plot(goal[1], goal[0], marker='o', color='blue', label='Goal')      # Goal in blue

    plt.title("Grid with A* Path")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xticks(range(len(grid[0])))
    plt.yticks(range(len(grid)))
    plt.grid()
    plt.legend()
    plt.show()

# Example usage
grid = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
]

start = (0, 0)  # Starting position
goal = (4, 4)   # Goal position

# Find the path using A*
path = a_star(start, goal, grid)

# Visualize the grid and the path
visualize_grid(grid, path, start, goal)
