import Franka
import numpy as np
import random
import pickle
import RobotUtil as rt
import time

random.seed(13)

#Initialize robot object
mybot=Franka.FrankArm()

#Create environment obstacles - # these are blocks in the environment/scene (not part of robot) 
pointsObs=[]
axesObs=[]

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,0,1.0]),[1.3,1.4,0.1])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,-0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1, 0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[-0.5, 0, 0.475]),[0.1,1.2,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

# Central block ahead of the robot
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.45, 0, 0.25]),[0.5,0.4,0.5])
pointsObs.append(envpoints), axesObs.append(envaxes)

prmVertices=[] # list of vertices
prmEdges=[] # adjacency list (undirected graph)
start = time.time()

# TODO: Create PRM - generate collision-free vertices
# TODO: Fill in the following function using prmVertices and prmEdges to store the graph. 
# The code at the end saves the graph into a python pickle file.

maxVertices_= 1000
nnDistance_ = 2.0


# Function to compute L2 norm distance between two vertices
def compute_distance(vertex1, vertex2):
    return np.linalg.norm(vertex1 - vertex2)

# Function to find nearest vertices
def find_nearest_vertices_idx(target_vertex, vertices):
    nearest_vertices_idx = []
    for idx, vertex in enumerate(vertices):
        distance = compute_distance(target_vertex[:5], vertex[:5])
        if distance <= nnDistance_:
            # Check edge collision & append to list if no collision
           if not mybot.DetectCollisionEdge(target_vertex, vertex, pointsObs, axesObs):
                nearest_vertices_idx.append(idx)
            
    # print(f"nn_list :{nearest_vertices_idx}")
    # exit()
    return nearest_vertices_idx

prmEdges = {i: [] for i in range(1000)}
while len(prmVertices) < maxVertices_:
    # Randomly generated point
    q_r = np.array(mybot.SampleRobotConfig())
    
    # Check self collision
    if not mybot.DetectCollision(q_r, pointsObs, axesObs):
        prmVertices.append(q_r)
        q_r_idx = len(prmVertices)-1
        nn_points_idx = find_nearest_vertices_idx(q_r, prmVertices)
        # prmEdges[q_r_idx].append(nn_points_idx)

        for point_idx in nn_points_idx:
            prmEdges[point_idx].append(q_r_idx)
            prmEdges[q_r_idx].append(point_idx)
        # prmEdges.append(find_nearest_vertices_idx(q_r, prmVertices))
        # prmEdges[q_r].append(find_nearest_vertices_idx(q_r, prmVertices))
        print(f"Prm vertices :{len(prmVertices)}")
    



def PRMGenerator():
    global prmVertices
    global prmEdges
    global pointsObs
    global axesObs
    
    pointsObs = np.array(pointsObs)
    axesObs = np.array(axesObs)
    
    while len(prmVertices)<1000:
        # sample random poses
        print(len(prmVertices))

    #Save the PRM such that it can be run by PRMQuery.py
    f = open("myPRM.p", 'wb')
    pickle.dump(prmVertices, f)
    pickle.dump(prmEdges, f)
    pickle.dump(pointsObs, f)
    pickle.dump(axesObs, f)
    f.close

if __name__ == "__main__":

    # Call the PRM Generator function and generate a graph
    PRMGenerator()

    print("\n", "Vertices: ", len(prmVertices),", Time Taken: ", time.time()-start)