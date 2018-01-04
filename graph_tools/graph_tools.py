import numpy as np
from itertools import chain



def get_edgelist_from_adjacency_list(A_list):
	'''
	convert adjacency list to edge list

	Paramters
	-------------
	A_list (list of lists): Elements must be integers

	
	Returns
	-------------
	edge_list (List of tuples)

	
	Example
	-------------

	A_list = [[1,2],[0,2],[1]]

	>>> get_edgelist_from_adjacency_list(A_list)
	
	>>> [[0, 1], [0, 2], [1, 0], [1, 2], [2, 1]]

	'''
	edge_list=[]

	for i,vertex in enumerate(A_list):
		for edge in A_list[i]:
			edge_list.append([i,edge])

	return edge_list



def get_adjacency_matrix_from_edgelist(edge_list,N=False):
	'''
	convert edge list to adjacency matrix

	Paramters
	-------------
	edge_list (list of tuples): Elements must be integers

	
	Returns
	-------------
	A (M,M) : boolian ndarray
	
	N int : (default False)
			when N is integer it gives number of states (vertices) 
			else the number of states is the maximal number occuring in edge list
	
	Example
	-------------

	edge_list = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 1]]

	>>> get_adjacency_matrix_from_edgelist(edge_list)
	
	>>> array([	[False,  True, False],
       			[ True, False,  True],
       			[ True,  True, False]], dtype=bool)

	'''

	if N == False:
		N= max(list(chain(*edge_list)))+1


	A=np.zeros([N,N],dtype=bool)

	for edge in edge_list:          
		A[edge[1],edge[0]]=True


	return A

def get_adjacencylist_from_adjacencymatrix(A_matrix):
	'''
	convert adjacency matrix to adjacency list

	Paramters
	-------------
	A_matrix (M,M) : ndarray
	
	Returns
	-------------
	A (M,M) : boolian ndarray
	
	A_list (list of lists): Elements must be integers

	
	Example
	-------------

	A_matrix  = np.array([[1,2,0],[0,4,2],[1,1,1]])

	>>> get_adjacencylist_from_adjacencymatrix(A_matrix)
	
	>>> [[0, 2], [0, 1, 2], [1, 2]]


	'''

	A_list=[]
	for col in A_matrix.T:
		A_list.append(list(np.where(col != 0)[0]))

	return A_list



def get_edgelist_from_adjacencymatrix(A):
	'''

	converts adjacency matrix to edge list

	Paramters
	-------------

	A (N,N) : ndarray
			  adjacency matrix 

	Returns
	-------------
	
	edge list : list of tuples
				contains all edges read off from the adjacency matrix


	Example
	-------------

	A = np.array([[1,2,0],[0,4,2],[1,1,1]])

	>>> get_edgelist_from_adjacencymatrix(A)

	>>> [[0, 2], [1, 0], [1, 2], [2, 1]]


	'''

	edge_list=[]

	rows,cols=np.where(A.T> 0)

	for i,row in enumerate(rows):
		if row != cols[i]:
			edge_list.append([row,cols[i]])

	return edge_list 


def create_spanning_tree(starting_state,
						Adj_matrix=False,
						Adj_list=False,
						edge_list=False,
						chords_out=True):

	'''
	
	Create a spannung tree, i.e., a graph connecting all vertices without loops
	
	Parameters:
	------------

	starting_state :  int, vertex from which to start building the tree

	Adj_matrix : ndarray (default False)
				 adjacency matrix 

	Adj_list  : list of lists (default False) 
				adjacency list

	edge_list : list of tuples (default False)
				edge_list

	chords_out : bool (default True)
				when True chords (list of edges) is returned, adding a single chord to the tree creates a loop

	Returns
	------------

	A_tree_edge_list : list of tuples
					   contains all edges for tree graph

	chords : list of tuples (optional)
			 contains all chords (edges) for the specific tree

	
	Example
	------------

	A_matrix = np.array([[1,2,0],[0,4,2],[1,1,1]])
	
	>>> create_spanning_tree(0,Adj_matrix=A_matrix,chords_out=True)

	>>> ([[0, 2], [2, 1]], [[1, 0]])


	'''

	if isinstance(Adj_list,bool) == True and isinstance(Adj_matrix,bool) == False:   
		Adj_list = get_adjacencylist_from_adjacencymatrix(Adj_matrix)

	if isinstance(edge_list,bool) == True:
		edge_list=get_edgelist_from_adjacency_list(Adj_list)

	if isinstance(Adj_list,bool) == True and isinstance(Adj_matrix,bool) == True:        
		raise ValueError("EITHER ADJMATRIX OR ADJLIST MUST BE AN INPUT ARGUMENT!!!")

	A_tree_edge_list=[]

	queue=[]
	queue.append(starting_state)     
	write = 0
	read=0
	visited_states=np.zeros(len(Adj_list),dtype=bool)
	visited_states[queue[read]]=True

	while read <= write:

		for j in range(0,len(Adj_list[queue[read]])):

			if visited_states[Adj_list[queue[read]][j]] == False: 
				write+=1

				A_tree_edge_list.append([queue[read],Adj_list[queue[read]][j]])

				queue.append(Adj_list[queue[read]][j])
				visited_states[Adj_list[queue[read]][j]]=True

		read+=1

	if chords_out == True:
		chords=find_chords(edge_list, A_tree_edge_list)
		return A_tree_edge_list,chords
	else:
		 return A_tree_edge_list



def find_chords(edge_list,tree_edge_list):
	'''
	finds all chords for a given graph and its tree

	Parameters
	-------------
	
	edge_list : list of tuples 
				represents original graph


	tree_edge_list : list of tuples
					 represents tree of original graph

	Returns
	------------

	chords : list of tuples
			 list of all chords (edges) 

	
	Example
	------------

	edge_list = [[0, 2], [1, 0], [1, 2], [2, 1]]

	tree_edge_list = [[0, 2], [2, 1]]

	>>> find_chords(edge_list,
					tree_edge_list)
	
	>>> [[1, 0]]

	'''

	state_list = [state for sub in tree_edge_list for state in sub]
	state_list=np.unique(state_list).tolist()

	chords=[]

	for edge in edge_list:
		if edge not in tree_edge_list and edge[::-1] not in tree_edge_list and edge not in chords and edge[::-1] not in chords:

			if edge[0] in state_list and edge[1] in state_list and edge[0] != edge[1]:

				chords.append(edge)
    
	return chords 		 