import numpy as np
from graph_tools.graph_tools import create_spanning_tree


def check_4_transition_matrix(W,transition_prob=False,eps=1e-14):
	'''
	check if input matrix possess characteristics of transition matrix

	Parameters
	-------------

	W : (N,N) ndarray
			either transition rate or transition probability matrix

	transition_prob : bool (default False)
					  when False W is rate matrix 
	eps : float : numerical threshold for zero
	
	Returns
	-------------

	Bool : True 


	Example
	--------------

	W = np.array([[-0.4,   0.,    0.01,  0.5 ],
                  [ 0.,   -0.4,   0.09,  0.2 ],
                  [ 0.3,   0.2,  -0.9,   0.1 ],
                  [ 0.1,   0.2,   0.8,  -0.8 ]])

    >>> check_4_transition_matrix(W,transition_prob=False)

    >>> True


	'''
	M = np.copy(W)
	#check if input matrix has any negative non-diagonal entries

	if transition_prob == True:

		if np.any(M) <0:
			print 'Negative nondiagonal elements detected'
			return False

	else:


		np.fill_diagonal(M,0)

		if np.any(M) <0:
			print 'Negative nondiagonal elements detected'
			return False
		
		np.fill_diagonal(M,np.diag(W))

	if transition_prob == True:

		if np.all(np.abs(np.sum(M,axis=0) -1)<eps) == False:

			if np.all(np.abs(np.sum(M,axis=1) -1)<eps) == False:
				print 'Input transition matrix is neither column nor row stochastic'
				return False
				
			
			else:
				M = M.T

	else:
		print M
		if np.all(np.abs(np.sum(M,axis=0))<eps) ==False:
			
			if np.all(np.abs(np.sum(M,axis=1))<eps) == False:
				print 'Input transition matrix is neither column nor row stochastic'
				return False
			
			else:
				M = M.T




	#check if all egdes possess inverse edge, i.e. i->j --> j-->i
	if np.any(((M>0)*1.0 + (M.T>0)*1.0) == 1):
		print 'Input transition matrix does not possess reversible edges'
		return False


	tree = create_spanning_tree(0,Adj_matrix=M,chords_out=False)

	if len(tree) +1 !=M.shape[0]:
		print 'Input transition matrix is not connected'
		return False

	return True






def calc_stationary_distribution(W,
								 transition_prob=False,
								 checked=False):
	'''

	compute startionary probability distribution
	
	Parameters
	---------------

	W : (N,N) ndarray
			either transition rate or transition probability matrix

	transition_prob : bool (default False)
					  when False W is rate matrix 

	checked : bool (default False)
					when True check if input matrix possess characteristics of transition matrix

	Returns
	----------------

	p : (N) ndarray
			steady-state probability distribution

	

	Examples
	----------------

	W =  np.array([[-0.4,   0.,    0.01,  0.5 ],
                  [ 0.,   -0.4,   0.09,  0.2 ],
                  [ 0.3,   0.2,  -0.9,   0.1 ],
                  [ 0.1,   0.2,   0.8,  -0.8 ]])

  	>>> calc_stationary_distribution(W) 
  	
  	>>>  


   '''

	#check if matrix is valid transition matrix
	if check_4_transition_matrix(W,transition_prob=transition_prob) == False:
		raise ValueError('Input matrix is not a valid transition matrix')


	#compute eigenvectors and values
	lam,EV = np.linalg.eig(W)

	if transition_prob == False:
		ind = np.real(np.abs(lam)).argmin()
		p_star = np.real(EV[:,ind]/np.sum(EV[:,ind]))

	elif transition_prob == True:
		ind = np.real(lam).argmax()
		p_star = np.real(EV[:,ind]/np.sum(EV[:,ind]))

	return p_star