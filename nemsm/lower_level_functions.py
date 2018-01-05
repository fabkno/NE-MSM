import numpy as np
from graph_tools.graph_tools import create_spanning_tree


def check_4_transition_matrix(W,transition_prob=False,eps=1e-14):
	'''
	check if input matrix possess characteristics of transition matrix

	Parameters
	-------------

	W : (N,N) ndarray either transition rate or transition probability matrix

	transition_prob : bool (default False) when False W is rate matrix 

	eps : float       numerical threshold for zero
	
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


def calc_stationary_distribution(W,transition_prob=False,checked=False):
	'''

	compute startionary probability distribution
	
	Parameters
	---------------

	W : (N,N) ndarray either transition rate or transition probability matrix

	transition_prob : bool (default False) when False W is rate matrix 

	checked : bool (default False) when True check if input matrix possess characteristics of transition matrix

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
  	>>>  array([ 0.35212134,  0.18138963,  0.18856323,  0.2779258 ])

	'''

	#check if matrix is valid transition matrix
	if checked == False:
		if check_4_transition_matrix(W,transition_prob=transition_prob) == False:
			raise ValueError('Input matrix is not a valid transition matrix')

	#check for row/col stochastic
	if (np.abs(np.sum(W[:,0])) < 1e-14) or (np.abs(np.sum(W[:,0]) -1) < 1e-14) == True:
		#compute eigenvectors and values
		lam,EV = np.linalg.eig(W)		
	else:
		lam,EV = np.linalg.eig(W.T)

	if transition_prob == False:
		ind = np.real(np.abs(lam)).argmin()
		p_star = np.real(EV[:,ind]/np.sum(EV[:,ind]))

	elif transition_prob == True:
		ind = np.real(lam).argmax()
		p_star = np.real(EV[:,ind]/np.sum(EV[:,ind]))

	return p_star



def convert_prob_to_ratematrix(T,lagtime,method='series',eps=1e-10,out_log=True,checked=False):
	'''
	Convert given transition probability matrix in rate matrix

	Parameters
	------------------

	T (N,N) : ndarray
			  row or column stochastic transition probability matrix

	lagtime : float
			  lagtime in absolute units 

	method  : string 
			  conversion method. Currently implemented are: "exact", "pseudo" and "series"

	eps  : float 
			.....

	out_log : bool (default True)
			  when True print number of iterations within method "series"

	checked : bool (default False)
			  whether or not the input matrix T has been check for its characteristics


	Returns
	------------------

	W (N,N) : ndarray
			  time-continuous rate matrix


	Examples
	-----------------
	
	W_true =  np.array([[-0.4,   0.,    0.01,  0.5 ],
				[ 0.,   -0.4,   0.09,  0.2 ],
				[ 0.3,   0.2,  -0.9,   0.1 ],
				[ 0.1,   0.2,   0.8,  -0.8 ]])

	T = np.array([[ 0.69710135,  0.03781962,  0.10800954,  0.285797  ],
 			[ 0.01827284,  0.6906434,   0.08956199,  0.11798734],
 			[ 0.16598029,  0.11634119,  0.44123802, 0.09287974],
 			[ 0.11864552,  0.15519579,  0.36119045,  0.50333592]]

	>>> convert_prob_to_ratematrix(T,1,method='exact')
	>>> [[-0.4   0.    0.01  0.5 ]
 		 [ 0.   -0.4   0.09  0.2 ]
 		 [ 0.3   0.2  -0.9   0.1 ]
 		 [ 0.1   0.2   0.8  -0.8 ]]

	>>> convert_prob_to_ratematrix(T,1,method='pseudo')
	>>> [[-0.30289865  0.03781962  0.10800954  0.285797  ]
 		 [ 0.01827284 -0.3093566   0.08956199  0.11798734]
 		 [ 0.16598029  0.11634119 -0.55876198  0.09287974]
 		 [ 0.11864552  0.15519579  0.36119045 -0.49666408]]

 	>>> convert_prob_to_ratematrix(T,1,method='series')
 	>>> [[ -4.03741281e-01   2.52113747e-03   5.87054092e-02   4.70049660e-01]
 		[  4.30453967e-04  -3.96434526e-01   1.03721905e-01   1.87817760e-01]
 		[  2.79671071e-01   1.86134042e-01  -8.65490143e-01   1.11391890e-01]
 		[  1.23639756e-01   2.07779347e-01   7.03062828e-01  -7.69259310e-01]]

	'''

	#check if matrix is valid transition matrix
	if checked == False:
		if check_4_transition_matrix(T,transition_prob=True) == False:
			raise ValueError('Input matrix is not a valid transition matrix')

	#exact conversion 
	if method == 'exact':

		lam,U = np.linalg.eig(T)

		U_inv = np.linalg.inv(U)

		D_W = np.zeros([T.shape[0],T.shape[0]],dtype=complex)

        #check if real parts are larger than zero
		neg_ind = np.where(np.real(lam)<0)[0]        

		#negative real parts violate T = exp(W*t) --> setting to almost zero

		lam[neg_ind] = 1e-16 + lam[neg_ind].imag*1j

		np.fill_diagonal(D_W,np.log(lam)/lagtime)

		W = np.dot(U,np.dot(D_W,U_inv))

		#check if W is valid rate matrix

		W[np.abs(W)<1e-14] = 0
		W_tmp = np.copy(W)

		np.fill_diagonal(W_tmp,0)

		if np.any(np.real(W_tmp)<0) == True:        
			raise ValueError('Caution transformation does not return correct rate matrix try method="pseudo"')

		return np.real(W)

	elif method == 'pseudo':
		return (T - np.eye(T.shape[0]))/lagtime       

	#matrix logarithm expansion
	elif method == 'series':

		id_m = np.eye(T.shape[0])
		arg = (T - id_m)
		TMP = np.zeros([T.shape[0],T.shape[0]])
		W = arg

		TMP = arg
		
		W_old = W
		i = 2

		while i < 1000:

			TMP = np.dot(TMP,arg)
        
			if np.mod(i,2) == 0:
				W = W_old - TMP/np.float(i)
			else:
				W =W_old + TMP/np.float(i)

			if (np.max(np.abs(TMP))/np.float(i))<eps:
				if out_log == True:
					print "returned after " ,i," iterations"
				return W/lagtime
            
			if np.any(W-np.diag(W)*id_m < 0):
			
				if out_log == True:
					print "negative values: returned after " ,i-1," iterations"

				return W_old/lagtime

			W_old = W
			i += 1
		if out_log == True:
			print "returned after 1000 iterations"
		return W/lagtime

	else: 
		print "non valid method"

def calc_flux_array(M,transition_prob=False,p_steady=None,prob_out = False,eps=1e-10,checked=False):

	'''
	compute flux matrix 	

	Parameters
	--------------
	M (N,N) : ndarray transition matrix 

	transition_prob : bool (default False) when False W is rate matrix 
	
	p_steady (N) : ndarray (default None) steady state probability distribution
	
	prob_out : bool (default False) when True, return steady state probabilities
	
	eps : float (default 1e-10) numerical threshold below which fluxes are set to zero

	checked : bool (default False) when True, input transition matrix is checked to be valid

	Returns
	--------------

	F (N,N) : ndarray probability flux matrix

	p_steady (N) : ndarray (optional) steady state probability distribution

	Examples
	-------------
	
	W = np.array([[-0.4,   0.,    0.01,  0.5 ],
                  [ 0.,   -0.4,   0.09,  0.2 ],
                  [ 0.3,   0.2,  -0.9,   0.1 ],
                  [ 0.1,   0.2,   0.8,  -0.8 ]])
	
	>>> calc_flux_array(W,transition_prob=False)
	>>> [[-0.          0.          0.00188563  0.1389629 ]
 		 [ 0.         -0.          0.01697069  0.05558516]
 		 [ 0.1056364   0.03627793 -0.          0.02779258]
 		 [ 0.03521213  0.03627793  0.15085058 -0.        ]]

	'''

	if checked == False:
		if check_4_transition_matrix(M,transition_prob=transition_prob) is False:
			raise ValueError('Non valid transition matrix detected')

	if p_steady is None:
		p_steady = calc_stationary_distribution(M,transition_prob=transition_prob,checked=True)

	F = np.zeros(M.shape)

	if (np.abs(np.sum(M[:,0])) < 1e-14) or (np.abs(np.sum(M[:,0]) -1) < 1e-14) == True:
		F = M * p_steady
		F *= (F>eps)

	else:
		F = M *p_steady[:,np.newaxis]
		F *= (F>eps)
	
	#check for Kirchhoff's current law, when not valid input p_steady is not correct
	I = F-F.T
	I *= I>0

	if np.any(np.abs(np.sum(I,axis=0) - np.sum(I,axis=1)) > 1e-14) == True:
		raise ValueError('Input steady state probability is not correct')

	if prob_out == True:
		return F,p_steady
	else:
		return F

