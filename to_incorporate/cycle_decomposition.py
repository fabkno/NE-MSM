import numpy as np
import time
import sys

from cycle_decomp import adjacencylist_from_adjacencymatrix as adjlist_from_adjmatrix_cython,\
                         find_shortest_path as find_shortest_path_cython

class CycleDecomposition(object):
    r""" Flux-cycle decomposition
    
    The flux-cycle decomposition is an algorithm that linearly decomposes a given probability flux array into a list of cycles and assigns to each cycle a cycle weight.
    
    Parameters
    --------------
    rate_matrix: ndarray (N,N) column (row) stochastic probability rate matrix. All non-diagonal elements are equal or larger than zero while
                               its diagonal elements are negative so that the sum over each column (row) is zero.
    
    column_stochastic: bool, default = True, indicates that input rate matrix is column stochastic. When "False" row stochastic matrix is assumned which
                                             will be converted into a column stochastic matrix. Note the all output arrays follow the column stochastic convention
                               
    
    eps: float, default = 1e-10, threshold value below which cycle weights are assumed to be zero
       
    
    References
    --------------
    [1] B. Altaner, S. Grosskinsky, S. Herminghaus, L. Katthaen, M. Timme, and J. Vollmer, 
        "Network representations of nonequilibrium steady states: Cycle decompositions, symmetries, and dominant paths,"
         Phys. Rev. E 85,041133 (2012).
         
    [2] F. Knoch and T. Speck,
        "Cycle representatives for the coarse-graining of systems driven into a non-equilibrium steady state," 
        New J. Phys. 17, 115004 (2015).
    
    
    """
    
    
    def __init__(self,rate_matrix,column_stochastic = True,eps=1e-10):
        
        self.rate_matrix = rate_matrix
        self.eps = eps
        self.phi = []
        self.cycles = []
        
        if column_stochastic == False:
            self.rate_matrix = self.rate_matrix.T
        
        if self.check_if_rate_matrix_is_valid() == False:
            raise ValueError('Caution non-valid rate matrix detected \n column sums do not add up to zero !')
        
        self.compute_flux_matrix()
        
       
        
    def check_if_rate_matrix_is_valid(self):
        # ensures that input rate matrix is stochastic
        if np.all(np.abs(np.sum(self.rate_matrix,axis=0)) < 1e-15):
            return True
        else: 
            return False
    
    
    def compute_flux_matrix(self):
        #computes the probabilty flux matrix by computing the eigenvector corresponding to the largest eigenvalue
        self.flux_matrix=np.copy(self.rate_matrix)
              
        self._eigenvalues,EV = np.linalg.eig(self.rate_matrix)
      
        ind = np.real(np.abs(self._eigenvalues)).argmin()
        
        self.prob = np.real(EV[:,ind]/np.sum(EV[:,ind]))  

        self.flux_matrix *= self.prob    
        
        np.fill_diagonal(self.flux_matrix,0)
        
        
                    
    def back_transform_flux_matrix(self):
        # transforms back all computed cycles and cycle weights to the initial probability flux matrix
        
        if len(self.cycles) <1:
            raise ValueError('The flux-decomposition algorithm has not beed executed \ntry ">> flux_decomposition_slow()" first!! ')
        N = self.rate_matrix.shape[0]
        F = np.zeros([N,N])
    
        for alpha,cycle in enumerate(self.cycles):
            for i in range(len(cycle)-1):
                 F[cycle[i+1],cycle[i]] += self.phi[alpha]
       
        return F
    
    
    def decomposition_quality(self):
        #Check what percentage of the probability flux matrix is lost during the cycle decomposition. 
        #If the percentage is lower than 100% changing the threshold value "eps" increases it for the cost increasing runtime
            
          
        R = np.sum(np.abs(self.flux_matrix - self.back_transform_flux_matrix()))/np.sum(self.flux_matrix)
      
        print 100 - R * 100,"% probabilty flux is preserved during cycle decomposition \nTo increase this value try to decrease the threshold value eps"

    def flux_decomposition(self,time_out=False):
        
        
        self._flux = np.copy(self.flux_matrix)
        
        self.phi=[]
        cycles_add =[]
        start = time.time()
    
        #subtraction of all trivial cycles (detailded balance part)
        for i in range(0,self._flux.shape[0]):
            for j in range(i+1,self._flux.shape[0]):
                if self._flux[i,j] > self.eps:
                    cycles_add.append([j,i,j])
        
        
        
        phi_add = self._flux_decomp_step(cycles_add)
        
        phi_add,self.cycles= self._get_non_zero_cycles(phi_add, cycles_add)
        
        self.phi = np.append(self.phi,phi_add)
               
        
        #compute Betti number
        N_max = np.sum(self._flux >0) -self._flux.shape[0]+1
        k=0
        #start main routine
        while k<N_max: 
            #find index of max flux element
            ind_max = np.unravel_index(self._flux.argmax(),self._flux.shape)
        
            #compute shortest way (breadth first search algorithm) and return cycle
            c_tmp = self._find_shortest_path(self._get_adjlist_from_adjmatrix(self._flux), ind_max[0], ind_max[1])
  
            #compute cycle weight and return modified flux array
            phi_tmp= self._flux_decomp_step([c_tmp])
       
            # if new cycle weight is smaller than eps, exit loop
            if phi_tmp[0] < self.eps:
                if time_out == True:
                    print "time needed: ",time.time() - start
                
		self._get_non_trivial_cycles()
               
		break
              
            self.phi = np.append(self.phi,phi_tmp[0]) 
            self.cycles.append(c_tmp)
            k +=1
            

        if time_out == True:
            print "time needed (Beti nr.): ",time.time() - start
	
	self._get_non_trivial_cycles()    
                            
    def back_transform_flux_matrix(self):
        # transforms back all computed cycles and cycle weights to the initial probability flux matrix
        
        if len(self.cycles) <1:
            raise ValueError('The flux-decomposition algorithm has not beed executed \ntry ">> flux_decomposition_slow()" first!! ')
        N = self.rate_matrix.shape[0]
        F = np.zeros([N,N])
    
        for alpha,cycle in enumerate(self.cycles):
            for i in range(len(cycle)-1):
                 F[cycle[i+1],cycle[i]] += self.phi[alpha]
       
        return F
    
    
    def decomposition_quality(self):
        #Check what percentage of the probability flux matrix is preserved during the cycle decomposition. 
        #If the percentage is lower than 100% changing the threshold value "eps" increases the preserved percentage for the cost of increasing runtime    
          
        R = np.sum(np.abs(self.flux_matrix - self.back_transform_flux_matrix()))/np.sum(self.flux_matrix)
      
        print 100 - R * 100,"% probability flux is perserved during cycle decomposition \nTo increase this value (if not 100% already) try to decrease the threshold value eps"
      
          
    
    def _get_non_trivial_cycles(self):
        if len(self.cycles) <1:
                raise ValueError('The flux-decomposition algorithm has not beed executed \ntry ">> flux_decomposition_slow()" first!! ')
         
        self.non_trivial_cycles = [cycle for cycle in self.cycles if len(cycle)>3]
        self.non_trivial_phi = np.array([self.phi[i] for i in range(len(self.cycles)) if len(self.cycles[i]) >3])    
        
            
    def _flux_decomp_step(self,_cycles):
       #determines cycle weight for given cycle and subtracts its value from the probability flux array
         
       N_cycles = len(_cycles)
       phi_alpha = np.zeros(N_cycles) 
       for n_c in range(N_cycles):
          
            phi_alpha[n_c]=min([self._flux[_cycles[n_c][i+1],_cycles[n_c][i]] for i in range(len(_cycles[n_c])-1)])

            for j in range(len(_cycles[n_c])-1):
                 self._flux[_cycles[n_c][j+1],_cycles[n_c][j]] -= phi_alpha[n_c]   
        
       self._flux[self._flux <=self.eps] = 0 
        
       return phi_alpha
        
        
    def _get_non_zero_cycles(self,_phi,_cycles):
    
        ind = np.where(_phi > self.eps)[0]
       
        cycles_new = [_cycles[i] for i in ind]
        
        return _phi[ind],cycles_new
    
    
    def _find_shortest_path(self,Adj_list,start_state,end_state):
        #finds shortest path by employing breadth-first search, see for example "Newman, Mark. Networks: An Introduction. Oxford university press, 2010."
        
        visited_states=np.zeros(len(Adj_list),dtype=bool)
        predecessor = np.zeros(len(Adj_list),dtype=int)

        visited_states[start_state]=True
        predecessor-=1
 
        queue=[]
        queue.append(start_state)     
        write = 0
        read=0
        l=0
   
        while read <= write:
            l+=1 
       
            for j in range(0,len(Adj_list[queue[read]])):
            
                if visited_states[Adj_list[queue[read]][j]] == False: 
                    write+=1                             
                         
                    queue.append(Adj_list[queue[read]][j])
                    visited_states[Adj_list[queue[read]][j]]=True

                    predecessor[Adj_list[queue[read]][j]]=queue[read]
                
            if visited_states[end_state] == True:
               break
                
            read+=1
    
        cycle=[start_state,end_state]
    
        step = end_state
    
        while 1!=0:        
            step=predecessor[step]
        
            if step == -1:
                break
        
            cycle.append(step)    
      
        cycle.reverse()
        return cycle
    
    def _get_adjlist_from_adjmatrix(self,matrix):

        adj_list=[]
        for col in matrix.T:
            adj_list.append(list(np.where(col != 0)[0]))
        
        return adj_list
    
    
    
