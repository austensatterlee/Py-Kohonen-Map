import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
from numpy import inf

class KMap(object):
    """Kohonen Map (self-organizing map)

    Data is fed into the map with the update method.
    Find which weight an input is mapped to with the findBMU method.

    For more info on self-organizing maps see:
        http://en.wikipedia.org/wiki/Self-organizing_map
    For more info on the Kohonen algorithm, see:
        http://www.ai-junkie.com/ann/som/som1.html

    """
    def __init__(self,num_inputs,dim_outputs,domain_range):
        """Initialize weight vectors randomly over the domain

        Arguments:
        num_inputs
        dim_outputs
        domain_range

        """
        self.d_range = domain_range
        self.dim_outputs = dim_outputs
        self.num_inputs = num_inputs
        self.input_data = []
        self.sigma_list = []
        self.t = 0
        self.weights = []
        weight_init = lambda:(self.d_range[1]-self.d_range[0])*rand.rand()+self.d_range[0]
        num_outputs = np.prod(np.array(dim_outputs))
        for i in range(num_outputs):
          self.weights.append([])
          for j in range(num_inputs):
            self.weights[i].append(weight_init())
    def _dist(self,a,b):
        if len(a)!=len(b):
            print len(a),len(b)
            raise ValueError("a and b do not have same length")
        dists = []
        for i in range(len(a)):
            dists.append((a[i]-b[i])**2)
        return np.sqrt(sum(dists))
    def _ind2coords(self,ind):
        return np.unravel_index(ind,self.dim_outputs)
    def _coords2ind(self,coords):
        return np.ravel_multi_index(coords,self.dim_outputs)
    def findBMU(self,data):
        """Find and return the closest weight to some input data"""
        mindist = inf
        BMU = None  # Best matched unit
        for i in range(len(self.weights)):
            wvec = self.weights[i]
            distance = self._dist(wvec,data)
            if distance<mindist:
                BMU = i
                mindist = distance
        return self._ind2coords(BMU)
    def update(self,data,e_num=1):
        """Add given data point to the map and update
        weights accordingly. The sigma parameter still
        needs to be tweaked quite a bit for the KMap to
        work correctly. 
        
        To add a list of data, use updates method.

        Arguments:
        data    --  an input value
        e_num   --  used to calculate the sigma parameter. Feel free
                    to change how this is used.

        """
        self.input_data.append(data)
        # Initalize parameters
        t = self.t
        sig0 = (self.d_range[1]-self.d_range[0])/len(self.weights)
        sig0 = self._dist(max(self.weights),min(self.weights))
        timeconst = t/e_num
        sig = sig0*np.exp(-timeconst)
        self.sigma_list.append(sig)
        self.lrate = 1.0 # 0.9*np.exp(-t/e_num)
        # Find BMU location and update nearby weights
        BMU_Pos = self.findBMU(data)
        BMU_Vec = self.weights[self._coords2ind(BMU_Pos)]
        for i in range(len(self.weights)):
            wvec = self.weights[i]
            dist = self._dist(BMU_Vec,wvec)
            feedback = np.exp(-(dist**2.0)/(np.sqrt(2.0)*sig))
            for j in range(len(wvec)):
                w = wvec[j] + self.lrate*feedback*(-wvec[j]+data[j])
                self.weights[i][j]=w
        self.t+=1
    def updates(self,datalist,show=0,e_num=None,**kwargs):
        """Call update method on a list of data points.

        Arguments:
        datalist    --  list of data points
        show        --  if greater than 0, plot weights and 
                        delay each update by that many seconds (default 0)
        e_num       --  see update method

        """
        if not e_num:
            e_num = len(datalist)
        for d in datalist:
            self.update(d,e_num=e_num,**kwargs)
            if show:
                self.plot_2dweights()
                plt.pause(show)
    def reset(self):
        """Reinitialize KMap"""
        KMap.__init__(self,self.num_inputs,self.dim_outputs,self.d_range)
    def plot_2dweights(self):
        """Plot weights against current data set.
        Only works if data and weights are 2-dimensional.

        """
        
        plt.clf()
        wx=[w[0] for w in self.weights]
        wy=[w[1] for w in self.weights]
        dx=[d[0] for d in self.input_data]
        dy=[d[1] for d in self.input_data]
        plt.scatter(wx,wy,c='r')
        plt.scatter(dx,dy,c='blue',marker='|',s=40)

