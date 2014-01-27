import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    def __init__(self,num_inputs,dim_outputs,input_range,**kwargs):
        """Initialize weight vectors and learning parameters

        Arguments:
        num_inputs      - number of input dimensions
        dim_outputs     - a tuple designating shape of output layer
        domain_range    - nested tuple designating range of each input dimensions

        Keyword Arguments:
            rate - value between 0 and 1 specifying decay rate of learning (default: 0.9)

        Example:

            # Generate random RGB color vectors
            input = np.random.rand(100,3)
            som = KMap(2,(3,1),((0,1),(0,1)))
            # Learn over input data 10 times, vizualize progress after each iteration.
            som.updates(input,iterations=10,show=1,plotrate=0)
            som.plot_colors() # Colors should appear grouped together 

        """
        self.learning_rate = kwargs.get('rate',0.7)
        self.i_range = input_range
        self.dim_outputs = dim_outputs
        self.num_inputs = num_inputs
        self.input_data = []
        self.sigma_list = []
        self.t = 0
        num_outputs = np.prod(np.array(dim_outputs))
        self.weights = np.zeros((num_outputs,num_inputs))
        self.dist_map = np.zeros((num_outputs,num_outputs))
        # Initialize weight vectors randomly over the domain
        weight_init = lambda j:(self.i_range[j][1]-self.i_range[j][0])*rand.uniform()+self.i_range[j][0]
        for i in xrange(num_outputs):
          for j in xrange(num_inputs):
            self.weights[i][j] = (weight_init(j))
    def _dist(self,a,b):
        """Find distance between two vectors of arbitrary length.

        a and b must be equal in length.

        """
        # if len(a)!=len(b):
            # print len(a),len(b)
            # raise ValueError("a and b do not have same length")
        # dists = []
        # for i in xrange(len(a)):
            # dists.append((a[i]-b[i])**2)
        # return np.sqrt(sum(dists))
        return np.sqrt(np.sum(np.power((np.array(a)-np.array(b)),2.0)))
    def _magnitude(self,a):
        return np.sqrt(np.sum(np.power(a,2.0)))
    def _ind2coords(self,ind):
        return np.unravel_index(ind,self.dim_outputs)
    def _coords2ind(self,coords):
        return np.ravel_multi_index(coords,self.dim_outputs)
    def __getitem__(self,ind):
        if hasattr(ind,'__getitem__'):
            return self.weights[self._coords2ind(ind)]
        else:
            return self.weights[ind]
    def findBMU(self,data):
        """Find and return the closest weight to some input data"""
        mindist = inf
        BMU = None  # Best matched unit
        for i in xrange(len(self.weights)):
            wvec = self.weights[i]
            distance = (self._dist(wvec,data))
            if distance<mindist:
                BMU = i
                mindist = distance
        return self._ind2coords(BMU)
    def update(self,data,e_num,live=True):
        """Add given data point to the map and update
        weights accordingly. The sigma parameter still
        needs to be tweaked quite a bit for the KMap to
        work correctly. 
        
        To add a list of data, use updates method.

        Arguments:
        data    --  an input value
        e_num   --  used to calculate the sigma parameter (size of neighborhoods).
                    Feel free to change how this is used.
        live    --  when true, input data is recorded

        """
        e_num = float(e_num)
        if live:
            self.input_data.append(data)
        # Calculate neighborhood size
        t = float(self.t)
        weight_dimensions = len(self.dim_outputs)
        sig0 = np.prod(self.dim_outputs)
        timeconst = e_num/np.log(sig0)
        sig = sig0*np.exp(-t/timeconst)
        self.sigma_list.append(sig)
        lrate = self.learning_rate*np.exp(-t/e_num)
        # Find BMU location and update nearby weights
        BMU_Pos = self.findBMU(data)
        BMU_Ind = self._coords2ind(BMU_Pos)
        for i in xrange(len(self.weights)):
            wvec = self.weights[i]
            wpos = self._ind2coords(i)
            if BMU_Pos == wpos:
                continue
            else:
                if self.dist_map[BMU_Ind][i]==0:
                    self.dist_map[BMU_Ind][i] = self._dist(BMU_Pos,wpos)
                dist = self.dist_map[BMU_Ind][i]
                feedback = np.exp(-(dist**2.0)/(2.0*sig))
                for j in xrange(len(wvec)):
                    self.weights[i][j]=wvec[j] + lrate*feedback*(-wvec[j]+data[j])
        self.t+=1
    def updates(self,datalist,e_num=None,iterations=1,rand=True,show=0,plotrate=1,plotfunc=None,verbose=1,**kwargs):
        """Call update method on a list of data points.

        Arguments:
        datalist    --  list of data points

        Keyword Arguments:
        e_num       --  see update method
        iterations  --  number of learning iterations over input data
        rand        --  when true, feeds data to the KMap in uniform random order
        show        --  if show>0, plot weights and 
                        delay each update by that many milliseconds 
                        (default -1)
        plotrate    --  number of updates to perform before redrawing visualization.
                        when plotrate=0, redraws occur after len(datalist) updates.
        plotfunc    --  function used to visualize map when 'show' is >0
        verbose     --  level of progress feedback

        """
        datalist = np.array(datalist)
        datalength = len(datalist)-1
        self.input_data = datalist
        if not e_num:
            e_num = datalength*iterations*1.0
        if rand:
            numupdates = datalength*iterations
            order = np.random.permutation(np.repeat(range(datalength),iterations))
        else:
            order = np.repeat(np.arange(datalength),iterations)

        # Set up vizualization
        if show>0:
            if plotfunc==None:
                plotfunc = plot_colors
            if plotrate==0:
                plotrate = datalength
            plt.ion()
            fig,ax = plt.subplots()
            plt.show()

        # Commence learning!
        updatenum=0
        for i in order:
            try:
                self.update(datalist[i],e_num,live=(False),**kwargs)
                if show>0 and (np.mod(updatenum,plotrate)==0):
                    plotfunc.__call__(self)
                    plt.pause(show/1000.0)
                if verbose>1:
                    print "Update #%d, feeding in datalist[%d]"%(updatenum,i)
                if verbose>0:
                    if np.mod(updatenum,datalength)==0:
                        print "Iteration #%d..."%(updatenum/(datalength))
                updatenum+=1
            except KeyboardInterrupt:
                if show>0:
                    plt.close(fig)
                    show=0
                else:
                    print "aborted..."
                    return
        if show>0:plt.close(fig)
        print "done..."
    def reset(self):
        """Reinitialize KMap"""
        KMap.__init__(self,self.num_inputs,self.dim_outputs,self.i_range)

def plot_2dweights(kmap,data=True,scatters=None):
    """Plot weights against current data set.
    Input/weight dimensions must be 2.

    """
    weights = kmap.weights
    if not data:
        input_data = []
    else:
        input_data = kmap.input_data
    wx=[w[0] for w in weights]
    wy=[w[1] for w in weights]
    dx=[d[0] for d in input_data]
    dy=[d[1] for d in input_data]
    if scatters:
        scatters[0].set_offsets(zip(wx,wy))
        scatters[1].set_offsets(zip(dx,dy))
        return scatters
    else:
        plt.cla()
        plt.scatter(wx,wy,c='r',marker='*',s=50)
        plt.scatter(dx,dy,c='b',marker='o',s=30)
def plot_colors(kmap,**kwargs):
    """Visualize output layer weights with colors
    Input/weight dimensions must be 3 or less

    Keyword arguments are passed to matplotlib's imshow

    """

    if kmap.num_inputs>3:
        raise ValueError("This plot only works for input dimensions <=3")
    w = kmap.weights
    if kmap.num_inputs<3:
        w = map(lambda x:np.concatenate((x,np.zeros(3-kmap.num_inputs))),w)
    w = np.array(w)
    w = w.reshape(kmap.dim_outputs[0],kmap.dim_outputs[1],3)
    if 'interpolation' not in kwargs:
        kwargs['interpolation']='none'
    plt.imshow(w,**kwargs)
