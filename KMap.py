import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy.random as rand
from numpy import inf
import sys
from collections import Iterable

def plot_umap(kmap,alpha=0.1,show=True,**kwargs):
    # Set defaults
    kwargs['interpolation'] = kwargs.get('interpolation','none')
    kwargs['cmap'] = kwargs.get('cmap',plt.cm.bone_r)
    kwargs['ax'] = kwargs.get('ax',None)
    # umap tiem
    newdims = np.dot(2.0,kmap.dim_outputs)-1
    imdata = np.zeros(newdims)
    imoverlaydata = np.zeros(list(newdims)+[4])
    nodecolor = [1,1,1,1.0]-np.array(kwargs['cmap'](0))
    nodecolor[-1]=0
    for ind in xrange(kmap.num_outputs):
        weight = kmap[ind]
        kcoords = kmap._ind2coords(ind)
        imcoords = tuple(np.dot(kcoords,2))
        imoverlaydata[imcoords] = nodecolor
        neighbors = ([1,0],[1,1],[0,1])
        for neighbor in neighbors:
            ncoords = tuple(np.add(kcoords,neighbor))
            if ncoords[0]<kmap.dim_outputs[0] and ncoords[1]<kmap.dim_outputs[1]:
                distance = dist(kmap[kcoords],kmap[ncoords])
                # Plain
                newcoord = tuple(np.add(imcoords,neighbor))
                imdata[newcoord]=distance
                # Overlay
                imoverlaydata[newcoord] = kwargs['cmap'](distance)
                imoverlaydata[newcoord][-1] = alpha
    if show:
        if kwargs['ax'] == None:
            f = plt.gcf()
            f.clear()
            ax = f.add_subplot(111)
        else:
            ax = kwargs['ax']
            f = ax.get_figure()
        del kwargs['ax']
        ax.clear()
        im=ax.imshow(imdata,**kwargs)
        ax.imshow(imoverlaydata,interpolation=None)
        ax.axis('tight')
        limit = ax.axis()
        plt.colorbar(im)
        wy,wx = zip(*[kmap._ind2coords(i) for i in xrange(len(kmap.weights))])
        ax.scatter(np.dot(wx,2.0),np.dot(wy,2.0),s=300,marker=',',cmap=kwargs['cmap'],c=(0,0,0,0.7),linewidths=0) 
        ax.axis(limit)
        plt.tight_layout()
        plt.show()
    return imdata

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
        self.dim_outputs = dim_outputs
        self.num_inputs = num_inputs
        self.i_range = self.complete_irange(input_range)
        self.input_data = []
        self.sigma_list = []
        self.t = 0
        self.num_outputs = num_outputs = np.prod(np.array(dim_outputs))
        self.weights = np.zeros((num_outputs,num_inputs))
        self.dist_map = np.zeros((num_outputs,num_outputs))
        # Initialize weight vectors randomly over the domain
        weight_init = lambda j:(self.i_range[j][1]-self.i_range[j][0])*rand.uniform()+self.i_range[j][0]
        for i in xrange(num_outputs):
          for j in xrange(num_inputs):
            self.weights[i][j] = (weight_init(j))
        self.labelcolors = {}

    def complete_irange(self,irange):
        if len(irange)==1 and self.num_inputs>1:
            new_irange = []
            for i in range(self.num_inputs-len(irange)+1):
                new_irange.append(irange[0])
            return new_irange
        else:
            return irange
    def _ind2coords(self,ind):
        return np.unravel_index(ind,self.dim_outputs)
    def _coords2ind(self,coords):
        return np.ravel_multi_index(coords,self.dim_outputs)
    def __getitem__(self,ind):
        if hasattr(ind,'__getitem__'):
            return self.weights[self._coords2ind(ind)]
        else:
            return self.weights[ind]
    def __call__(self,*args,**kwargs):
        return self.findBMU(*args,**kwargs)
    def findBMU(self,data,pair=False,retdist=False):
        """Find and return the closest weight to some input data.
        
        Arguments:
        retdist    -   when true, returns the distance along with the BMU
        pair - when true, returns a coordinate pair instead of index
        
        """
        mindist = inf
        BMU = None  # Best matched unit
        for i in xrange(len(self.weights)):
            wvec = self.weights[i]
            distance = (dist(wvec,data))
            if distance<mindist:
                BMU = i
                mindist = distance
        if pair:
            BMU = self._ind2coords(BMU)
        if retdist:
            ret = (BMU,mindist)
        else:
            ret = BMU
        return ret
    def sortInputData(self,show_progress=False,input_data=None):
        """Sort all input data we've seen so far by their BMUs"""
        if not input_data:
            input_data = self.input_data
        # Create sorted data storage
        num_outputs = np.prod(np.array(self.dim_outputs))
        self.sortedInputData = []
        for i in xrange(num_outputs):
            self.sortedInputData.append([])
        i=0
        totalOperations = len(input_data)
        for d in input_data:
            if show_progress:
                progressbar(i,totalOperations)
            bmuind,dist = self.findBMU(d,retdist=True)
            self.sortedInputData[bmuind].append(d)
            i+=1
        return self.sortedInputData
    def reverseBMU(self,weight_coord,sort=False):
        """Return all input data seen with the specified BMU"""
        if not self.sortedInputData:
            if sort:
                self.sortInputData()
            else:
                raise ValueError("input data isn't sorted yet")
        mindist=float('inf')
        minval = None
        weight = self[weight_coord]
        weight_coord = self._coords2ind(weight_coord) if isinstance(weight_coord,Iterable) else weight_coord
        data = self.sortedInputData[weight_coord]
        for d in data:
            currdist = dist(d,weight)
            if currdist < mindist:
                mindist = currdist
                minval = d
        return minval
    def update(self,data,e_num,live=True,**kwargs):
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
        self.sortedInputData = []
        e_num = float(e_num)
        if live:
            self.input_data.append(data)
        # Calculate neighborhood size
        t = float(self.t)
        weight_dimensions = len(self.dim_outputs)
        sig0 = np.prod([sum([abs(x) for x in y]) for y in self.i_range])
        sig0 /= float(np.prod(self.dim_outputs))
        timeconst = e_num/np.log(sig0)
        sig = sig0*np.exp(-t/timeconst)
        self.sigma_list.append(sig)
        lrate = self.learning_rate#*np.exp(-t/e_num)
        # Find BMU location and update nearby weights
        BMU_Pos = self.findBMU(data,pair=True)
        BMU_Ind = self._coords2ind(BMU_Pos)
        for i in xrange(len(self.weights)):
            wvec = self.weights[i]
            wpos = self._ind2coords(i)
            if self.dist_map[BMU_Ind][i]==0:
                self.dist_map[BMU_Ind][i] = dist(BMU_Pos,wpos)
            distance = self.dist_map[BMU_Ind][i]
            feedback = np.exp(-(distance**2.0)/(2.0*sig))
            for j in xrange(len(wvec)):
                self.weights[i][j]=wvec[j] + lrate*feedback*(-wvec[j]+data[j])
        self.t+=1
    def updates(self,datalist,e_num=None,iterations=1,rand=True,show=0,plotrate=1,plotfunc=plot_umap,verbose=1,showprogress=False,**kwargs):
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
        showprogress --  show progress bar

        """
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
        updatenum=0.0
        total=len(order)
        for i in order:
            if showprogress:
                progressbar(updatenum,total)
            try:
                self.update(datalist[i],e_num,live=False,**kwargs)
                if show>0 and (np.mod(updatenum,plotrate)==0):
                    ax.clear()
                    plotfunc.__call__(self,**kwargs)
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
                    continue
                if verbose>0:
                    print "silencing..."
                    verbose-=1
                    continue
                print "aborted..."
                return
        if show>0:plt.close(fig)
        if verbose>0:
            print "done..."

    def reset(self):
        """Reinitialize KMap"""
        KMap.__init__(self,self.num_inputs,self.dim_outputs,self.i_range)

def dist(a,b):
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
def magnitude(a):
    return np.sqrt(np.sum(np.power(a,2.0)))

def plot_2doutputs(kmap,intol=2.0,outtol=float('inf'),linewidth=4.0,alpha=0.7,cmap=None,ax=None):
    """Plot weights as graph
    Output layer dimensions must be 2.

    """
    weights = kmap.weights
    bgcolor = (0.0,0.0,0.2)
    wy,wx = zip(*[kmap._ind2coords(i) for i in xrange(len(kmap.weights))])
    plt.scatter(wx,wy,s=300,marker='s',c=(1.0,1.0,1.0,alpha))
    if not ax:
        ax = plt.gca()
    ax.set_xlim((-.5,kmap.dim_outputs[1]-.5))
    ax.set_ylim((kmap.dim_outputs[0]+0.5,-.5))
    ax.set_axis_bgcolor(bgcolor)
    i=0
    lines = []
    mindist,maxdist = float('inf'),-float('inf')
    filteredmin,filteredmax = float('inf'),-float('inf')
    distsum = 0
    n=0
    for v in kmap.weights:
        j=0
        vcoord = kmap._ind2coords(i)
        for w in kmap.weights:
            if j>i:
                wcoord = kmap._ind2coords(j)
                wvdist = dist(vcoord,wcoord)
                weightdist = dist(w,v)
                distsum+=weightdist
                maxdist = max(maxdist,weightdist)
                mindist = min(mindist,weightdist)
                if wvdist<intol and weightdist<outtol:
                    filteredmax = max(filteredmax,weightdist)
                    filteredmin = min(filteredmin,weightdist)
                    lines.append((zip(wcoord,vcoord),weightdist))
                n+=1
            j+=1
        i+=1

    legendy = kmap.dim_outputs[1]
    legendx = kmap.dim_outputs[0]-1
    N = 10
    legendpts = zip(np.linspace(0,legendx,N),np.linspace(legendy,legendy,N))
    legenddists = np.linspace(filteredmin,filteredmax,N-1)
    
    mindist,maxdist = filteredmin,filteredmax
    avg=distsum/float(n)
    std = 1./(1+filteredmax-filteredmin)
    mean = 0.
    if not cmap:
        color = plt.cm.hot_r
        # color = lambda(wdist):[(1.5-((wdist)/(maxdist))**2.0)/1.5,0.0,0.0,1.0]
    else:
        color = cmap
    linewidth_func = lambda(wdist):1.5-(1.5-((wdist-mindist)/(maxdist-mindist))**2.0)/1.5
    for i in xrange(1,len(legendpts)):
        lastpts = legendpts[i-1]
        currpts = legendpts[i]
        weightdist = legenddists[i-1]
        l = linewidth_func(weightdist)
        c = list(color(weightdist))
        c[-1] = alpha*(1.5-l)
        line = zip(lastpts,currpts)
        ax.plot(*line,linewidth=(l)*linewidth,color=c)
        ax.text(sum(line[0])/2.0-np.diff(line[0])/2.0,legendy+.25,"%.2f"%weightdist,color=1-np.array(bgcolor))
    for line,weightdist in lines:
        l = linewidth_func(weightdist)
        c = list(color(weightdist))
        c[-1] = alpha*(1.5-l)
        liney,linex = line
        ax.plot(linex,liney,linewidth=(l)*linewidth,color=c)
    plt.tight_layout()
    plt.show()



def plot_hist(kmap,sizeconst=2000.,ax=None,alpha=0.9,cmap=plt.cm.jet,**kwargs):
    if not kmap.sortedInputData:
        kmap.sortInputData(True)
    weights = kmap.weights
    bgcolor = (1,0.9,0.9)
    wy,wx = zip(*[kmap._ind2coords(i) for i in xrange(len(kmap.weights))])
    
    sizes = np.array([len(x) for x in kmap.sortedInputData])
    maxsize = max(sizes)
    dispsizes = sizeconst*(sizes/float(maxsize))**2.0
    dispsizes = dispsizes.reshape(kmap.dim_outputs)
    if not ax:
        ax = plt.gca()
    ax.set_xlim((-.5,kmap.dim_outputs[1]+.5))
    ax.set_ylim((kmap.dim_outputs[0]+0.5,-.5))
    # ax.set_axis_bgcolor(bgcolor)
    pts = ax.scatter(wx,wy,marker='o',s=dispsizes,c=sizes,cmap=cmap,**kwargs)
    # ax.invert_yaxis()
    plt.show()
    plt.colorbar(pts)
    return sizes

def imagify(vectors,shape,indices=None,rotate=False):
    """Construct a single image from a list of vectors and 2d shape"""
    if not indices:
        indices = range(len(vectors[0]))
    elementsize = len(indices)
    vectors = np.array(vectors).reshape(shape[0],shape[1],len(vectors[0]))
    vectorshape = shape
    newdims = list(np.dot(vectorshape,elementsize))
    imdata = np.zeros(newdims)
    for c in xrange(0,vectorshape[1]):
        for r in xrange(0,vectorshape[0]):
            imcol = c*elementsize
            imrow = r*elementsize
            imcols = np.arange(imcol,imcol+elementsize)
            imrows = np.arange(imrow,imrow+elementsize)
            element = vectors[(r,c)].take(indices)
            if rotate:element = element.reshape(1,-1).T
            imdata[imrow:imrow+elementsize,imcols] = element
    return imdata

def show_vectors(vectors,shape,orientation=2,interpolation='none',**kwargs):
    """Displays a list of vectors as an image"""
    import matplotlib as mpl
    imhoriz = imagify(vectors,shape)
    imvert = imagify(vectors,shape,rotate=True)
    f = plt.figure()
    ax=[]
    cols = 2 if orientation==2 else 1
    if orientation in (0,2):
        ax0 = f.add_subplot(1,cols,len(ax)+1);ax.append(ax0)
        ax0.autoscale(True);ax0.set_adjustable('box-forced')
        im=ax0.imshow(imhoriz,interpolation=interpolation,**kwargs)
        plt.axis('tight')
    if orientation in (1,2):
        ax1 = f.add_subplot(1,cols,len(ax)+1);ax.append(ax1)
        ax1.autoscale(True);ax1.set_adjustable('box-forced')
        im=ax1.imshow(imvert,interpolation=interpolation,**kwargs)
        plt.axis('tight')
    f.tight_layout()
    cax,kw = mpl.colorbar.make_axes([a for a in ax],orientation='horizontal')
    cax.autoscale(True);cax.set_adjustable('box-forced')
    plt.colorbar(im, cax=cax, **kw)
    plt.show()

def show_all(vector_list,shape=None,item_dims=None,image=False,**kwargs):
    """deprecated in favor of show_vectors"""
    vector_list = np.array(vector_list)
    if not shape:
        shape = vector_list.shape
    maxv,minv = np.max(vector_list),np.min(vector_list)
    f,ax = plt.subplots(*shape)
    for i in xrange(len(vector_list)):
        coords = np.unravel_index(i,shape)
        if len(vector_list[i])>0:
            if not item_dims:
                datadim = (-1,len(vector_list[i]))
            else:
                datadim = item_dims
            if image:
                ax[coords].imshow(vector_list[i].reshape(datadim),interpolation='none',**kwargs)
                ax[coords].axis('tight')
                ax[coords].yaxis.set_ticklabels([])
            else:
                ax[coords].stem(vector_list[i],**kwargs)
                ax[coords].axis([-.2,len(vector_list[i]),minv,maxv])
                for ytick in ax[coords].yaxis.get_major_ticks():
                    ytick.label.set_fontsize('x-small')
            ax[coords].xaxis.set_ticklabels([])
        else:
            f.delaxes(ax[coords])
    plt.show()

def getclosestfeatures(kmap):
    if not kmap.sortedInputData:
        raise ValueError("K-Map input data must be sorted")
    closest = np.zeros((kmap.num_outputs,kmap.num_inputs)) 
    for i in xrange(np.prod(kmap.dim_outputs)):
        coords = kmap._ind2coords(i)
        if len(kmap.sortedInputData[i])>0:
            mindist=float('inf')
            minval = None
            for d in kmap.sortedInputData[i]:
                currdist = dist(d,kmap.weights[i])
                if currdist < mindist:
                    mindist = currdist
                    minval = d
            closest[i]= (minval)
    return closest

def plot_2dinputs(kmap,showdata=False,scatters=None,alpha=0.7,wsize=80,dsize=40):
    """Plot weights against current data set.
    Input dimensions must be 2.

    """
    weights = kmap.weights
    if not showdata:
        input_data = []
    else:
        input_data = kmap.input_data
    colors = []
    labels = {}
    labelmarkers = {}
    labelcolors = {}
    markers = ['v','o','4','3','2','1']
    for d in input_data:
        label = kmap.findBMU(d,pair=True)
        if label not in labels:
            labels[label]=[]
            labelmarkers[label] = markers.pop(0)
            markers.append(labelmarkers[label])
            if label not in kmap.labelcolors:
                kmap.labelcolors[label] = (rand.random(),rand.random(),rand.random())
            labelcolors[label] = kmap.labelcolors[label]
        labels[label].append(d)
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
        for label in labels:
            plt.scatter(*zip(*labels[label]),marker=labelmarkers[label],s=dsize,alpha=alpha,color=labelcolors[label])
        # plt.scatter(dx,dy,marker='v',s=dsize,alpha=alpha,color=colors)
        plt.scatter(wx,wy,marker='o',s=wsize,color=(0,0,1,0.5))
    plt.axis([kmap.i_range[0][0],kmap.i_range[0][1],kmap.i_range[1][0],kmap.i_range[1][1]])


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
    w = (w - np.min(w))/(np.max(w)-np.min(w))
    w = w.reshape(kmap.dim_outputs[0],kmap.dim_outputs[1],3)
    if 'interpolation' not in kwargs:
        kwargs['interpolation']='none'
    plt.imshow(w,**kwargs)
    plt.axis('tight')

def progressbar(sofar,total):
    stream = sys.stdout
    sofar=float(sofar)
    total=float(total)
    progress_str = str(sofar/total)
    stream.flush()
    stream.write(progress_str)
    stream.write('\b'*len(progress_str))
