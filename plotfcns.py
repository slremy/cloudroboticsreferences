from numpy import uint8, array, arange, nan, unique
import plotly.graph_objs as go
from plotly.offline import plot
from  matplotlib import cm

#text=["<a href=%s>%s</a>"%(i,j) for j, i in zip(data['Document Title'][thecolors == i],data[u'PDF Link'][thecolors == i])],

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = map(uint8, array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    
    return pl_colorscale

def plot_n_save(data, positions, thecolors, name):
    my_cmap = cm.get_cmap('jet')
    cscale = matplotlib_to_plotly(my_cmap, 255)
    index = arange(0,data.shape[0],1);
    traces = [];
    for i in unique(thecolors):
        trace = go.Scatter3d(
                         x=positions[thecolors == i,0],
                         y=positions[thecolors == i,1],
                         z=positions[thecolors == i,2],
                         mode='markers',
                         name=i,
                         text=data['title'][thecolors == i] if hasattr(data, 'title') else data['document title'],
                         #text=["<a href=%s>%s</a>"%(i,j) for j, i in zip(data['Document Title'][thecolors == i],data[u'PDF Link'][thecolors == i])],
                         marker=dict(
                                     #color=thecolors[thecolors == i],
                                     colorscale=cscale,            # choose a colorscale
                                     size=8,
                                     opacity=1
                                     )
                         )
        traces.append(trace);
    plot(traces,filename='%s.html'%(name))

def plot2d_n_save(data, positions, thecolors, name):
    from numpy import zeros
    my_cmap = cm.get_cmap('jet')
    cscale = matplotlib_to_plotly(my_cmap, 255)
    index = arange(0,data.shape[0],1);
    traces = [];
    for i in unique(thecolors):
        trace = go.Scatter3d(
                             x=positions[thecolors == i,0],
                             z=positions[thecolors == i,1],
                             y=zeros(positions[thecolors == i].shape[0]),
                             mode='markers',
                             name=i,
                             text=data['title'][thecolors == i],
                             #text=["<a href=%s>%s</a>"%(i,j) for j, i in zip(data['Document Title'][thecolors == i],data[u'PDF Link'][thecolors == i])],
                             marker=dict(
                                         #color=thecolors[thecolors == i],
                                         colorscale=cscale,            # choose a colorscale
                                         size=8,
                                         opacity=1
                                         )
                             )
        traces.append(trace);
    plot(traces,filename='%s.html'%(name))

