from numpy import *
import random as rnd
import BigMat as bm
import weakref
from DataSet import BatchSet
from Util import ensure_dir
import collections
import time
from time import time as now
import logging
import os,shutil

# Import and configure everything we need from matplotlib
publish_mode = False
import matplotlib
#matplotlib.use("agg")
matplotlib.rcParams.update({'font.size': 9, 'font.family': 'serif', 'text.usetex' : publish_mode})
from matplotlib.pyplot import *
from matplotlib.ticker import NullFormatter
from matplotlib import colors as colors
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
import Tkinter as Tk

_curr_logfile = None
_num_logfile_entries = 0
_best_logfile_errors = {}
#_logs_dir = "C:/Users/Andrew/Dropbox/share/tom/logs"
_logs_dir = "logs"

def open_logfile(prefix='train',msg=""):
    global _curr_logfile
    global _num_logfile_entries
    global _best_logfile_errors
    global _logs_dir

    dir = ensure_dir(_logs_dir)
    _curr_logfile = "%s/%s-%s.html" % (dir,prefix,time.strftime("%m%d%H%M",time.localtime()))
    with open(_curr_logfile,"w") as f:
        f.write("<html><head><title>%s</title></head>\n<body>\n" % _curr_logfile)
        f.write("<center style='font-size:120%%'>%s</center>\n" % _curr_logfile)
        f.write("<center style='font-size:120%%'>%s</center>\n" % msg)
        f.flush()
        f.close()
    _num_logfile_entries = 0
    _best_logfile_errors = {}
    return _curr_logfile

def close_logfile():
    global _best_logfile_errors
    str = '<hr><div><table cellspacing=6 cellpadding=0><tr><td></td><td><b>train</b></td><td><b>valid</b></td><td><b>test</b></td></tr>'
    best_errors = sorted(_best_logfile_errors.items(),key = lambda item: item[1][1])
    for test in best_errors:
        str += '<tr>'
        str += '<td><a href="#case%d">RUN %d:</a></td>' % (test[0],test[0])
        str += ('<td>%.2f%%</td>' % test[1][0]) if test[1][0] != None else '<td></td>'
        str += ('<td>%.2f%%</td>' % test[1][1]) if test[1][1] != None else '<td></td>'
        str += ('<td>%.2f%%</td>' % test[1][2]) if test[1][2] != None else '<td></td>'
        str += '</tr>'
    str += '</table></div>\n'
    str += '</body></html>\n'

    with open(_curr_logfile,"a") as f:
        f.write(str)
        f.flush()
        f.close()


#######################################################

class TrainingReport(object):

    def __init__(self,trainer,verbose=True,interval=10,interval_rate=1.0,visualize=False,log_mode='none',window_size="compact",callback=None):
        self.trainer = weakref.ref(trainer)
        self.verbose = verbose
        self.visualize = visualize
        self._last_update_epoch = 0
        self.interval  = interval
        self.interval_rate = interval_rate
        self.log_mode = log_mode  # "none", "html", "html_anim" for animated gifs
        self.callback = callback
        self._callback_msg = None
        self._start_time = now()
        self._screenshots = []
        self._best_validerror = inf

        if visualize:
            self._window = TrainingReportWindow(trainer,window_size)

    def __del__(self):
        if self._window:
            self._window.destroy()

    def __call__(self,event):
        '''
        Collects statistics about the model currently being trained by
        the 'trainer'. 
        '''
        trainer = self.trainer()
        # Only update stats at certain epoch intervals, for the sake of training speed
        if event == "epoch" and (trainer.epoch - self._last_update_epoch) < int(self.interval):
            return
        self._last_update_epoch = trainer.epoch
        self.interval *= self.interval_rate  # speed up or slow down the rate of updates that get logged

        stats = {}
        stats["time"]  = now() - self._start_time
        stats["epoch"] = trainer.epoch
        stats["learn_rate"] = trainer.learn_rate
        stats["momentum"]   = trainer.momentum
        for fold in ('train','valid','test'):
            stats[fold] = self._collect_stats_on_fold(fold)

        if self.callback:
            self._callback_msg = self.callback(event,stats)

        self.log(event,stats)

    def log(self,event,stats):
        # common prologue
        msg = '%5.1fs: ' % stats['time']
        if event == 'epoch': msg += '[%3d]' % stats['epoch']
        else:                msg += event
        trstats = stats['train']

        if self.trainer().task() == "classification":
            # classification-specific output format
            msg += ': err=%.2f%%' % trstats['error rate']
            if stats['valid']:  msg += '/%.2f%%' % stats['valid']['error rate']
            if stats['test']:   msg += '/%.2f%%' % stats['test']['error rate']
        else:
            # regression-specific output format
            msg += ': loss=%.3f' % trstats['loss']
            if stats['valid']:  msg += '/%.3f' % stats['valid']['loss']
            if stats['test']:   msg += '/%.3f' % stats['test']['loss']

        # common epilogue: training loss + regularizer + penalty
        msg += '; (%.3f+%.3f+%.3f)' % (trstats['loss'],trstats['regularizer'],trstats['penalty'])

        # common epilogue: learning rate and momentum
        msg += ' r=%.3f' % stats['learn_rate']
        if stats['momentum'] > 0.0:
            msg += ' m=%.2f' % stats['momentum']

        logger = logging.getLogger()
        logger.info(msg)

        if self.visualize:
            self._window.log(event,stats,msg)

        self._update_log(event,stats,msg)

    ########################################################

    def _collect_stats_on_fold(self,fold):
        '''
        Give a particular fold (training/testing) this evaluates the model using
        the current fold, and collects statistics about how the model is performing.
        '''
        data = self.trainer().data[fold]
        if data.size == 0:
            return None
        
        # Calculate the current performance stats in batches,
        # so that we don't blow the GPUs memory by sending the
        # whole dataset through at once
        stats = {}
        batches = BatchSet(data.X,data.Y,512)
        for batch in batches:
            bstats = self._collect_stats_on_batch(batch)
            for key,val in bstats.items():
                if not stats.has_key(key):
                    stats[key] = []
                stats[key].append(val)

        # For each key, call a 'reducer' to combine all values collected
        def stackHbatches(Hbatches):
            Hall = []
            nlayer = len(Hbatches[0])
            for i in range(nlayer):
                Hall.append(vstack([Hbatch[i]  for Hbatch in Hbatches]))
            return Hall

        reducers = {"H" : stackHbatches,
                    "loss" : mean,
                    "regularizer" : mean,
                    "penalty" : mean,
                    "error rate" : mean}
        for key,val in stats.items():
            stats[key] = reducers[key](val)

        return stats
        

    def _collect_stats_on_batch(self,batch):
        '''
        Give a particular fold (training/testing) this evaluates the model using
        the current fold, and collects statistics about how the model is performing.
        '''
        X,Y = batch
        model = self.trainer().model
        H = model.eval(X,want_hidden=True)
        stats = {}
        stats["H"]           = [bm.as_numpy(Hi).copy() for Hi in H]  # make a copy of hidden activations
        stats["loss"]        = model.loss(H[-1],Y)  # scalar loss value
        stats["regularizer"] = model.regularizer(H) # scalar hidden unit regularization penalty
        stats["penalty"]     = model.penalty()      # scalar weight penalty
        if self.trainer().task() == "classification":
            stats["error rate"] = 100*count_nonzero(array(argmax(H[-1],axis=1)) != argmax(Y,axis=1)) / float(batch.size)
        return stats
        

    def _update_log(self,event,stats,msg):
        global _num_logfile_entries
        global _best_logfile_errors

        if self.log_mode == 'none':
            return

        str = ""
        if event == 'start':
            # If this is a new training run, 
            _num_logfile_entries += 1
            id = _num_logfile_entries
            bgcolors = ['#eeeeee','#eeeeff','#eeffee','#ffeeee','#ffffee','#ffeeff','#eeffff','#ffffff']
            str += '\n\n\n<hr/><div style="background:%s"><center><a name="case%d"></a><b style="font-size:150%%">RUN %d</b></center>\n' % (bgcolors[mod(_num_logfile_entries,len(bgcolors))],_num_logfile_entries,_num_logfile_entries)
            if self._callback_msg:
                str += self._callback_msg

        # Take a screenshot of the current figure, possibly for future use
        if self.log_mode == "html_anim" or event == "stop" and self.visualize:
            screenshot = self._window.save_figure()
            self._screenshots.append(screenshot)

        # If this update has the lowest validation error for this run, remember 
        # the ideal early stopping point. Oh god this code is so horrible :(
        #
        if stats["valid"] != None:
            id = _num_logfile_entries
            errname  = "error rate" if stats["valid"].has_key('error rate') else "loss"
            errors = [ (stats[fold][errname] if stats[fold] != None else None) for fold in ('train','valid','test')]
            if (not _best_logfile_errors.has_key(id)) or stats["valid"][errname] < _best_logfile_errors[id][1]:
                _best_logfile_errors[id] = errors
        
        # Add the output 'msg' to the log
        str += "<pre style='margin:0;padding:0;font-weight:%s'>%s</pre>\n" % ("bold" if event == "stop" else "normal",msg.strip())

        if event == 'stop':
            # Insert an image or an animation, and close the outer-most DIV element
            if len(self._screenshots) > 0:
                thumbfile = self._screenshots[-1]
                thumburl = "/".join(thumbfile.split('/')[-2:])
                animurl  = thumburl
                if len(self._screenshots) > 1:
                    animfile = os.path.splitext(thumbfile)[0] + '-anim.gif'
                    animurl = "/".join(animfile.split('/')[-2:])
                    inputs = os.path.split(thumbfile)[0] + '/img-%05d-%%02d.png[0-%d]' % (_num_logfile_entries,len(self._screenshots)-1)
                    os.system('convert -delay 30 -coalesce -layers optimize ' + inputs + ' ' + animfile)

                    # Always remove screenshots that are no longer needed
                    for screenshot in self._screenshots[:-1]:
                        os.remove(screenshot)
                
                str += "<div><a href='%s'><img src='%s' border=0/></a></div>\n" % (animurl,thumburl)


            str += "</div>\n"
        
        self._append_to_log(str)

    def _append_to_log(self,str):
        global _curr_logfile
        with open(_curr_logfile,"a") as f:
            f.write(str)
            f.flush()
            f.close()


##############################################################################
#                               WINDOW
##############################################################################


class TrainingReportWindow(Tk.Frame):
    '''
    A window that visualizes the current state of training progress.
    '''
    def __init__(self,trainer,window_size):
        Tk.Frame.__init__(self)
        self._unique_id = 0

        # Set up a 2x2 grid, where each cell will have its own kind of figure
        self.master.rowconfigure(0,weight=1)
        self.master.rowconfigure(1,weight=1)
        self.master.rowconfigure(2,weight=1)
        self.master.columnconfigure(0,weight=1)
        self.master.columnconfigure(1,weight=1)
        self.master.columnconfigure(2,weight=1)
        dpi = 80.0
        self.plots = {}
        if window_size == "compact":
            col0_wd = 300
            col1_wd = 300
            row0_ht = 200
            row1_ht = 100
        else:
            col0_wd = 320
            col1_wd = 770
            row0_ht = 400
            row1_ht = 220

        # Add error plot in top-left cell
        self.plots["errors"] = TrainingReportErrorPlot(self.master,(col0_wd,row0_ht),dpi,trainer.task())
        self.plots["errors"].canvas.get_tk_widget().grid(row=0,column=0,sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        has_input_feat  = trainer.data.Xshape[0] > 1 and trainer.data.Xshape[1]
        has_output_feat = trainer.data.Yshape[0] > 1 and trainer.data.Yshape[1]

        # Input feature grid in top-right cell
        if has_input_feat:
            self.plots["feat_in"] = TrainingReportFeatureGrid(self.master,(col1_wd,row0_ht),dpi,trainer.model,trainer.data.Xshape,"input")
            self.plots["feat_in"].canvas.get_tk_widget().grid(row=0,column=1,columnspan=(1 if has_output_feat else 2),sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        # Output feature grid in top-right-right cell
        if trainer.data.Yshape[0] > 1 and trainer.data.Yshape[1]:
            self.plots["feat_out"] = TrainingReportFeatureGrid(self.master,(col1_wd,row0_ht),dpi,trainer.model,trainer.data.Yshape,"output")
            self.plots["feat_out"].canvas.get_tk_widget().grid(row=0,column=(2 if has_input_feat else 1),columnspan=(1 if has_input_feat else 2),sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        # *Weight* statistics in bottom-left cell
        weights_ref = weakref.ref(trainer.model.weights)
        get_weightmats = lambda event,stats: [bm.as_numpy(abs(layer.W)) for layer in weights_ref()]
        weight_percentiles =  list(100*(1-linspace(0.1,.9,10)**1.5))
        self.plots["wstats"] = TrainingReportPercentiles(self.master,(col0_wd,row1_ht),dpi,get_weightmats,weight_percentiles,True,title="W")
        self.plots["wstats"].canvas.get_tk_widget().grid(row=1,column=0,sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        # *Hidden activity* statistics in bottom-right cell
        get_hidden = lambda event,stats: stats["train"]["H"]
        hidden_percentiles =  list(100*(1-linspace(0.1,.9,10)**1.5))
        ranges = [layer.f.actual_range() for layer in trainer.model._cfg[1:]]
        self.plots["hstats"] = TrainingReportPercentiles(self.master,(col0_wd,row1_ht),dpi,get_hidden,hidden_percentiles,False,ranges=ranges,title="H")

        # For problems with 2D output, draw the target and the reconstruction side by side
        if trainer.data.Yshape[0] > 1 and trainer.data.Yshape[1]:
            if trainer.data["test"].size > 0:
                self.plots["recons_tr"] = TrainingReportReconstructGrid(self.master,(col1_wd,row1_ht),dpi,trainer.data,"train")
                self.plots["recons_tr"].canvas.get_tk_widget().grid(row=1,column=1,rowspan=2,sticky=Tk.N+Tk.S+Tk.E+Tk.W)
                self.plots["recons_te"] = TrainingReportReconstructGrid(self.master,(col1_wd,row1_ht),dpi,trainer.data,"test")
                self.plots["recons_te"].canvas.get_tk_widget().grid(row=1,column=2,rowspan=2,sticky=Tk.N+Tk.S+Tk.E+Tk.W)
            else:
                self.plots["recons_tr"] = TrainingReportReconstructGrid(self.master,(col1_wd,row1_ht),dpi,trainer.data,"train")
                self.plots["recons_tr"].canvas.get_tk_widget().grid(row=1,column=1,columnspan=2,rowspan=2,sticky=Tk.N+Tk.S+Tk.E+Tk.W)

            self.plots["hstats"].canvas.get_tk_widget().grid(row=2,column=0,sticky=Tk.N+Tk.S+Tk.E+Tk.W)
        else:
            self.plots["hstats"].canvas.get_tk_widget().grid(row=1,column=1,sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        self.master.geometry('+%d+%d' % (0,180))
        self.master.title("Training Report")
        self.update()
        self._redraw_interval = 500

    def log(self,event,stats,msg):
        for plot in self.plots.values():
            plot.log(event,stats)
        self.redraw()

    def redraw(self):
        for plot in self.plots.values():
            plot.redraw()
        self.update()
        self.update_idletasks()

    def save_figure(self):
        global _curr_logfile
        global _num_logfile_entries
        dir = ensure_dir(os.path.splitext(_curr_logfile)[0])
        filename = dir + ('/img-%05d-%02d.png'     % (_num_logfile_entries,self._unique_id))
        tempname = dir + ('/img-%05d-%02d-%%s.png' % (_num_logfile_entries,self._unique_id))
        self._unique_id += 1
        
        # First save the individual Figure canvases to files
        pnames = ('errors','feat_in','feat_out','wstats','hstats','recons_tr','recons_te')
        fnames = []
        for pname in pnames:
            if self.plots.has_key(pname):
                fnames.append(tempname % pname)
                plot = self.plots[pname]
                plot.redraw()
                plot.savefig(fnames[-1],dpi=80)

        # Then use ImageMagick to put them together again
        cmd = 'montage'
        for fname in fnames:
            cmd += ' ' + fname
        cmd += ' -tile 2x2 -geometry +0+0 ' + filename
        os.system(cmd)

        # And delete the temporary images of separate parts of the overall figure
        for fname in fnames:
            os.remove(fname)

        return filename




##############################################################################
#                               ERROR PLOT
##############################################################################



class TrainingReportErrorPlot(Figure):

    def __init__(self,master,size,dpi,task):
        Figure.__init__(self,figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=True,linewidth=0)
        FigureCanvas(self,master=master)
        self.master = master
        self._errors = collections.OrderedDict()
        self._dirty = True
        self._task = task
        self.add_subplot(111,axisbg='w')
        
    def log(self,event,stats):
        yaxis = "error rate" if self._task == "classification" else "loss"
        for fold in ('train','valid','test'):
            if stats[fold] == None:
                continue
            # Add the point (epoch,errors) to the plot series for this fold
            x,y = stats["epoch"],stats[fold][yaxis]
            series = self._errors.setdefault(fold,[[],[]])  # list of X values, list of Y values
            series[0].append(x)
            series[1].append(y)
        self._dirty = True

    def redraw(self):
        if self._dirty:
            self._dirty = False
            ax = self.axes[0]
            ax.cla()
            ax.hold("on")
            ax.set_xlabel('epoch')
            ax.set_ylabel("error rate" if self._task == "classification" else "loss")
            colours = [[0.0,0.2,0.8],  # train
                       [0.4,0.1,0.5],  # valid
                       [1.0,0.0,0.0]]  # test
            styles = ['-','--',':']
            minerr,maxerr = 1e10,-1e10
            for series,colour,style in zip(self._errors.items(),colours,styles):
                X,Y = series[1]
                ax.semilogy(X,Y,color=colour,linestyle=style,label=series[0]);
                maxerr = max(maxerr,max(Y))
                minerr = min(minerr,min(Y))
            minerr = max(0.0001 if self._task == "regression" else 0.01,minerr)
            maxerr = max(0.001+minerr,maxerr)

            ax.set_ylim([10**floor(log10(minerr)),10**ceil(log10(maxerr))])
            ax.grid(True,which='major',axis='y',linestyle=':',color=[0.2,0.2,0.2])
            ax.grid(True,which='minor',axis='y',linestyle=':',color=[0.8,0.8,0.8])
            ax.legend()
            ax.set_position([0.125,0.14,.83,.82])
            self.canvas.draw()
    

##############################################################################
#                               FEATURE GRID
##############################################################################


class TrainingReportFeatureGrid(Figure):

    def __init__(self,master,size,dpi,model,featshape,direction='input'):
        Figure.__init__(self,figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=True,linewidth=0)
        FigureCanvas(self,master=master)
        self.master = master
        self._dirty = True
        self._model = model
        self._feat  = None
        self._featrange = None
        self._featshape = featshape
        self._direction = direction
        self._sorted = True
        self._ordering = None
        self.add_subplot(111,axisbg='w')
        
    def log(self,event,stats):
        if self._direction == 'input':
            # Filters going into first layer of hidden units
            W = self._model.weights[0].W
            W = bm.as_numpy(W).copy()
            W = W.reshape(self._featshape + tuple([-1]))
        else:
            # Templates going out of final layer of hidden units
            W = self._model.weights[-1].W
            W = bm.as_numpy(W).copy().transpose()
            W = W.reshape(self._featshape + tuple([-1]))
        self._feat  = W # col[i] contains weights entering unit i in first hidden layer
        self._featrange = (min(W.ravel()),max(W.ravel()))
        self._dirty = True
        if event == 'epoch' and self._sorted and (stats['epoch'] < 5):
            # Sort by decreasing L2 norm
            ranks = [-sum(self._feat[:,:,j].ravel()**2) for j in range(self._feat.shape[2])]
            self._ordering = argsort(ranks)
        if self._ordering != None:
            self._feat = self._feat[:,:,self._ordering]


    def redraw(self):
        if self._dirty:
            self._dirty = False
            feat = self._feat
            self.clf()

            # Convert list of features into a grid of images, fitting the current drawing canvas
            wd,ht = self.canvas.get_width_height()
            zoom = max(1,16//max(feat.shape[0:2]))
            absmax = max(abs(feat.ravel()))
            img = _feat2grid(feat,zoom,1.0,[wd-2,ht-30],vminmax=(-absmax,absmax))

            # Draw the image centered
            x0,y0 = (wd-img.shape[1])/2, (ht-img.shape[0])/2
            self.figimage(img,x0,y0,None,None,cm.gray,zorder=2)

            # Print the range of the colormap we're seeing
            self.text(float(x0)/wd,float(y0+img.shape[0]+5)/ht,'%s features (%.3f,%.3f)' % (self._direction,self._featrange[0],self._featrange[1]),zorder=5)

            self.canvas.draw()
    




##############################################################################
#                               PERCENTILE STATISTICS
##############################################################################


class TrainingReportPercentiles(Figure):

    def __init__(self,master,size,dpi,get_matrices_fn,percentiles,transposed,ranges=None,title=""):
        Figure.__init__(self,figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=True,linewidth=0)
        FigureCanvas(self,master=master)
        self.master = master
        self._dirty = True
        self._get_matrices_fn = get_matrices_fn
        self._P = []
        self._t = percentiles
        self._ranges = ranges
        self._title = title
        self._transposed = transposed
        self.add_subplot(111,axisbg='w')
        
    def log(self,event,stats):
        self._P = []
        matrices = self._get_matrices_fn(event,stats)
        for A in matrices:
            if self._transposed:
                A = A.transpose()
            P = make_matrix_percentiles(A,self._t)  # percentiles over rows first, then percentiles over those last
            self._P.append(P)
        self._dirty = True

    def redraw(self):
        if self._dirty:
            self._dirty = False
            self.clf()

            # Convert list of features into a grid of images, fitting the current drawing canvas
            wd,ht = self.canvas.get_width_height()
            nlayer = len(self._P)
            for k in range(nlayer):
                P = self._P[k]
                if self._ranges != None: 
                    Prange = self._ranges[k]
                else:
                    Prange = [-inf,inf]
                if Prange[0] == -inf: Prange[0] = P.ravel().min()
                if Prange[1] ==  inf: Prange[1] = P.ravel().max()
                P -= Prange[0]
                if Prange[1] != Prange[0]:
                    P /= (Prange[1]-Prange[0])
                P *= 255
                P = minimum(P,255)
                P = maximum(0,P)
                zoom = 6
                img = asarray(P,dtype='uint8')
                img = repeat(img,zoom,axis=0)
                img = repeat(img,zoom,axis=1)

                cellwd = float(wd)/nlayer
                x0 = k*cellwd + (cellwd-img.shape[0])/2
                y0 = (ht-img.shape[0])/2
                self.figimage(img,x0,y0,None,None,cm.gray,zorder=2,vmin=0,vmax=255)

                # Print the range of the colormap we're seeing
                self.text(float(x0)/wd,float(y0+img.shape[0]+5)/ht,'$%s_%d$ (%.2f,%.2f)' % (self._title,k,Prange[0],Prange[1]),zorder=5,size='smaller')

            self.canvas.draw()
    





##############################################################################
#                               RECONSTRUCTION GRID
##############################################################################


class TrainingReportReconstructGrid(Figure):

    def __init__(self,master,size,dpi,data,fold):
        Figure.__init__(self,figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=True,linewidth=0)
        FigureCanvas(self,master=master)
        self.master = master
        self._dirty = True
        self._fold = fold
        self._indices = rnd.sample(arange(data[self._fold].size),minimum(data[self._fold].size,128))#256))
        self._targets = bm.as_numpy(data[self._fold].Y[self._indices,:]).transpose().reshape(data.Yshape + tuple([len(self._indices)]))
        self._outputs = None
        self._outshape = data.Yshape
        self._outrange = data.Yrange
        self.add_subplot(111,axisbg='w')
        
    def log(self,event,stats):
        Z = stats[self._fold]["H"][-1][self._indices]
        self._outputs = Z.transpose().reshape(self._outshape + tuple([-1])) # format outputs as stack of (ht x wd) matrices
        self._dirty = True

    def redraw(self):
        if self._dirty:
            self._dirty = False
            self.clf()

            # Concatenate outputs and targets, side-by-side, and arrange
            # into a grid of images, fitted to the current drawing canvas
            pairs = hstack([self._outputs,self._targets])
            wd,ht = self.canvas.get_width_height()
            img = _feat2grid(pairs,zoom=1.0,gamma=1.0,bbox=[wd-2,ht-25],vminmax=self._outrange)

            # Draw the grid image centered
            x0,y0 = (wd-img.shape[1])/2, (ht-img.shape[0])/2
            self.figimage(img,x0,y0,None,None,cm.gray,zorder=2,vmin=0,vmax=255)

            self.text(float(x0)/wd,float(y0-15)/ht,self._fold,zorder=5)
            self.canvas.draw()
    



############################################################################



def _feat2grid(feat,zoom=1.0,gamma=1.0,bbox=[300,300],want_transpose=False,outframe=True,pad=1,vminmax=None):
    '''
    Like feature2img, except returns a single image with
    a grid layout, instead of a list of individual images
    '''
    ht,wd,n = feat.shape
    wd *= zoom
    ht *= zoom
    opad = pad if outframe else 0
    bbwd,bbht = bbox
    maxcols = int(bbwd-2*opad+pad)//(wd+pad)
    maxrows = int(bbht-2*opad+pad)//(ht+pad)
    if want_transpose: numrows,numcols = _gridsize(maxrows,maxcols,n)
    else:              numcols,numrows = _gridsize(maxcols,maxrows,n)

    # Convert 3D array features into a 3D array of images
    img = _feat2img(feat[:,:,:min(n,numrows*numcols)],zoom=zoom,gamma=gamma,vminmax=vminmax)

    # Pull each image out and place it within a grid cell, leaving space for padding
    framecolor = 0
    numchannels = img[0].shape[2]
    grid = zeros([numrows*(ht+pad)-pad+2*opad,numcols*(wd+pad)-pad+2*opad,numchannels],dtype=ubyte) + framecolor
    for i in range(len(img)):
        ri = (i % numrows) if     want_transpose else floor(i / numcols)
        ci = (i % numcols) if not want_transpose else floor(i / numrows)
        row0 = ri * (ht+pad) + opad 
        col0 = ci * (wd+pad) + opad
        grid[row0:row0+ht,col0:col0+wd,:] = img[i]

    if numchannels==1:
        grid.shape = grid.shape[0:2]  # matplotlib's figimage doesn't like extra dimension on greyscale images
    return grid
    


def _gridsize(max1,max2,n):
    '''Calculates the dimensions of the grid, in 'cells', for n items'''
    max1 = float(max1); max2 = float(max2); n = float(n)
    if max1 <= 1:
        return (0,0)
    num1 = min(max1,n)
    num2 = 1.0
    while ceil(n/(max1-1)) <= max1-1 and ceil(n/(max1-1)) <= max2 and ceil(n/(max1-1)) > num2:
        num1 -= 1
        num2 = ceil(n/num1)
    if num1 == 0:
        return (0,0)
    num2 = min(max2,ceil(n/num1))
    return (int(num1),int(num2))


def _feat2img(feat,zoom,gamma,vminmax=None):
    '''
    Given an NxMxT stack of features (filters), splits it
    into a list of NxM images.
    '''
    if vminmax == None:
        vrange = (min(feat.ravel()),max(feat.ravel()))
    ht,wd,n = feat.shape # n = number of features
    img = []
    for i in range(n):
        I = reshape(feat[:,:,i],(ht,wd,1))
        if vminmax != None:
            I -= vminmax[0]
            I *= 1./(vminmax[1]-vminmax[0])
            pass
        if gamma != 1.0:
            I = pow(I,gamma)
        I *= 255
        I = uint8(minimum(255,maximum(0,I)))
        if zoom != 1:
            I = repeat(I,zoom,axis=0)
            I = repeat(I,zoom,axis=1)
        img.append(I)
    return img



############################################################################

# Input: n-dimensional matrix A and and m percentile thresholds in list t. 
# Returns an n-dimensional matrix P, where each dimension has size m, 
# where element P[j1,j2,...jn] is 
#       the t[j1]-percentile along dimension 1
#    OF the t[j1]-percentile along dimension 2
#       ...
#    OF the t[jn]-percentile along dimension n
#
# So if A were 8x6 (2 dimensions) and t=[1,10,50] then P is a 3x3 matrix
# where P[i,j] = percentile(percentile(A,t[j],axis=1),t[i],axis=0)
#
def make_matrix_percentiles(A,t):
    n,m = A.ndim,len(t)
    if n == 0:
        return A # any percentile of a single number is just that number

    # First collect all the t[j] percentiles of A along the last dimension
    # Result: list of length m, each item being an (n-1)-dimensional matrix
    Ap = percentile(A,t,n-1)

    # For each Ap[j] matrix, compute its (n-1)-dimensional P matrix 
    # (sides of length m) using recursion
    Pp = []
    for Apj in Ap:
        Pp.append(make_matrix_percentiles(Apj,t))

    # Finally, stack each Pp
    newshape = tuple([m]) + Pp[0].shape
    P = ndarray(newshape,dtype=A.dtype)
    for j in range(m):
        P[j] = Pp[j]

    return P


#########################################################################
# Set up default logging config

_log = logging.getLogger()

def setup_logging(filename,clear=False):

    if clear:
        with open(filename, 'w'):
            pass  # clear the log file if it already exists

    log_file = logging.FileHandler(filename)
    log_file.setLevel(logging.DEBUG)

    log_console = logging.StreamHandler(sys.stdout)
    log_console.setLevel(logging.DEBUG)

    _log.setLevel(logging.DEBUG)
    _log.addHandler(log_file)
    _log.addHandler(log_console)

setup_logging('basic_learn.log',clear=True)
