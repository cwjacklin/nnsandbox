from Report import *
from BigMat import sync_backend,garbage_collect,memory_info

class TrainingRun(object):
    '''
    Trains a given model by stochastic gradient descent.
    '''
    def __init__(self,model,data,report_args={},
                 learn_rate=1.0,learn_rate_decay=.995,momentum=0.0,
                 batchsize=64,epochs=3000):
        
        self.model = model
        self.data  = data

        self.learn_rate_max   = learn_rate
        self.learn_rate       = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.momentum_schedule= momentum
        self.momentum         = 0.0
        self.batchsize        = batchsize
        self.batches          = data.train.make_batches(batchsize)
        self.epochs           = epochs

        # wstep is pre-allocated memory for storing gradient matrices
        self.wstep      = model.make_weights()
        self.wstep_prev = model.make_weights() if momentum else None
        self.epoch = 0
       
        if report_args['verbose']: self.log = TrainingReport(self,**report_args) 
        else:                      self.log = lambda event: 0  # do nothing

    def train(self,epochs_this_call=None):
        '''
        Train the current model up the the maximum number of epochs.
        '''
        model,weights    = self.model,self.model.weights
        wstep,wstep_prev = self.wstep,self.wstep_prev

        self.log('start')

        # Outer loop over epochs
        last_epoch = self.epochs if epochs_this_call == None else (self.epoch+epochs_this_call)
        for self.epoch in xrange(self.epoch+1,last_epoch+1):
            self.batches.shuffle()

            # Compute momentum for this epoch
            self._update_momentum()

            # Inner loop over one shuffled sweep of the data
            for batch in self.batches:
                # Add Nesterov look-ahead momentum, before computing gradient
                if self.momentum:
                    weights.step_by(wstep_prev,self.momentum)

                # Compute gradient, storing it in wstep
                model.grad(batch,out=wstep)
                
                if self.momentum:
                    # Add momentum to the step, then adjust the weights
                    wstep *= -self.learn_rate
                    weights += wstep
                    wstep,wstep_prev = wstep_prev,wstep  # move wstep into wstep_prev by swapping arrays
                else:
                    weights.step_by(wstep,alpha=-self.learn_rate)

                # Apply any model constraints, like norm of weights
                model.apply_constraints()

            self.learn_rate *= self.learn_rate_decay
            self.log('epoch')

        self.log('stop')
        sync_backend()    # make sure all gpu operations are complete


    def _update_momentum(self):
        if not isinstance(self.momentum_schedule,list):
            self.momentum = float(self.momentum_schedule)
            return

        n = len(self.momentum_schedule)
        if n == 1:
            epoch0,m0 = self.momentum_schedule[0]
            if self.epoch >= epoch0:
                self.momentum = m0
            return

        for i in range(n-1):
            epoch0,m0 = self.momentum_schedule[i]
            epoch1,m1 = self.momentum_schedule[i+1]
            assert(epoch0 < epoch1)
            if self.epoch >= epoch0 and self.epoch <= epoch1:
                t = float(self.epoch - epoch0) / (epoch1 - epoch0)
                self.momentum = m0 + t*(m1-m0)
                return

    def task(self):
        if self.model._loss_type == "nll":
           return "classification" 
        return "regression"