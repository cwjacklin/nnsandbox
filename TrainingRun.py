from Report import *
from BigMat import sync_backend,garbage_collect,memory_info

class TrainingRun(object):
    '''
    Trains a given model by stochastic gradient descent.
    '''
    def __init__(self,model,data,report_args={},
                 learn_rate=1.0,learn_rate_decay=.995,momentum=0.0,slowness=0.0,
                 batchsize=64,epochs=3000):
        
        self.model = model
        self.data  = data

        self.batchsize        = batchsize
        self.batches          = data.train.make_batches(batchsize)
        self.epochs           = epochs
        self.epoch = 0

        self.learn_rate_schedule = learn_rate
        self.learn_rate = self._calc_scheduled_rate(learn_rate,0.0)
        self.learn_rate_decay = learn_rate_decay
        self.momentum_schedule= momentum
        self.momentum = self._calc_scheduled_rate(momentum,0.0)
        self.slowness_schedule = slowness
        self.slowness = self._calc_scheduled_rate(slowness,0.0)

        # wstep is pre-allocated memory for storing gradient matrices
        self._wgrad = model.make_weights()
        self._wstep = model.make_weights() if momentum else None
       
        if report_args['verbose']: self.log = TrainingReport(self,**report_args) 
        else:                      self.log = lambda event: 0  # do nothing

    def train(self,epochs_this_call=None):
        '''
        Train the current model up the the maximum number of epochs.
        '''
        model,weights = self.model,self.model.weights
        wgrad,wstep   = self._wgrad,self._wstep

        model.apply_constraints()

        self.log('start')

        # Outer loop over epochs
        last_epoch = self.epochs if epochs_this_call == None else (self.epoch+epochs_this_call)
        for self.epoch in xrange(self.epoch+1,last_epoch+1):
            self.batches.shuffle()

            # Compute learning rate and momentum for this epoch
            self.learn_rate  = self._calc_scheduled_rate(self.learn_rate_schedule,self.learn_rate)
            self.learn_rate *= self.learn_rate_decay**(self.epoch-1)
            self.momentum    = self._calc_scheduled_rate(self.momentum_schedule,self.momentum)
            self.slowness    = self._calc_scheduled_rate(self.slowness_schedule,self.slowness)

            # Inner loop over one shuffled sweep of the data
            for batch in self.batches:
                if self.momentum:
                    # Add Nesterov look-ahead momentum, before computing gradient
                    wstep *= self.momentum
                    weights += wstep

                    # Compute gradient, storing it in wstep
                    model.grad(batch,out=wgrad)
                
                    # Add momentum to the step, then adjust the weights
                    wgrad *= -self.learn_rate;
                    weights += wgrad
                    wstep   += wgrad
                else:
                    model.grad(batch,out=wgrad)
                    weights.step_by(wgrad,alpha=-self.learn_rate)

                # Apply any model constraints, like clipping norm of weights
                model.apply_constraints()

            self.log('epoch')

        self.log('stop')
        sync_backend()    # make sure all gpu operations are complete


    def _calc_scheduled_rate(self,schedule,rate):
        if not isinstance(schedule,list):
            return float(schedule)

        n = len(schedule)
        if n == 1:
            epoch0,m0 = schedule[0]
            if self.epoch >= epoch0:
                return m0
            return rate

        if self.epoch < schedule[0][0]:
            return rate

        for i in range(n-1):
            epoch0,m0 = schedule[i]
            epoch1,m1 = schedule[i+1]
            assert(epoch0 < epoch1)
            if self.epoch >= epoch0 and self.epoch <= epoch1:
                t = float(self.epoch - epoch0) / (epoch1 - epoch0)
                return m0 + t*(m1-m0)

        if self.epoch < schedule[0][0]:
            return schedule[0][1]
        return schedule[-1][1]

    def task(self):
        if self.model._loss_type == "nll":
           return "classification" 
        return "regression"