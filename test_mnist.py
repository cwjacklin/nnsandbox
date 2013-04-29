from NeuralNet import *
from DataSet import *
from Util import *
from BigMat import *
from TrainingRun import *

def main():
    #set_backend("numpy")   # Run code on CPU with numpy backend
    set_backend("gnumpy")   # Run code on GPU with GNUMPY/CUDAMAT backend

    ######################################################
    # Load MNIST dataset
    tic()
    data = load_mnist()
    print ("Data loaded in %.1fs" % toc())

    ######################################################
    # Create a neural network with matching input/output dimensions
    cfg = NeuralNetCfg(L1=1e-6,init_scale=0.05)
    cfg.input(data.Xshape)
    cfg.hidden(800,"logistic",dropout=0.5)
    cfg.hidden(800,"logistic",dropout=0.25)
    cfg.output(data.Yshape,"softmax")

    model = NeuralNet(cfg)

    ######################################################
    # Rescale the data to match the network's domain/range
    data.rescale(model.ideal_domain(),model.ideal_range())

    ######################################################
    # Train the network
    report_args = { 'verbose'   : True,
                    'interval'  : 5,       # how many epochs between progress reports (larger is faster)
                    'window_size' : "compact",
                    'visualize' : True}

    trainer = TrainingRun(model,data,report_args,
                          learn_rate=2,
                          learn_rate_decay=.995,
                          momentum=[(0,.5),(400,0.9)],
                          batchsize=64)

    print "Memory available after data loaded:", memory_info(gc=True)

    tic()
    trainer.train(1000)  # train for 1000 epochs
    print ("Training took %.1fs" % toc())

    #####################################################
    
    raw_input()


main()


