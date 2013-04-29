from NeuralNet import *
from DataSet import *
from Util import *
from BigMat import *
from TrainingRun import *
import sys

def main(viz=False):

    tic()
    data = load_mnist()
    print ("Data loaded in %.1fs" % toc())

    # Create a neural network with matching input/output dimensions
    #
    cfg = NeuralNetCfg(L1=1e-6,init_scale=0.05)
    cfg.input(data.Xshape)
    cfg.hidden(800,"logistic",dropout=0.5)
    cfg.hidden(800,"logistic",dropout=0.25)
    cfg.output(data.Yshape,"softmax")

    model = NeuralNet(cfg)

    # Rescale the data to match the network's domain/range
    #
    data.rescale(model.ideal_domain(),model.ideal_range())

    # Train the network
    #
    report_args = { 'verbose'   : True,
                    'interval'  : 5,       # how many epochs between progress reports (larger is faster)
                    'window_size' : "compact",
                    'visualize' : viz}

    trainer = TrainingRun(model,data,report_args,
                          learn_rate=2,
                          learn_rate_decay=.995,
                          momentum=[(0,.5),(400,0.9)],
                          batchsize=64)

    print "Memory available after data loaded:", memory_info(gc=True)

    tic()
    trainer.train(100)  # train for several epochs
    print ("Training took %.1fs" % toc())

###################################
def print_usage():
    print "Usage: test_mnist.py <backend> [viz]"
    print "where <backend> is one of:"
    print "  cpu:  use the CPU"
    print "  gpu:  use the GPU"
    print "and optional argument 'viz' will show a window that"
    print "visualizes the filters learned by the network."
    sys.exit()


if not len(sys.argv) in (2,3): print_usage()

backend = sys.argv[1]
if   backend == "cpu": set_backend("numpy")
elif backend == "gpu": set_backend("gnumpy")
else: print_usage()

viz = False
if len(sys.argv) > 2:
    if sys.argv[2] != "viz": print_usage()
    viz = True

main(viz)


