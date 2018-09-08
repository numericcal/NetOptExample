# NetOptExample

This demo contains the result of running Numericcal NetOptimizer on Mobilenet trained using CIFAR10 dataset.
The NetOptimizer performs size/accuracy trade-off automatically. This process often uncovers redundant expressiveness in the network, 
which can then be substantially reduced in size without losing accuracy. Such an example is given in this repo.

## Included Mobilenet Models 
There are two models in the repo: the original mobilenet trained to 92.17% accuracy and a pruned version at 92.37% accuracy (improved accuracy due to regularizing effect of pruning). 
The pruned network is about 10x smaller than the original model and depending on the underlying libraries, can be 7-8x faster.
As mentioned earlier, the NetOptimizer actually explore the size/accuracy trade-off and it has generated even smaller/faster models at the cost of lower accuracy. For simplicity's sake, they are not included in this demo repo. 


## Running and Testing the Models
To run the example models, you need python3 and pytorch. 

In a python 3 environment, go to the root directory (NetOptExample/) and run:</br> 
pip install -r requirements.txt</br>
python test_models.py</br>

The script parses the included models and you can test their performance on CIFAR test datasets. 

Note: the first time you run the script, it will take some time to download the CIFAR dataset before you can initiate the test, also the testing on CIFAR can take some time in a CPU only environment.


 
