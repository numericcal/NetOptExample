# NetOptExample

This demo contains the result of running Numericcal NetOptimizer on Mobilenet trained using CIFAR10 dataset.
The NetOptimizer automatically remove redundant expressiveness in the network, resulting in a much smaller/faster model without sacrificing accuracy.

## Included Mobilenet Models 
There are two models in the repo: the original mobilenet trained to 92.17% accuracy and a pruned version at 92.37% accuracy (improved accuracy due to regularizing effect of pruning). 
The NetOptimizer actually can produce even smaller models whose accuracy is lower than the original, they are not included in this demo repo. 
The pruned version is about 10x smaller than the original model and depending on the underlying libraries, can be 7-8x faster.


## Running and Testing the Models
To run the example models, you need python3 and pytorch. 

In a python 3 environment, go to the root directory (NetOptExample/) and run:</br> 
pip install -r requirements.txt</br>
python test_models.py</br>

The script parses the included models and you can test their performance on CIFAR test datasets. 

Note: the first time you run the script, it will take some time to download the CIFAR dataset before you can initiate the test, also the testing on CIFAR can take some of time in a CPU only environment.

 
