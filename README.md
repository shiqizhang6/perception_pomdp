# perception_pomdp
The codebase for the paper:
["Multi-modal Predicate Identification using Dynamically Learned Robot Controllers"](http://www.cs.utexas.edu/~pstone/Papers/bib2html-links/IJCAI18-saeid.pdf)

## Installation
Please make sure to use Python 2.X,
### Libraries
`$ pip install scikit-learn`

### POMDP Solver
Also, please download and compile ["this"](https://github.com/AdaCompNUS/sarsop)
 pomdp solver. Then provide the 'pomdpsol' path to the `pathlist` ["here"](https://github.com/shiqizhang6/perception_pomdp/blob/shiqi/ijcai_dataset_testing_new/src_py/simulator_ijcai.py#L461).

## Run the experiments:
Please download the repo and make sure you are in the branch `shiqi/ijcai_dataset_testing_new`:
`$ git clone https://github.com/shiqizhang6/perception_pomdp.git `
`$ cd perception-pomdp`
`$ git checkout shiqi/ijcai_dataset_testing_new`
And then create a folder for storing the classifiers.
`$ cd src_py`
`$ mkdir pomdp_classifiers`

### Training classifers
In order to train classifiers,
`python classifier_ijcai2016.py`
It will take around 2 hours and half to train all classifiers.

### Figure 7 of the paper:
`python simulator_ijcai.py`
### Figure 6 of the paper:
`git checkout 4e8d4`
`python simulator_ijcai.py`  

## Important files:
`constructor.py`  builds the pomdp model in the form of a file called `model.pomdp`. To better understand how a pomdp file is formatted, check ["here"](https://www.pomdp.org/code/pomdp-file-spec.html). </br>
`simulator_ijcai.py` simulates the experiments. </br>
`policy.py` in charge of generating a policy file (`model.policy`) and parsing it.
