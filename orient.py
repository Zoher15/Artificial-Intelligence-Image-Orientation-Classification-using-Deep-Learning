#!/usr/bin/env python
#
from orient_knn import *
from orient_nnet import *
from orient_adaboost import *
import sys

def main():
    (parameter, t_fname, model_fname, model) = sys.argv[1:]
    if (model == "nearest"):
        knn_main(parameter, t_fname, model_fname)
    elif (model == "nnet"):
        neural_net(parameter, t_fname, model_fname)
    elif (model == "adaboost"):
        adaboost(parameter, t_fname, model_fname)
    elif (model == "best"):
        neural_net(parameter, t_fname, model_fname)

if __name__== "__main__":
  main()
