# Dark Machines LHC dark matter challenge
In this work we try to find a method to detect _any_ BSM model.
Currently, we will use a DeepSet AutoEncoder.


The file structure is given by
```
.
├── data
│   ├── interim      <- data for training the networks
│   ├── processed    <- final data for figures
│   └── raw          <- data from zenodo
├── docs
│   ├── channels.pdf <- description of the selection cuts and BSM models
│   └── dataset.pdf  <- description of the data set
├── LICENSE
├── Makefile         <- used to run the project
├── model
├── notebooks
│   ├── 01-bo-examine-datastructure.ipynb
│   └── 02-bo-intial-network-setup.ipynb
├── README.md
├── references
├── reports
│   └── figures
└── src
    ├── data
    │   └── ConvertToNpz.py <- Converts the raw data to numpy files
    ├── features
    ├── __init__.py
    ├── models
    │   ├── __init__.py
    │   ├── train.py
    │   └── utitlities.py
    └── visualilzation
 ```
