# Information-PC-Paper
Code associated with the paper, ["Higher-Order Spatial Information for Self-Supervised Place Field Learning."](https://arxiv.org/abs/2407.06195) Some important files are:

-- [pc_info.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/pc_info.py) contains functions for spatial information rates, including Skaggs' spatial information rates I_sec(A), I_spike(A) and higher-order spatial information rates I_sec_joint(A, B), and I_spike_joint(A, B). These are described in detail in Section 2 of the paper. 

-- [main.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/main.py) contains a simple instance of training a model for self-supervised learning via spatial information.

-- [model.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/model.py) contains the construction of the RNN class used to build models.

-- [inspect.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/inspect.py) contains an example of model inspection, i.e obtaining rate maps, place cell scores, etc.

-- [decoding_functions.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/decoding_functions.py) contains functions to perform neural decoding using place cell activations. Options include leave-one-out classification and support-vector-machine quadrant classification. 

-- [placescores.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/placescores.py) contains a function to calculate place cell score. This is detailed
Section 3.2 of the paper. 

-- [place_cells.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/place_cells.py) contains the place cell class, used to embed the initial position into the RNN. 

-- [grid_cells.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/grid_cells.py) contains the grid cell class, (optionally) used to force grid cell activations in the hidden layer. None of the models/results in the paper rely upon this, but it may be explored by using -- hardcoded_gcs True in a model.

-- [sensitivity.py](https://github.com/JaredDeightonUTK/Information-PC-Paper/blob/main/sensitivity.py) contains a script for running the sensitivity analysis and recording saving the results.

Please cite this code via: 

@article{deighton2024higher,
  title={Higher-Order Spatial Information for Self-Supervised Place Cell Learning},
  author={Deighton, Jared and Mackey, Wyatt and Schizas, Ioannis and Boothe Jr, David L and Maroulas, Vasileios},
  journal={arXiv preprint arXiv:2407.06195},
  year={2024}
}
