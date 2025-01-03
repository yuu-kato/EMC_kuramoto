Simulation codes and the EMC estimation results for ``Bayesian estimation of coupling strength and heterogeneity in a coupled oscillator model from macroscopic observations'' by Y Kato, S Kashiwamura, E Watanabe, M Okada, and H Kori.

# Data generation and EMC simulation
You can perform the simulation using either your local environment or HPCs.
## If you use your local environment
### How to reproduce the environment
The C++ environment with necessary packages can be reproduced by Docker.
- With VS Code ``Dev Containers'' extension and Docker Desktop
  - Simply download this repository
  - Open the directory as container, using `Dev Containers: Open Folder in Container...` command.
    - You may need to reload the window after container is built.
  - You can run the scripts in the container.
- With Docker Desktop
  - Use `Dockerfile` to reproduce the environment.
### How to run the code

## If you use HPCs
### How to reproduce the environment
The C++ environment with necessary packages can be reproduced by Apptainer.

### How to run the code


Results stored in `data` directory were calculated using the scripts named `calc_something.py`.  
You can use them to reproduce the stored data and to generate more results.  
Note however that we ran the scripts on HPC with Apptainer.  

# Plotting the figures
### How to reproduce the environment
The python environment with necessary packages can be reproduced by Docker.
- With VS Code ``Dev Containers'' extension and Docker Desktop
  - Simply download this repository
  - Open the directory as container, using `Dev Containers: Open Folder in Container...` command.
    - You may need to reload the window after container is built.
  - You can run the scripts in the container.
- With Docker Desktop
  - Use `Dockerfile` to reproduce the environment.
### How to run the code
Run the scripts named `figX_somthing.py`.  
Most of the necessary data are stored in `data` directory, and loaded by the scripts.  
They have flag variables `if_show` and `if_save`. Change their values if necessary.

If you have any question about running the codes, please contact Yusuke Kato.
