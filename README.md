# A continuous Deep Reinforment Learning case study : Unity Reacher environment
Clone this Git:
```
git clone https://github.com/GabrielLinear/PPO.git
```
Set-up your environment like this [GitHub Pages](https://github.com/udacity/Value-based-methods#dependencies).
Previous to the operation ***pip install .*** , you will have to install torch 0.4 then uninstall it and install the torch version you want.

Then you will have to install tensorboard to access to the log files.
```
pip uninstall tensorboard
pip uninstall tensorboardX
conda install tensorboard
```

You can re-train the agent with the algorithm by launching the notebook, then on the terminal in the cloned git hub folder :
```
tensorboard --logdir=Tensorboard-files
```

### Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space and the action space is continuous. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved when the agent reach an average score of +30 over 100 consecutive episodes

### Results of the algorithms




For more information you can check the [report](https://github.com/GabrielLinear/PPO/blob/main/Report.pdf). 
