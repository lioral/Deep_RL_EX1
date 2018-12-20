# Deep Reinforcement Learning - Exercise GYM Taxi and Acrobot #

All files for the submission are attached.  
clone all files with:  
```
$ git clone https://github.com/LiorAl/Deep_RL_EX1
```

**DRL_Report.pfg** -> Exercise report documente   


## Taxi DQN section ##
**DQN_train.py** -> training script for the taxi agent.   
**DQN_eval.py** -> Load trained Taxi agent and evaluate performance over 1000 episodes.   
**Taxi-v2_model_DQN.pt** -> Model saved weights. 

For load and run the trained agent use:  
```
$ python DQN_eval.py  
```

## Taxi AC section ##
**AC_train.py** -> training script for the taxi agent.  
**AC_eval.py** -> Load trained Taxi agent and evaluate performance over 1000 episodes.  
**Taxi-v2_model_AC.pt** -> Model saved weights. 

For load and run the trained agent use:  
```
$ python AC_eval.py 
```

## Acrobot section ##
**Acrobot_training.py** -> training script for the acrobot agent.  
**utils.py** -> utilities function for the acrobot DQN architecture.   
**Acrobot_Wieghts.pt** -> Model saved weights for best performance in the demonstration.  
**Acrobot_Demo.py** -> Load trained acrobot agent and evaluate performance over 10 episodes.  
**Acrobot.avi** -> 10 min clip of acrobot agent in action.

For load and run the trained agent use:  
```
$ python Acrobot_Demo.py  
```
