## Dependencies
- python 3.7.0
- torch 1.13.1
- mujoco-py 1.50.1.0
- gym 0.26.2

## Train
To train the agent, use the following command
```
python ddpg.py --train --env_name Hopper-v2
```

## Test
To test the trained model, use the following command
```
python ddpg.py --test --env_name Hopper-v2 --model_path agent_1000
```