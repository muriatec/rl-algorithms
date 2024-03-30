## Dependencies
- python 3.11.3
- torch 2.0.1+cu118
- tensorboard 2.12.3
- gym 0.26.2

## Train
To train the agent, use the following command:
```
python ddqn.py --train --env_name PongNoFrameskip-v4
```

## Test
To test the trained model, use the following command:
```
python ddqn.py --test --env_name PongNoFrameskip-v4 --model_path out\PongNoFrameskip-v4-run1\model_last.pkl
```