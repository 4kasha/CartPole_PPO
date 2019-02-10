[//]: # (Image References)


# CartPole via PPO - PyTorch implementation

## Description

You will train an agent in CartPole-v0 (OpenAI Gym) environment via Proximal Policy Optimization (PPO) algorithm with GAE.  

A reward of **+1** is provided for every step taken, and a reward of **0** is provided at the termination step. The state space has **4** dimensions and contains the cart position, velocity, pole angle and pole velocity at tip. 
Given this information, the agent has to learn how to select best actions. 
Two discrete actions are available, corresponding to:

- **`0`** - 'Push cart to the left'
- **`1`** - 'Push cart to the right'

For more details, see the [wiki](https://github.com/openai/gym/wiki/CartPole-v0).

For training results and making animation, see [train.ipynb](train.ipynb). 

## Dependencies

- Python 3.6
- PyTorch 0.4.0
- OpenAI Gym 0.10.5 (for Installation, see [this](https://github.com/openai/gym#id8).)

## References

- [J. Schulman, et al. "Proximal Policy Optimization Algorithms"][ref1]
- [J. Schulman, et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation"][ref2]
- [Deep Reinforcement Learning: Pong from Pixels][ref3], ([Japanese][ref3-1])
- [PPO - PyTorch][ref4]
- [DeepRL][ref5] 

[ref1]: https://arxiv.org/pdf/1707.06347.pdf
[ref2]: https://arxiv.org/abs/1506.02438
[ref3]: http://karpathy.github.io/2016/05/31/rl/
[ref3-1]: https://postd.cc/deep-reinforcement-learning-pong-from-pixels-1/
[ref4]: https://github.com/dai-dao/PPO-Pytorch
[ref5]: https://github.com/ShangtongZhang/DeepRL