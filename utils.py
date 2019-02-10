import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collect_trajectories(envs, policy, rollout_length=200):
    """collect trajectories for a parallelized parallelEnv object
    
    Returns : Shape
    ======
    log_probs_old (tensor)   :  (rollout_length*n,)
    states (tensor)          :  (rollout_length*n, envs.observation_space.shape[0])
    actions (tensor)         :  (rollout_length*n,)
    rewards (list,np.array)  :  (rollout_length, n)  --> for advs
    values (list,np.array)   :  (rollout_length, n)  --> for advs
    dones (list,np.array)    :  (rollout_length, n)  --> for advs
    vals_last (list,np.array):  (n,)                 --> for advs
    """
    n=len(envs.ps)         # number of parallel instances

    log_probs_old, states, actions, rewards, values, dones = [],[],[],[],[],[]

    obs = envs.reset()
    
    for t in range(rollout_length):
        
        batch_input = torch.from_numpy(obs).float().to(device)
        traj_info = policy.act(batch_input)

        log_prob_old = traj_info['log_pi_a'].detach()
        action = traj_info['a'].cpu().numpy()
        value = traj_info['v'].cpu().detach().numpy()
        
        obs, reward, is_done, _ = envs.step(action)
        
        if is_done.any():
            if t < 199:
                idx = np.where(is_done==True)
                reward[idx] = 0

        log_probs_old.append(log_prob_old) # shape (rollout_length, n)
        states.append(batch_input)         # shape (rollout_length, n, envs.observation_space.shape[0])
        actions.append(action)             # shape (rollout_length, n)
        rewards.append(reward)             # shape (rollout_length, n)
        values.append(value)               # shape (rollout_length, n)
        dones.append(is_done)              # shape (rollout_length, n)
    
    log_probs_old = torch.stack(log_probs_old).view(-1,)   
    states = torch.stack(states)
    states = states.view(-1,envs.observation_space.shape[0])
    actions = torch.tensor(actions, dtype=torch.long, device=device).view(-1,)
    
    obs = torch.from_numpy(obs).float().to(device)
    traj_info_last = policy.act(obs)
    vals_last = traj_info_last['v'].cpu().detach().numpy()

    return log_probs_old, states, actions, rewards, values, dones, vals_last

def random_sample(inds, minibatch_size):
    inds = np.random.permutation(inds)
    batches = inds[:len(inds) // minibatch_size * minibatch_size].reshape(-1, minibatch_size)
    for batch in batches:
        yield torch.from_numpy(batch).long()
    r = len(inds) % minibatch_size
    if r:
        yield torch.from_numpy(inds[-r:]).long()