import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import mlx.core as mx

from ppo_rl.agent import PPOAgent
from ppo_rl.buffer import RolloutBuffer

def main():
    # Hyperparams
    env_name = "CartPole-v1"
    hidden_sizes = [64, 64]
    lr = 3e-4
    gamma = 0.99
    clip_ratio = 0.2
    
    epochs = 50
    steps_per_epoch = 4000
    ent_coef = 0.01  # Slight entropy regularization for exploration
    
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Initialize Agent and Buffer
    agent = PPOAgent(obs_dim, act_dim, hidden_sizes=hidden_sizes, lr=lr, clip_ratio=clip_ratio, ent_coef=ent_coef)
    buf = RolloutBuffer(steps_per_epoch, obs_dim, act_dim, gamma=gamma)
    
    obs, _ = env.reset(seed=42)
    ep_ret, ep_len = 0, 0
    
    # Logs
    avg_rets = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_rets = []
        
        for t in range(steps_per_epoch):
            # Get action from agent
            a, v, logp = agent.ac.step(mx.array(obs))
            
            next_obs, rew, terminated, truncated, _ = env.step(a)
            ep_ret += rew
            ep_len += 1
            
            # Save to buffer
            buf.store(obs, a, rew, v, logp)
            
            obs = next_obs
            
            timeout = ep_len == 500 # CartPole-v1 specific
            terminal = terminated or truncated or timeout
            epoch_ended = t == steps_per_epoch - 1
            
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    # Cut off
                    _, v, _ = agent.ac.step(mx.array(obs))
                else:
                    v = 0
                
                buf.finish_path(v)
                
                if terminal:
                    # only save return if episode ended normally
                    epoch_rets.append(ep_ret)
                
                obs, _ = env.reset()
                ep_ret, ep_len = 0, 0
        
        # Update PPO
        data = buf.get()
        pi_loss, v_loss, ent, _ = agent.update(data)
        
        # Log
        avg_ret = np.mean(epoch_rets) if epoch_rets else 0
        avg_rets.append(avg_ret)
        print(f"Epoch: {epoch+1}/{epochs}, Return: {avg_ret:.2f}, VL: {v_loss:.4f}, Ent: {ent:.4f}, Time: {time.time()-start_time:.1f}s")
        
        if avg_ret > 475:
            print("Solved!")
            # break # Let's run full epochs to see stability
            
    env.close()
    
    # Plotting
    plt.figure()
    plt.plot(avg_rets)
    plt.title("PPO CartPole-v1 Training")
    plt.xlabel("Epochs")
    plt.ylabel("Average Return")
    plt.grid(True)
    plt.savefig("ppo_cartpole.png")
    print("Saved ppo_cartpole.png")

if __name__ == "__main__":
    main()
