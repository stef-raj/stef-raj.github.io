import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from .networks import ActorCritic

def categorical_logp(logits, act):
    # act: (B,)
    # logits: (B, A)
    logp_all = nn.log_softmax(logits, axis=-1)
    return mx.take_along_axis(logp_all, act[..., None], axis=-1).squeeze(-1)

def ppo_loss_fn(model, obs, act, adv, ret, old_logp, clip_ratio, ent_coef):
    # 1. PI Loss
    logits = model.pi(obs)
    logp = categorical_logp(logits, act)
    
    # Calculate entropy for exploration bonus
    # Entropy of categorical: -sum(p * log p)
    probs = mx.softmax(logits, axis=-1)
    log_probs = nn.log_softmax(logits, axis=-1)
    entropy = -mx.sum(probs * log_probs, axis=-1).mean()
    
    # Ratio
    ratio = mx.exp(logp - old_logp)
    
    # Clipped Surrogate
    surr1 = ratio * adv
    surr2 = mx.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    pi_loss = -mx.minimum(surr1, surr2).mean()
    
    # 2. Value Loss
    v = model.v(obs)
    v_loss = mx.mean((v - ret) ** 2)
    
    # Total loss
    loss = pi_loss + 0.5 * v_loss - ent_coef * entropy
    return loss, pi_loss, v_loss, entropy

class PPOAgent:
    def __init__(self, obs_dim, act_dim, 
                 hidden_sizes=[64, 64], 
                 lr=3e-4, 
                 clip_ratio=0.2, 
                 train_iters=80, 
                 ent_coef=0.0):
        
        self.ac = ActorCritic(obs_dim, act_dim, hidden_sizes)
        self.optimizer = optim.Adam(learning_rate=lr)
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.ent_coef = ent_coef
        
        self.loss_and_grad = nn.value_and_grad(self.ac, ppo_loss_fn)

    def update(self, data):
        obs = data['obs']
        act = data['act'].astype(mx.int32)
        ret = data['ret']
        adv = data['adv']
        logp = data['logp']
        
        # Train loop
        pi_l_old, v_l_old, ent_l_old = 0, 0, 0
        
        for i in range(self.train_iters):
            # Compute loss and gradients
            (loss, pi_l, v_l, ent), grads = self.loss_and_grad(
                self.ac, obs, act, adv, ret, logp, self.clip_ratio, self.ent_coef
            )
            
            # Update weights
            self.optimizer.update(self.ac, grads)
            
            # Normalize gradients? (not doing clips here but typically standard)
            # mx.eval(self.ac.parameters()) # ensure computation
            
            if i == 0:
                pi_l_old = pi_l.item()
                v_l_old = v_l.item()
                ent_l_old = ent.item()
                
        return pi_l_old, v_l_old, ent_l_old, loss.item()
