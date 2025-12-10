import mlx.core as mx
import mlx.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64], activation=nn.Tanh):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def __call__(self, obs):
        # Returns logits for categorical distribution
        return self.net(obs)

    def _distribution(self, obs):
        logits = self(obs)
        return logits # Logic for sampling handled outside or via simple Categorical helper if creating one

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[64, 64], activation=nn.Tanh):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def __call__(self, obs):
        return mx.squeeze(self.net(obs), axis=-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64], activation=nn.Tanh):
        super().__init__()
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.v = Critic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        """
        Produce action, value, and logp for a single observation (during rollout).
        """
        # Ensure obs is a batched MLX array
        if isinstance(obs, (list, tuple)) or hasattr(obs, 'shape') and len(obs.shape) == 1:
             obs = mx.array(obs)[None, :]
        
        logits = self.pi(obs)
        v = self.v(obs)
        
        # Sample from categorical (Gumbel-max or mx.random.categorical)
        # mlx.random.categorical not available in all versions, let's check or implement
        # Using mx.random.categorical if available, else simple multinomial logic
        
        # Since I am using MLX 0.30, mx.random.categorical exists.
        act = mx.random.categorical(logits)
        
        # Compute logp of selected action
        # log_softmax(logits)[act]
        logp_all = nn.log_softmax(logits, axis=-1)
        logp = mx.take_along_axis(logp_all, act[..., None], axis=-1).squeeze(-1)
        
        return act.item(), v.item(), logp.item()

    def act(self, obs):
        return self.step(obs)[0]
