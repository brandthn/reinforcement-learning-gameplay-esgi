# PPO Agent — Complete Technical Documentation

## Table of Contents
1. [Overview](#1-overview)
2. [Theoretical Background](#2-theoretical-background)
   - 2.1 Policy Gradient Methods
   - 2.2 Actor-Critic Architecture
   - 2.3 GAE — Generalized Advantage Estimation
   - 2.4 PPO — Proximal Policy Optimization
3. [Architecture & Design Decisions](#3-architecture--design-decisions)
4. [Class Parameters](#4-class-parameters)
5. [Line-by-Line Code Explanation](#5-line-by-line-code-explanation)
   - 5.1 Imports
   - 5.2 `__init__`
   - 5.3 `act()`
   - 5.4 `observe()`
   - 5.5 `end_episode()`
   - 5.6 `save()` / `load()`
   - 5.7 `_compute_gae()`
   - 5.8 `_update()`
6. [Data Flow Diagram](#6-data-flow-diagram)
7. [Design Decisions Deep Dive](#7-design-decisions-deep-dive)

---

## 1. Overview

`agents/policy_gradient/ppo.py` implements **PPO (Proximal Policy Optimization)** in an **A2C-style** (Advantage Actor-Critic), designed for episodic reinforcement learning tasks, including adversarial self-play environments.

**What it does in plain terms:**

> An agent plays a game step-by-step. At each step, it looks at the game state and picks an action (Actor network). It also estimates how "good" the current state is (Critic network). After the full episode is over, it reviews everything that happened and updates its decision-making to make good outcomes more likely and bad outcomes less likely — but *not too aggressively* (that's the "Proximal" part of PPO).

**Where it fits in the project:**

| Concern | Solution |
|---|---|
| Agent interface | Inherits from `Agent` (base.py) |
| Network construction | Uses `build_mlp()` from `training/networks.py` |
| Action masking | Applied only in `act()`, not during training updates |
| Persistence | `.pt` checkpoint format, consistent with value-based agents |
| Self-play compatibility | Works with `SelfPlayTrainer` without modification |

---

## 2. Theoretical Background

### 2.1 Policy Gradient Methods

Unlike value-based methods (DQN, etc.) that learn *Q(s,a)* and derive a policy implicitly, **policy gradient methods directly parameterize the policy** π(a|s; θ) and optimize θ by gradient ascent on the expected return:

```
∇θ J(θ) = E[ ∇θ log π(a|s; θ) · G_t ]
```

Where `G_t` is the discounted return from time `t`. The key challenge is that this gradient has **high variance** because `G_t` involves summing many future rewards.

### 2.2 Actor-Critic Architecture

To reduce variance, we replace `G_t` with an **advantage estimate** `A(s,a)`:

```
A(s,a) = Q(s,a) - V(s)
```

This measures "how much better is this action compared to the average action in this state?". The system now has two components:

- **Actor**: The policy network π(a|s; θ_actor) — decides *what to do*
- **Critic**: The value network V(s; θ_critic) — evaluates *how good the state is*, used to compute the baseline

Using a baseline V(s) does **not** bias the gradient (it is provably zero in expectation) but **significantly reduces its variance**.

### 2.3 GAE — Generalized Advantage Estimation

GAE interpolates between two extremes via the λ parameter:

**TD(0) estimate (λ=0)** — low variance, high bias:
```
A_t = r_t + γ·V(s_{t+1}) - V(s_t)        (δ_t)
```

**Monte-Carlo estimate (λ=1)** — zero bias, high variance:
```
A_t = Σ_{k=0}^{T-t} (γ·λ)^k · δ_{t+k}
```

**GAE(λ)** — the general formula computed backwards:
```
δ_t   = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
A_t   = δ_t + γ·λ·(1 - done_t)·A_{t+1}
```

The `(1 - done_t)` mask ensures the advantage correctly resets to 0 at episode boundaries (no bootstrap across episodes).

### 2.4 PPO — Proximal Policy Optimization

The core PPO innovation is the **clipped surrogate objective**. The ratio of new to old policy probabilities is:

```
r_t(θ) = π_new(a_t|s_t) / π_old(a_t|s_t)
```

The clipped objective prevents the ratio from straying too far from 1.0:

```
L_CLIP(θ) = E[ min( r_t·A_t,  clip(r_t, 1-ε, 1+ε)·A_t ) ]
```

This means:
- If the update is going in a good direction but getting too large → clip it
- If the update is going in a bad direction → allow it (the `min` picks the pessimistic bound)

The full PPO loss combines three terms:
```
L_total = L_CLIP − c_v · L_value + c_e · H(π)
```
Where `H(π)` is the entropy bonus that encourages exploration.

---

## 3. Architecture & Design Decisions

```
State s ──► Actor MLP ──► logits ──► [mask] ──► Categorical distribution ──► Action a
        │
        └──► Critic MLP ──► V(s) (scalar)
```

**Two separate networks** (not shared weights) — explicit design choice. A shared backbone can be more efficient but harder to tune: the critic loss and actor loss may have conflicting gradient directions. Separate networks let each component converge at its own pace, and allow independent learning rate tuning if needed.

**On-policy, episodic** — the trajectory is collected over one full episode, then discarded. This is the fundamental constraint of PPO: you cannot reuse old data beyond `n_epochs` passes, because the clipping bound only corrects for *small* policy deviations.

---

## 4. Class Parameters

| Parameter | Type | Default | Role |
|---|---|---|---|
| `state_size` | int | required | Dimension of the flattened state vector |
| `action_size` | int | required | Total number of possible actions |
| `lr` | float | 3e-4 | Adam learning rate for both networks |
| `gamma` | float | 0.99 | Discount factor (how much future rewards matter) |
| `gae_lambda` | float | 0.95 | GAE λ: trade-off between bias and variance |
| `clip_epsilon` | float | 0.2 | PPO clipping range [1-ε, 1+ε] |
| `entropy_coef` | float | 0.01 | Weight of entropy bonus (exploration) |
| `value_coef` | float | 0.5 | Weight of critic loss in total loss |
| `hidden_layers` | list[int] | [128, 128] | MLP architecture for both networks |
| `n_epochs` | int | 4 | Optimization passes per episode |
| `batch_size` | int | 64 | Mini-batch size during update |
| `max_grad_norm` | float | 0.5 | Gradient clipping threshold |
| `device` | str | "cpu" | Torch computation device |

**Why these defaults?**
- `lr=3e-4`: The canonical PPO learning rate from the original paper (Schulman et al., 2017)
- `gamma=0.99`: Standard for episodic tasks; rewards up to ~100 steps in the future still matter
- `gae_lambda=0.95`: Empirically shown to be the best bias-variance trade-off across many environments
- `clip_epsilon=0.2`: Original paper recommendation; keeps updates conservative
- `entropy_coef=0.01`: Small enough not to dominate, large enough to prevent premature convergence
- `value_coef=0.5`: Balances critic learning speed relative to actor

---

## 5. Line-by-Line Code Explanation

### 5.1 Imports

```python
from __future__ import annotations
```
Enables forward references in type hints (e.g., `list[dict]` instead of `List[dict]` for Python < 3.10).

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional
```
Standard scientific Python stack. `torch.distributions.Categorical` (used later) is part of core PyTorch.

```python
from agents.base import Agent
from training.networks import build_mlp
```
Project-specific imports. `Agent` defines the interface contract (`act`, `observe`, `end_episode`, `save`, `load`). `build_mlp` is a factory that builds a `nn.Sequential` MLP with the specified layer sizes.

---

### 5.2 `__init__`

```python
name = "ppo"
```
Class-level attribute used by the training framework to identify and log the agent type.

```python
self._state_size = state_size
self._action_size = action_size
self._gamma = gamma
...
```
Store all hyperparameters as instance attributes. Prefixed with `_` to signal they are internal (not part of the public interface).

```python
self._device = torch.device(device)
```
Converts the string `"cpu"` or `"cuda"` to a `torch.device` object. All tensors will be sent to this device.

```python
hidden = hidden_layers or [128, 128]
```
If the caller passes `None` (or omits the argument), default to a two-layer MLP of size 128. This avoids mutable default arguments (`def __init__(self, hidden_layers=[128,128])` is a Python anti-pattern).

```python
self._actor: nn.Sequential = build_mlp(state_size, action_size, hidden).to(self._device)
self._critic: nn.Sequential = build_mlp(state_size, 1, hidden).to(self._device)
```
Two **separate MLPs**. The actor outputs `action_size` raw logits (unnormalized scores). The critic outputs a single scalar V(s). Both are immediately moved to the target device.

```python
self._optimizer = optim.Adam(
    list(self._actor.parameters()) + list(self._critic.parameters()),
    lr=lr,
)
```
A **single shared optimizer** for both networks. This is simpler than two separate optimizers and works well in practice. The parameter lists are concatenated so Adam tracks separate moment estimates (m, v) for each parameter individually.

```python
self._trajectory: list[dict] = []
```
In-memory buffer storing every step of the current episode. Each entry is a dict with keys: `state`, `action`, `reward`, `log_prob`, `value`, `done`.

```python
self._pending: Optional[dict] = None
```
Temporary storage for the output of `act()` (state, action, log_prob, value) until `observe()` is called with the resulting reward. This bridges the two-call interface: `act()` → environment step → `observe()`.

```python
self._step_count: int = 0
```
Global step counter for external logging/monitoring. Not used internally.

---

### 5.3 `act()`

```python
def act(self, state, available_actions, training=True) -> int:
```
The main decision method. `available_actions` is a list of legal action indices — critical for masked sampling.

```python
state_t = torch.as_tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
```
Converts the numpy state array to a float32 tensor on the correct device. `.unsqueeze(0)` adds a batch dimension: shape goes from `(state_size,)` to `(1, state_size)` to match the network's expected input format.

`torch.as_tensor` is preferred over `torch.FloatTensor(state)` because it avoids a data copy if the array is already the right dtype and device.

```python
with torch.no_grad():
    logits = self._actor(state_t).squeeze(0)       # (action_size,)
    value  = self._critic(state_t).squeeze().item()  # scalar float
```
`torch.no_grad()` disables gradient computation — we are only doing inference here, not training. This saves memory and speeds up the forward pass. `.squeeze()` removes the batch dimension. `.item()` converts a single-element tensor to a Python float.

```python
mask = torch.full((self._action_size,), float("-inf"))
mask[available_actions] = 0.0
masked_logits = logits + mask
```
**Action masking** (Design decision D-012). We create a vector of `-inf` for all actions and set `0.0` for legal ones. Adding this to the logits effectively makes illegal actions have `-inf` score. After `softmax`, `exp(-inf) = 0`, so illegal actions get exactly **zero probability**. This is mathematically cleaner than zeroing out probabilities after softmax (which would require renormalization).

```python
if training:
    probs = torch.softmax(masked_logits, dim=-1)
    dist  = torch.distributions.Categorical(probs)
    action_t = dist.sample()
    log_prob  = dist.log_prob(action_t).item()
    action    = action_t.item()
```
In training mode: sample stochastically from the masked probability distribution. `Categorical.log_prob()` gives us `log π(a|s)` which we need later in the PPO ratio calculation. This encourages **exploration** — the agent can try suboptimal actions and learn from the consequences.

```python
else:
    action   = int(masked_logits.argmax().item())
    probs    = torch.softmax(masked_logits, dim=-1)
    log_prob = torch.log(probs[action] + 1e-8).item()
```
In inference mode: take the greedy action (highest logit). The `+1e-8` epsilon prevents `log(0)` numerical errors (though in greedy mode `log_prob` is not used for training).

```python
self._pending = {
    "state"    : state.copy(),
    "action"   : action,
    "log_prob" : log_prob,
    "value"    : value,
}
```
Store the step's data. `state.copy()` is important — it prevents the dict from holding a reference to the original numpy array, which the caller might mutate before `observe()` is called.

```python
self._step_count += 1
return action
```
Increment the global counter and return the chosen action as a plain Python int.

---

### 5.4 `observe()`

```python
def observe(self, state, action, reward, next_state, done) -> None:
```
Called by the Trainer after the environment has processed the action from `act()`. `next_state` is received but not stored — in episodic PPO with a terminal bootstrap of 0, we don't need it directly (it's implicitly encoded in the next step's `value`).

```python
if self._pending is None:
    return
```
Guard against malformed call sequences (observe without a preceding act). In a well-formed training loop, this should never trigger.

```python
self._trajectory.append({
    "state"    : self._pending["state"],
    "action"   : self._pending["action"],
    "reward"   : float(reward),
    "log_prob" : self._pending["log_prob"],
    "value"    : self._pending["value"],
    "done"     : bool(done),
})
self._pending = None
```
Merge the pending act-data with the reward and done signal from the environment into a single trajectory entry. `float()` and `bool()` enforce type safety in case the environment returns numpy scalars. After appending, `_pending` is cleared.

---

### 5.5 `end_episode()`

```python
def end_episode(self) -> None:
```
Called by the Trainer exactly once after the episode's final `observe(done=True)`.

```python
if len(self._trajectory) < 2:
    self._trajectory = []
    return
```
A trajectory of length 1 cannot form a valid mini-batch of size ≥ 2. Skipping prevents degenerate updates that could destabilize training.

```python
self._update()
self._trajectory = []
```
Run the full PPO update, then **discard the trajectory**. This is the fundamental on-policy requirement: the collected data was generated under the old policy π_old, and after `n_epochs` updates we've moved too far for the PPO approximation to remain valid.

---

### 5.6 `save()` / `load()`

```python
def save(self, path: str) -> None:
    torch.save({
        "actor"      : self._actor.state_dict(),
        "critic"     : self._critic.state_dict(),
        "optimizer"  : self._optimizer.state_dict(),
        "step_count" : self._step_count,
    }, path)
```
Saves a checkpoint dictionary to a `.pt` file. `state_dict()` extracts only the learnable parameters (not the full model object), which is more portable across code refactors and PyTorch versions.

```python
def load(self, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    self._actor.load_state_dict(ckpt["actor"])
    self._actor.to(self._device)
    self._critic.load_state_dict(ckpt["critic"])
    self._critic.to(self._device)
    self._optimizer.load_state_dict(ckpt["optimizer"])
    self._step_count = ckpt.get("step_count", 0)
```
`map_location="cpu"` ensures checkpoints saved on GPU can be loaded on CPU (and vice versa). After loading, `.to(self._device)` moves the weights to the correct device for the current run. The two-step (load → move) pattern is safer than loading directly to the target device when the checkpoint's origin device is unknown. `ckpt.get("step_count", 0)` gracefully handles old checkpoints that predate the step counter field.

---

### 5.7 `_compute_gae()`

```python
T = len(self._trajectory)
advantages = np.zeros(T, dtype=np.float32)
gae        = 0.0
next_value = 0.0
```
Pre-allocate the advantages array. `gae` accumulates the running GAE sum during the backward pass. `next_value` is initialized to 0.0 — this is the **terminal bootstrap assumption**: at the end of the episode `V(s_T) = 0`, which is correct for any terminal state in a properly designed reward function.

```python
for t in reversed(range(T)):
```
GAE is computed **backwards** through time (from T-1 to 0) because `A_t` depends on `A_{t+1}`.

```python
    r    = self._trajectory[t]["reward"]
    v    = self._trajectory[t]["value"]
    done = float(self._trajectory[t]["done"])
```
Extract the three quantities needed: reward received, state value estimated at this step (by the critic at act-time), and whether this step ended the episode.

```python
    delta = r + self._gamma * next_value * (1.0 - done) - v
```
The **TD error** δ_t: how much better or worse did reality turn out compared to the critic's prediction? The `(1.0 - done)` factor ensures that if this step is terminal, we don't bootstrap from `next_value` (there is no "next state" after a terminal transition).

```python
    gae   = delta + self._gamma * self._gae_lambda * (1.0 - done) * gae
    advantages[t] = gae
    next_value    = v
```
Update the running GAE sum and store it. Then update `next_value` to `v` (the current state's value) for the next backwards iteration, which corresponds to `V(s_{t+1})` when processing step `t-1`.

```python
values  = np.array([tr["value"] for tr in self._trajectory], dtype=np.float32)
returns = advantages + values
return advantages, returns
```
The **returns** (critic targets) are computed as `A_t + V(s_t)`. This is algebraically equivalent to the lambda-return `G_t^λ` and is the standard PPO formulation. The critic is trained to predict these returns via MSE.

---

### 5.8 `_update()`

```python
advantages, returns = self._compute_gae()
```
Compute GAE advantages and returns for the full episode trajectory.

```python
adv_std = advantages.std()
if adv_std > 1e-8:
    advantages = (advantages - advantages.mean()) / adv_std
```
**Advantage normalization**: subtract the mean and divide by standard deviation. This is not theoretically required by PPO but is a near-universal practical improvement — it ensures the gradient magnitudes are consistent across episodes of varying length and reward scale. The `1e-8` guard prevents division by zero for constant-reward trajectories.

```python
states_t        = torch.as_tensor(np.stack([tr["state"]    for tr in self._trajectory]), dtype=torch.float32, device=self._device)
actions_t       = torch.as_tensor([tr["action"]   for tr in self._trajectory],           dtype=torch.long,    device=self._device)
old_log_probs_t = torch.as_tensor([tr["log_prob"] for tr in self._trajectory],           dtype=torch.float32, device=self._device)
returns_t       = torch.as_tensor(returns,                                                dtype=torch.float32, device=self._device)
advantages_t    = torch.as_tensor(advantages,                                             dtype=torch.float32, device=self._device)
```
**Batch tensor construction**: done once outside the epoch loop to avoid redundant numpy→torch conversions. All tensors are sent to the correct device. Using `torch.as_tensor` avoids data copies when the numpy arrays are already contiguous float32.

```python
T       = len(self._trajectory)
indices = np.arange(T)

for _ in range(self._n_epochs):
    np.random.shuffle(indices)
```
Outer loop: `n_epochs` passes over the full trajectory. Shuffling indices at each epoch ensures different mini-batch compositions, reducing correlation between consecutive gradient steps.

```python
    for start in range(0, T, self._batch_size):
        batch_idx = indices[start : start + self._batch_size]
        if len(batch_idx) < 2:
            continue
```
Inner loop: iterate over mini-batches. Skip batches smaller than 2 to avoid instabilities (e.g., with batch normalization or variance computations).

```python
        b_states     = states_t[batch_idx]
        b_actions    = actions_t[batch_idx]
        b_old_lp     = old_log_probs_t[batch_idx]
        b_returns    = returns_t[batch_idx]
        b_advantages = advantages_t[batch_idx]
```
Index-select the batch tensors. PyTorch supports integer-array indexing directly on tensors.

```python
        logits     = self._actor(b_states)               # (B, action_size)
        new_values = self._critic(b_states).squeeze(-1)  # (B,)
```
Forward pass through both networks with the **current** (updated) weights. This is the key: we're computing π_new while comparing against π_old (the log_probs stored in the trajectory).

```python
        log_probs = torch.log_softmax(logits, dim=-1)    # (B, A)
        new_lp    = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)  # (B,)
```
`log_softmax` is numerically more stable than `log(softmax(...))` (avoids overflow/underflow in the exp step). `.gather(1, b_actions.unsqueeze(1))` selects the log-probability of the *actually taken action* from each row — equivalent to `log_probs[i, actions[i]]` for each sample `i`.

**No action masking here** (D-012): the actor is trained on the actions that were actually taken (which are always legal). The network naturally learns to assign low probability to illegal actions over time, without needing the mask during the gradient update.

```python
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
```
Shannon entropy `H(π) = -Σ p(a) log p(a)`, computed across the full action space and averaged over the batch. Higher entropy = more exploration. Adding this to the loss (with a positive coefficient) discourages the policy from collapsing to a single deterministic action too quickly.

```python
        ratio  = torch.exp(new_lp - b_old_lp)
```
The **importance sampling ratio** `π_new(a|s) / π_old(a|s)`. Computed in log space first (`new_lp - b_old_lp`) then exponentiated for numerical stability (avoids dividing two potentially very small probabilities directly).

```python
        surr1  = ratio * b_advantages
        surr2  = torch.clamp(ratio, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon) * b_advantages
        actor_loss = -torch.min(surr1, surr2).mean()
```
The **PPO clipped objective**. `surr1` is the unconstrained update. `surr2` clips the ratio to prevent overly large policy updates. `torch.min` takes the pessimistic (conservative) bound. The negative sign converts maximization into minimization (gradient descent).

```python
        value_loss = nn.functional.mse_loss(new_values, b_returns)
```
Simple **Mean Squared Error** between the critic's current value predictions and the target returns computed via GAE. No value clipping is applied — the comment notes that value clipping can slow learning in simple, short-episode environments.

```python
        loss = (
            actor_loss
            + self._value_coef * value_loss
            - self._entropy_coef * entropy
        )
```
The **combined loss**:
- `actor_loss`: minimize (already negated, so gradient ascent on policy objective)
- `+ value_coef * value_loss`: minimize critic MSE
- `- entropy_coef * entropy`: maximize entropy (subtract because we're doing gradient descent)

```python
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._actor.parameters()) + list(self._critic.parameters()),
            self._max_grad_norm,
        )
        self._optimizer.step()
```
Standard PyTorch training step:
1. `zero_grad()`: clear accumulated gradients from the previous step
2. `backward()`: compute gradients via backpropagation
3. `clip_grad_norm_()`: **gradient clipping** — if the global gradient norm exceeds `max_grad_norm`, rescale all gradients proportionally. This prevents "gradient explosions" that can destabilize training, especially in recurrent or deep networks
4. `step()`: apply the Adam update rule

---

## 6. Data Flow Diagram

```
Episode begins
     │
     ▼
 act(state, available_actions)
     │  ├─ Actor forward → logits
     │  ├─ Apply action mask (-inf to illegal actions)
     │  ├─ Sample action from Categorical(softmax(masked_logits))
     │  ├─ Critic forward → V(s)
     │  └─ Store {state, action, log_prob, value} in _pending
     │
     │  [Environment executes action]
     │
     ▼
 observe(s, a, reward, next_s, done)
     │  └─ Append {_pending + reward + done} to _trajectory
     │
     │  [Repeat act() → observe() for each step]
     │
     ▼
 end_episode()
     │  ├─ _compute_gae()
     │  │    └─ Backwards pass over trajectory
     │  │         └─ Returns advantages[] and returns[]
     │  ├─ Normalize advantages
     │  ├─ Build full-episode tensors (states, actions, old_log_probs, ...)
     │  └─ For n_epochs:
     │       └─ For each mini-batch (shuffled):
     │            ├─ Actor forward → new_log_probs, entropy
     │            ├─ Critic forward → new_values
     │            ├─ ratio = exp(new_lp - old_lp)
     │            ├─ actor_loss = -min(ratio·A, clip(ratio)·A)
     │            ├─ value_loss = MSE(new_values, returns)
     │            ├─ loss = actor_loss + c_v·value_loss - c_e·entropy
     │            └─ Backprop + gradient clip + optimizer step
     │
     └─ Clear _trajectory → next episode
```

---

## 7. Design Decisions Deep Dive

### D-012: Action masking only at inference, not at training

**Decision**: The action mask (`-inf` on illegal logits) is applied in `act()` but NOT in `_update()`.

**Why**: During the update, we only process transitions that actually occurred — and since `act()` guarantees only legal actions are sampled, every action in the trajectory is legal. Re-applying the mask during training would:
- Require storing `available_actions` for every step (memory overhead)
- Change the entropy calculation (making it ignore illegal actions in the distribution)
- Create a mismatch between the probability distribution at collection time and update time

The network naturally learns to assign near-zero probability to actions that are never selected.

### Separate actor and critic networks

**Decision**: Two independent MLPs rather than a shared backbone.

**Why**: A shared backbone creates a gradient conflict — the critic gradients and actor gradients can point in different directions for the shared layers. Separate networks allow clean, independent optimization. The cost is higher parameter count, but with `[128,128]` layers this is negligible.

### Single shared optimizer

**Decision**: One Adam optimizer for both networks.

**Why**: Simpler code. Since both networks are updated on the same loss in the same backward pass, using separate optimizers would only matter if you wanted different learning rates — a rarely needed refinement that can be added later.

### Terminal bootstrap at 0

**Decision**: `next_value = 0.0` in `_compute_gae()`.

**Why**: The project guarantees `end_episode()` is always called after `done=True`. So the last step in every trajectory is a terminal state, and V(terminal) = 0 by definition (there are no future rewards). If the trainer ever called `end_episode()` mid-episode (e.g., for time limits), bootstrapping from the critic's prediction would be needed instead.

### No value loss clipping

**Decision**: Plain MSE for the critic, no clipped value loss.

**Why**: The original PPO paper includes value clipping to bound the critic update similarly to the actor. However, this is controversial — it can actually hurt performance in simple environments by slowing down the critic's convergence. Since this implementation targets simple/moderate-complexity games, plain MSE is preferred.