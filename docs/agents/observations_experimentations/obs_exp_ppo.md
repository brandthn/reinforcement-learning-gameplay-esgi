# Observations experimentations PPO
## Line_world_________________
### Exp 1 ------
#### Config
...
#### Resultats
...

## Grid_world_________________
### Exp 1 ------
#### Config
...
#### Resultats
...

## TicTacToe_________________
### Exp 1 ------
#### Config
...
#### Resultats
...


## Bobail_________________
### Exp 1 ------
#### Config
env: bobail
agent: ppo
opponent: random           # adversarial → SelfPlayTrainer
 
agent_params:
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95         # λ GAE : compromis biais/variance
  clip_epsilon: 0.2        # clipping PPO standard
  entropy_coef: 0.01       # encourage l'exploration initiale
  value_coef: 0.5          # pondération perte critic
  hidden_layers: [256, 256]
  n_epochs: 4              # passes d'optimisation par épisode
  batch_size: 64
  max_grad_norm: 0.5
 
training:
  num_episodes: 100000
  max_steps_per_episode: 500   # Bobail peut durer longtemps
 
eval:
  checkpoints: [1000, 10000, 100000]
  num_games: 100
 
seeds: [42, 123, 456]

#### Resultats
.. Beaucoup de mots pour essayer de comprendre ce qui c'est passé ..

### Exp 2 ------
#### Config
...
#### Resultats
...