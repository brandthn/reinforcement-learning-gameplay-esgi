"""Double DQN with Experience Replay (DDQN+ER) agent."""

from .ddqn import DDQNAgent


class DDQNERAgent(DDQNAgent):
    """DDQN with configurable experience replay warm-up.

    Adds learning_starts: minimum number of steps before training begins,
    allowing the replay buffer to accumulate diverse transitions first.
    All other behavior (DDQN targets, epsilon-greedy, target sync) inherited.
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float, gamma: float,
                 epsilon_start: float, epsilon_end: float,
                 epsilon_decay_steps: int,
                 hidden_layers: list[int],
                 batch_size: int, buffer_capacity: int,
                 target_update_freq: int,
                 learning_starts: int):
        super().__init__(
            state_size=state_size, action_size=action_size,
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            hidden_layers=hidden_layers,
            batch_size=batch_size, buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq,
        )
        self._learning_starts = learning_starts

    def observe(self, state, action, reward, next_state, done):
        self._buffer.push(state, action, reward, next_state, done)
        if (len(self._buffer) >= self._batch_size
                and self._step_count >= self._learning_starts):
            self._train_step()
