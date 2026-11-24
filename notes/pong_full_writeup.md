# Part 1: Pong agent trained until it wins (`pong_full`)

## Result

| | Class engine (M10 Q-ish, 500 games) | **This agent (DQN, 205 games)** |
| --- | --- | --- |
| 200-game greedy evaluation | −21.0 (20 games, never returned a ball) | **+21.00** (won all 200 games, 21–0) |
| Training games | 500 | **205** |
| Training time | ~38 min (Colab) | **45 min** (MacBook, Apple Silicon CPU only, no GPU) |
| Re-run of the evaluation | n/a | identical scores |

Learning milestones (100-game running mean of training scores):

| Milestone | Game | Wall clock |
| --- | --- | --- |
| First game won | 80 | ~0.3 h |
| Mean ≥ −10 | 109 | 0.41 h |
| Mean ≥ 0 (winning on average) | 136 | 0.52 h |
| Mean ≥ +15 | 177 | 0.67 h |
| Stopped at mean ≥ +18 | 205 | 0.75 h |

Files: `results/pong_full_train_log.csv`, `results/pong_full_eval_200.csv`, `results/pong_full_meta.json`, `models/pong_full.weights.h5`.

## What we tried, what worked, what failed

- **Baseline (failed):** the class Q-ish engine trains once per game on that game's own frames, with the discounted final reward of each point as the target. After 500 games it still lost every evaluation game 0–21. Learning from one correlated game at a time, with almost only negative targets early on, gave it nothing to learn from.
- **Kept from class:** the same network (Conv 16 → Conv 32 → Dense 256 → 3 Q-values), the same 80×80 preprocessing, 4 stacked frames, the 3 useful buttons (NOOP / UP / DOWN), an action every 4 frames, and a linearly annealed ε.
- **Biggest change: Bellman targets with a target network.** Instead of "did we eventually win this point?", each move is trained toward `r + 0.99 · Q_target(next state)`. The agent learns how good a position is from the position after it, so credit flows back through the rally. A frozen copy of the network (synced every 1,000 moves) keeps the target stable.
- **Replay buffer (helped):** 50,000 recent moves, with one gradient step on a random batch of 32 after every move. Random batches break the frame-to-frame correlation, and each move is reused many times. Frames are bit-packed (they are 0/1), so the buffer needs ~320 MB instead of ~2.6 GB.
- **Stabilisers:** Huber loss, gradient-norm clipping at 10, and Double DQN (the online net picks the next action, the target net values it). These are standard and cheap; we did not run an ablation without them.
- **Exploration schedule:** ε fell linearly from 1.0 to 0.02 over the first 150,000 moves (~100 games). Almost all the improvement happened in games 70–180, right as ε reached its floor.
- **Not used:** reward shaping and skipping no-ball frames. The Bellman target already credits a good return because it leads to a won point, and shaped rewards risk teaching the agent to prolong rallies. The agent won without them.
- **Loss is not a progress signal.** The training loss stayed around 0.0005–0.002 the whole time, even while the score went from −21 to +21. Rewards are rare, so most targets are near zero. We tracked the score instead.
- **Caveat: the emulator is deterministic.** As in class, sticky actions are off (`repeat_action_probability = 0`). With the same moves, every game plays out identically, so all 200 evaluation games are the same game, which is why the standard deviation is 0. The agent has effectively found a winning strategy against the fixed built-in opponent. A test with sticky actions (0.25, the research standard) would show how well it handles randomness; that would be the next step before using this in a real game.
- **Engineering:** checkpoints every 100 games (network + Adam state, target network, buffer, RNG state), written to a temporary folder and then swapped in, so training resumes exactly after a disconnect. Tested by stopping and resuming. Training is also fully reproducible: the same seed reproduced the same scores game by game.

## Key code chunks for the report

**1. The DQN learning step (the core change from the class engine)**

```python
@tf.function
def train_step(s, a, r, s2, done):
    # target y: reward + discounted value of the next state (Double DQN)
    q_next_target = target_net(s2, training=False)
    best_next = tf.argmax(q_net(s2, training=False), axis=1)          # online net picks a'
    q_next = tf.reduce_sum(q_next_target * tf.one_hot(best_next, N_ACTIONS), axis=1)
    y = r + gamma * (1.0 - done) * q_next

    # move Q(s, a) toward y
    with tf.GradientTape() as tape:
        q = q_net(s, training=True)
        q_taken = tf.reduce_sum(q * tf.one_hot(a, N_ACTIONS), axis=1)
        loss = huber(y, q_taken)
    grads = tape.gradient(loss, q_net.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, grad_clip)
    optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss
```

**2. One move of the training loop: explore, play, remember, learn**

```python
eps = epsilon_at(step, hp)                        # linearly annealed 1.0 -> 0.02
if rng.random() < eps:
    a = int(rng.integers(N_ACTIONS))              # explore
else:
    a = int(np.argmax(q_fn(state[None])[0]))      # exploit: best Q-value

obs, r, terminated, truncated, _ = env.step(PONG_ACTIONS[a])   # held for 4 frames
next_state = frames(obs)                          # 4 stacked 80x80 frames
buffer.add(state, a, r, next_state, terminated)

if buffer.n >= hp["replay_start"] and step % hp["train_every"] == 0:
    train_step(*buffer.sample(hp["batch_size"], rng))         # random mini-batch
if step % hp["target_sync"] == 0:
    target_net.set_weights(q_net.get_weights())               # refresh frozen target
```

**3. Bit-packed replay buffer (8x less memory)**

```python
def add(self, s, a, r, s2, done):
    i = self.ptr
    self.s[i] = np.packbits(s.astype(np.uint8))     # 80*80*4 zeros/ones -> 3,200 bytes
    self.s2[i] = np.packbits(s2.astype(np.uint8))
    self.a[i], self.r[i], self.done[i] = a, r, float(done)
    self.ptr = (self.ptr + 1) % self.capacity       # overwrite the oldest move when full
    self.n = min(self.n + 1, self.capacity)
```

**4. Linear epsilon annealing**

```python
def epsilon_at(step, hp=HP):
    frac = min(1.0, step / hp["eps_decay_steps"])
    return hp["eps_start"] + frac * (hp["eps_end"] - hp["eps_start"])
```
