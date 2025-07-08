import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# -----------------------------
# Environment setup (no Pygame)
# -----------------------------
WIDTH, HEIGHT = 800, 600
PADDLE_H = 100
PADDLE_W = 10
paddle_speed = 5
BALL_SIZE = 10

# -----------------------------
# RL Agent & Replay Buffer
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)
    def add(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Training setup
# -----------------------------
state_dim = 5  # ball_x, ball_y, ball_dx, ball_dy, paddle_y
action_dim = 3  # stay, up, down
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
buffer = ReplayBuffer()

epsilon, min_epsilon, decay = 1.0, 0.05, 0.995
gamma = 0.99
batch_size = 64
update_freq = 1000
steps_done = 0

# -----------------------------
# Main training loop
# -----------------------------
running = True
episode, max_score = 0, -np.inf
recent_rewards = deque(maxlen=100)

while running:
    paddle_y = HEIGHT // 2 - PADDLE_H // 2

    # Initialize ball with guaranteed vertical component
    ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    ball_dx = random.choice([-3, 3])
    ball_dy = random.uniform(-3, 3)
    if abs(ball_dy) < 1.0:
        ball_dy = 1.0 * np.sign(ball_dy) if ball_dy != 0 else 1.0 * random.choice([-1, 1])

    done = False
    episode_reward = 0
    step_count = 0
    max_steps = 1000  # aggressive cap on episode length

    while not done:
        step_count += 1

        # -----------------------------
        # RL: select action
        state = np.array([ball_x/WIDTH, ball_y/HEIGHT, ball_dx/5, ball_dy/5, paddle_y/(HEIGHT-PADDLE_H)], dtype=np.float32)
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                qvals = policy_net(torch.tensor(state).unsqueeze(0))
                action = qvals.argmax().item()

        # -----------------------------
        # Act on environment
        if action == 1:
            paddle_y -= paddle_speed
        elif action == 2:
            paddle_y += paddle_speed
        paddle_y = np.clip(paddle_y, 0, HEIGHT - PADDLE_H)

        ball_x += ball_dx
        ball_y += ball_dy

        # Prevent near-zero vertical movement causing infinite episodes
        if abs(ball_dy) < 0.1:
            ball_dy = 0.1 * np.sign(ball_dy) if ball_dy != 0 else 0.1 * random.choice([-1, 1])

        # Bounce top/bottom
        if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
            ball_dy *= -1

        # Check paddle collision
        reward, done = -0.01 * abs((paddle_y + PADDLE_H/2) - (ball_y + BALL_SIZE/2)), False
        if ball_x <= 50 + PADDLE_W:
            if paddle_y < ball_y + BALL_SIZE and ball_y < paddle_y + PADDLE_H:
                ball_dx *= -1
                reward = +1.0
            else:
                reward = -5.0
                done = True

        # Bounce off right wall
        if ball_x >= WIDTH - BALL_SIZE:
            ball_dx *= -1

        # -----------------------------
        # RL: end episode if too many steps or horizontal stuck
        if step_count > 300 and abs(ball_dy) < 0.5:
            done = True
            reward = -2.0
            episode_reward += reward

        if step_count > max_steps:
            done = True
            reward = -20.0
            episode_reward += reward

        # -----------------------------
        # RL: store transition
        next_state = np.array([ball_x/WIDTH, ball_y/HEIGHT, ball_dx/5, ball_dy/5, paddle_y/(HEIGHT-PADDLE_H)], dtype=np.float32)
        buffer.add((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        # -----------------------------
        # RL: train
        if len(buffer) > batch_size:
            s, a, r, s_, d = buffer.sample(batch_size)
            s, s_, a, r, d = map(lambda x: torch.tensor(x, dtype=torch.float32), [s, s_, a, r, d])
            q = policy_net(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
            q_target = r + gamma * target_net(s_).max(1)[0] * (1 - d)
            loss = nn.MSELoss()(q, q_target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        steps_done += 1
        if steps_done % update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # -----------------------------
        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * decay)

    # -----------------------------
    # End of episode: log progress
    episode += 1
    recent_rewards.append(episode_reward)
    avg_reward = np.mean(recent_rewards)
    max_score = max(max_score, episode_reward)
    print(f"Episode {episode:5d} | Steps: {step_count:5d} | Reward: {episode_reward:7.2f} | Avg100: {avg_reward:7.2f} | Epsilon: {epsilon:.3f} | MaxReward: {max_score:7.2f}")

    # -----------------------------
    # Optional: stop after enough episodes
    if episode >= 50:
        print("Training complete.")
        torch.save(policy_net.state_dict(), "pong_rl_policy.pth")
        print("Model saved to pong_rl_policy.pth")
        break
