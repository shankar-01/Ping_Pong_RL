import pygame
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import pickle

# =========================
# Load Supervised Learning Model
# =========================
with open("./SL/knn_model_tuned.pkl", "rb") as f:
    knn_model, knn_scaler = pickle.load(f)

# =========================
# Load DQN RL Model
# =========================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn_model = DQN(state_size=5, action_size=3)
dqn_model.load_state_dict(torch.load("./RL/pong_rl_policy.pth", map_location=device))
dqn_model.eval()

# =========================
# Game Setup
# =========================
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong SL vs RL")
clock = pygame.time.Clock()
FPS = 60

PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
paddle_speed = 5
paddle_y_sl = HEIGHT // 2 - PADDLE_HEIGHT // 2  # Left paddle (SL)
paddle_y_rl = HEIGHT // 2 - PADDLE_HEIGHT // 2  # Right paddle (RL)

BALL_SIZE = 10
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_dx = 4 * random.choice([-1, 1])
ball_dy = 4 * random.choice([-1, 1])

# Scores
score_sl = 0
score_rl = 0

# Hit/Miss counters
hits_sl = 0
misses_sl = 0
hits_rl = 0
misses_rl = 0

# Fonts
pygame.font.init()
score_font = pygame.font.SysFont("Arial", 30)

# =========================
# Functions
# =========================
def predict_action_sl(ball_x, ball_y, ball_dx, ball_dy, paddle_y):
    input_data = [[ball_x, ball_y, ball_dx, ball_dy, paddle_y]]
    scaled_input = knn_scaler.transform(input_data)
    predicted_action = knn_model.predict(scaled_input)[0]
    return int(predicted_action)

def predict_action_rl(ball_x, ball_y, ball_dx, ball_dy, paddle_y):
    state = np.array([
        ball_x / WIDTH,
        ball_y / HEIGHT,
        ball_dx / 5,
        ball_dy / 5,
        paddle_y / (HEIGHT - PADDLE_HEIGHT)
    ], dtype=np.float32)
    with torch.no_grad():
        q_values = dqn_model(torch.tensor(state).unsqueeze(0))
        action = q_values.argmax().item()
    return action

def update_paddle(action, paddle_y):
    if action == 1:
        paddle_y -= paddle_speed
    elif action == 2:
        paddle_y += paddle_speed
    return max(0, min(paddle_y, HEIGHT - PADDLE_HEIGHT))

def reset_ball():
    x = WIDTH // 2
    y = random.randint(50, HEIGHT - 50)
    dx = 4 * random.choice([-1, 1])
    dy = 4 * random.choice([-1, 1])
    return x, y, dx, dy

def update_ball(ball_x, ball_y, dx, dy, paddle_y_sl, paddle_y_rl):
    global score_sl, score_rl, hits_sl, misses_sl, hits_rl, misses_rl

    ball_x += dx
    ball_y += dy

    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        dy *= -1

    # Left paddle (SL)
    if dx < 0 and ball_x <= 50 + PADDLE_WIDTH:
        if paddle_y_sl <= ball_y + BALL_SIZE and ball_y <= paddle_y_sl + PADDLE_HEIGHT:
            dx *= -1  # reflect
            hits_sl += 1
        else:
            score_rl += 1  # miss by SL, RL scores
            misses_sl += 1
            return reset_ball()

    # Right paddle (RL)
    if dx > 0 and ball_x + BALL_SIZE >= WIDTH - 50 - PADDLE_WIDTH:
        if paddle_y_rl <= ball_y + BALL_SIZE and ball_y <= paddle_y_rl + PADDLE_HEIGHT:
            dx *= -1  # reflect
            hits_rl += 1
        else:
            score_sl += 1  # miss by RL, SL scores
            misses_rl += 1
            return reset_ball()

    return ball_x, ball_y, dx, dy

def draw(paddle_y_sl, paddle_y_rl, ball_x, ball_y, score_sl, score_rl):
    screen.fill((0, 0, 0))

    # Draw paddles & ball
    pygame.draw.rect(screen, (0, 255, 0), (50, paddle_y_sl, PADDLE_WIDTH, PADDLE_HEIGHT))  # SL paddle
    pygame.draw.rect(screen, (0, 0, 255), (WIDTH - 50 - PADDLE_WIDTH, paddle_y_rl, PADDLE_WIDTH, PADDLE_HEIGHT))  # RL paddle
    pygame.draw.rect(screen, (255, 0, 0), (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

    # Draw scores
    sl_text = score_font.render(f"SL: {score_sl}", True, (0, 255, 0))
    rl_text = score_font.render(f"RL: {score_rl}", True, (0, 0, 255))
    screen.blit(sl_text, (30, 10))
    screen.blit(rl_text, (WIDTH - rl_text.get_width() - 30, 10))

    # Draw hit/miss info
    hit_miss_font = pygame.font.SysFont("Arial", 20)
    sl_info = hit_miss_font.render(f"SL Hits: {hits_sl}  Misses: {misses_sl}", True, (0, 255, 0))
    rl_info = hit_miss_font.render(f"RL Hits: {hits_rl}  Misses: {misses_rl}", True, (0, 0, 255))
    screen.blit(sl_info, (30, HEIGHT - 60))
    screen.blit(rl_info, (WIDTH - rl_info.get_width() - 30, HEIGHT - 60))

    pygame.display.flip()

# =========================
# Main Loop
# =========================
running = True
while running:
    clock.tick(FPS)
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_q]:
            running = False

    # Get actions
    action_sl = predict_action_sl(ball_x, ball_y, ball_dx, ball_dy, paddle_y_sl)
    action_rl = predict_action_rl(ball_x, ball_y, ball_dx, ball_dy, paddle_y_rl)

    # Update paddles
    paddle_y_sl = update_paddle(action_sl, paddle_y_sl)
    paddle_y_rl = update_paddle(action_rl, paddle_y_rl)

    # Update ball
    ball_x, ball_y, ball_dx, ball_dy = update_ball(ball_x, ball_y, ball_dx, ball_dy, paddle_y_sl, paddle_y_rl)

    # Draw frame
    draw(paddle_y_sl, paddle_y_rl, ball_x, ball_y, score_sl, score_rl)

pygame.quit()
sys.exit()
