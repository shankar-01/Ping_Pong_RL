import pygame
import torch
import torch.nn as nn
import numpy as np
import sys
import random

# -----------------------------
# DQN Model Definition
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

# -----------------------------
# Load Trained DQN Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(state_size=5, action_size=3)
model.load_state_dict(torch.load("pong_rl_policy.pth", map_location=device))
model.eval()

# -----------------------------
# Game Setup
# -----------------------------
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong with DQN Agent")
clock = pygame.time.Clock()
FPS = 60

PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
paddle_speed = 5
paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2

BALL_SIZE = 10
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_dx = 4 * random.choice([-1, 1])
ball_dy = 4 * random.choice([-1, 1])

# -----------------------------
# Functions
# -----------------------------
def draw(paddle_y, ball_x, ball_y):
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), (50, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, (255, 0, 0), (ball_x, ball_y, BALL_SIZE, BALL_SIZE))
    pygame.display.flip()

def update_paddle(action, paddle_y):
    if action == 1:
        paddle_y -= paddle_speed
    elif action == 2:
        paddle_y += paddle_speed
    return max(0, min(paddle_y, HEIGHT - PADDLE_HEIGHT))

def update_ball(ball_x, ball_y, dx, dy, paddle_y):
    ball_x += dx
    ball_y += dy

    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        dy *= -1

    if ball_x <= 50 + PADDLE_WIDTH and paddle_y < ball_y + BALL_SIZE and ball_y < paddle_y + PADDLE_HEIGHT:
        dx *= -1

    if ball_x >= WIDTH - BALL_SIZE:
        dx *= -1

    if ball_x < 0:
        ball_x = WIDTH // 2
        ball_y = random.randint(50, HEIGHT - 50)
        dx = 4 * random.choice([-1, 1])
        dy = 4 * random.choice([-1, 1])

    return ball_x, ball_y, dx, dy

def predict_action(ball_x, ball_y, ball_dx, ball_dy, paddle_y):
    # Normalize input just like in training
    state = np.array([
        ball_x / WIDTH,
        ball_y / HEIGHT,
        ball_dx / 5,
        ball_dy / 5,
        paddle_y / (HEIGHT - PADDLE_HEIGHT)
    ], dtype=np.float32)
    
    with torch.no_grad():
        q_values = model(torch.tensor(state).unsqueeze(0))
        action = q_values.argmax().item()
    return action

# -----------------------------
# Main Loop
# -----------------------------
running = True
while running:
    clock.tick(FPS)
    keys = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_q]:
            running = False

    action = predict_action(ball_x, ball_y, ball_dx, ball_dy, paddle_y)
    paddle_y = update_paddle(action, paddle_y)
    ball_x, ball_y, ball_dx, ball_dy = update_ball(ball_x, ball_y, ball_dx, ball_dy, paddle_y)

    draw(paddle_y, ball_x, ball_y)

pygame.quit()
sys.exit()
