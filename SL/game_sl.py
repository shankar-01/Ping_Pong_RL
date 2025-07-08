import pygame
import csv
import sys
import random

import pickle

# Load model and scaler
with open("knn_model_tuned.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong with Dataset Collection")

# Game clock
clock = pygame.time.Clock()
FPS = 60

# Paddle settings
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
paddle_speed = 5
paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2

# Ball settings
BALL_SIZE = 10
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_dx = 4 * random.choice([-1, 1])
ball_dy = 4 * random.choice([-1, 1])

# Dataset collection
dataset = []

def predict_action(ball_x, ball_y, ball_dx, ball_dy, paddle_y):
    input_data = [[ball_x, ball_y, ball_dx, ball_dy, paddle_y]]
    scaled_input = scaler.transform(input_data)
    predicted_action = model.predict(scaled_input)[0]
    return int(predicted_action)

def draw(paddle_y, ball_x, ball_y):
    screen.fill((0, 0, 0))  # Black background
    pygame.draw.rect(screen, (255, 255, 255), (50, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))  # Paddle
    pygame.draw.rect(screen, (255, 0, 0), (ball_x, ball_y, BALL_SIZE, BALL_SIZE))  # Ball
    pygame.display.flip()

def get_action(keys, paddle_y):
    if keys[pygame.K_UP] and paddle_y > 0:
        return 1  # UP
    elif keys[pygame.K_DOWN] and paddle_y < HEIGHT - PADDLE_HEIGHT:
        return 2  # DOWN
    else:
        return 0  # STAY

def update_paddle(action, paddle_y):
    if action == 1:
        paddle_y -= paddle_speed
    elif action == 2:
        paddle_y += paddle_speed
    return max(0, min(paddle_y, HEIGHT - PADDLE_HEIGHT))

def update_ball(ball_x, ball_y, dx, dy, paddle_y):
    ball_x += dx
    ball_y += dy

    # Bounce off top/bottom walls
    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        dy *= -1

    # Bounce off left paddle
    if ball_x <= 50 + PADDLE_WIDTH and paddle_y < ball_y + BALL_SIZE and ball_y < paddle_y + PADDLE_HEIGHT:
        dx *= -1

    # Bounce off right wall
    if ball_x >= WIDTH - BALL_SIZE:
        dx *= -1

    # Ball missed (reset)
    if ball_x < 0:
        ball_x = WIDTH // 2
        ball_y = random.randint(50, HEIGHT - 50)
        dx = 4 * random.choice([-1, 1])
        dy = 4 * random.choice([-1, 1])

    return ball_x, ball_y, dx, dy

# Main loop
running = True
while running:
    clock.tick(FPS)
    keys = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_q]:
            running = False

    use_ai = True  # Set to False if you want manual control
    if use_ai:
        action = predict_action(ball_x, ball_y, ball_dx, ball_dy, paddle_y)
    else:
        action = get_action(keys, paddle_y)
    paddle_y = update_paddle(action, paddle_y)
    ball_x, ball_y, ball_dx, ball_dy = update_ball(ball_x, ball_y, ball_dx, ball_dy, paddle_y)

    draw(paddle_y, ball_x, ball_y)

    # Save game state to dataset
    # dataset.append([ball_x, ball_y, ball_dx, ball_dy, paddle_y, action])

# Save dataset to CSV
# with open("pong_dataset.csv", mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["ball_x", "ball_y", "ball_dx", "ball_dy", "paddle_y", "action"])
#     writer.writerows(dataset)

# print("Dataset saved to pong_dataset.csv")
pygame.quit()
sys.exit()