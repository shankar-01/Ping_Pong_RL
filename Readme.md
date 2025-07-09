# Pong AI: SL vs RL & RL vs SL

This project features a Pong game where you can pit a **Supervised Learning (SL)** agent against a **Reinforcement Learning (RL)** agent, and vice versa.

## 🛠️ Setup Instructions

1. **Install dependencies with Pipenv**  
   Open a terminal in the project directory and run:

   ```
   pipenv install
   ```

2. **Activate the virtual environment**  
   Once installed, activate the shell:

   ```
   pipenv shell
   ```

## 🎮 How to Run the Game

- **To run SL vs RL (Supervised Learning on the left, RL on the right):**

   ```
   python game_combined_v1.py
   ```

- **To run RL vs SL (Reinforcement Learning on the left, SL on the right):**

   ```
   python game_combined_v2.py
   ```

## 🧠 Models Required

Make sure the following model files are present in the project directory:

- `knn_model_tuned.pkl` — Trained KNN model and scaler for SL agent
- `pong_rl_policy.pth` — Trained DQN model for RL agent


## 📁 Project Structure

```
PING_PONG_PYGAME/
│
├── RL/
│ ├── game_rl.py
│ ├── pong_rl_policy.pth
│ ├── train_rl.py
│ └── old_models/
│
├── SL/
│ ├── game_sl.py
│ ├── knn_model_tuned.pkl
│ ├── training/
│ │ ├── dataset_manis.csv
│ │ ├── pong_dataset_20_min_ali.csv
│ │ ├── pong_dataset_20_min_ali2.csv
│ │ ├── pong_dataset_40_min_shankar.csv
│ │ └── PingPongGameSL.ipynb
│ └── old_models/
│
├── game_combined_v1.py # SL vs RL
├── game_combined_v2.py # RL vs SL
├── Pipfile
├── Pipfile.lock
└── Readme.md or README.txt
```

---

Enjoy the game and experiment with AI agents!
