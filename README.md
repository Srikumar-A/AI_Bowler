# AI_Bowler
_________________________________________________________________
## Introduction
🎯 Objective
The goal is to train a reinforcement learning (RL) agent that acts as a bowler and plays against a human batter. The RL agent learns to:

- Bowl deliveries that are legal (per cricket rules)
- Strategically adjust line, length, and speed
- Maximize the chances that the batsman:
    Misses the ball, or
    Does not hit it convincingly

The human player's objective is to respond with optimal batting technique, while the AI agent adapts to exploit observed weaknesses over time.

🧩 How It Works
📹 Input: Cricket Batting Videos
Videos of batters practicing in a controlled environment serve as the primary input.

🧠 Preprocessing Pipeline:

Human pose estimation (to extract batting postures and movement) - mediapipe

Ball trajectory tracking (frame-by-frame detection of ball coordinates) - YOLOv8 model

Output: A structured dataset (CSV) containing posture keypoints + ball position per frame

🧹 Data Cleaning:
The raw extracted data is cleaned and normalized to remove noise, missing values, and outliers before being fed into the RL agent.

🕹️ Reinforcement Learning Agent:
The agent is trained using this posture and trajectory dataset to learn:

Patterns in the batsman's movement
Weaknesses or timing issues - manual annotation
Legal bowling strategies that reduce the quality of the shot or induce misses
The reinforcement learning algorithm used: Deep Deterministic policy gradients (DDPG)

🔧 Tools & Technologies
Computer Vision: OpenCV, Mediapipe (for pose estimation), Yolov8 (for ball tracking and stump detection)
Depth estimation: MiDaS (depth approximation neural network)
Reinforcement Learning: DDPG
Data Processing: Pandas, NumPy
Environment Simulation (if virtual testbed used): Custom cricket simulation setup or synthetic environment

💡 Future Scope
1. Integration of the DDPG agent with a bowling machine, to automate the ball feeding and delivery planning process.
2. Real-time feedback system for bowlers using wearable sensors
3. Expansion to 1-on-1 AI-powered cricket training systems
4. Modeling batter’s skill progression and adapting RL difficulty dynamically
5. Integration with smart cameras for fully automated coaching
________________________________________________________________
_________________________________________________________________
## Dependencies
________________________________________________________________
## License
_________________________________________________________________
## Author
- Name: Srikumar A
- contact: arivlagansrikumar@gmail.com
