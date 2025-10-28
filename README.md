# AI_Bowler in the sport of cricket
(Under further research and development)
________________________________________________________________
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
The ball trajectory at certain instances, misses few frames along the extraction process. To counter this, removal outfliers was performed followed by interpolation of ball coordinates in the missing frames.

Data Tranformation:
The cleaned data is then used to extract information/parameters like line, length and velocity of the ball along the trajectory.

1. Line: average of ball x-axis along the ball trajectory.

2. Length: trend analyzed based on z axis(extracted from MiDAS) and y axis of the ball along its trajecotry. The y-axis coordinate, increases as the ball progresses towards the batsman, when it bounces off the pitch the trend reverses for a small period of time, this is 
compared with z-axis data to ensure if the ball pitches, else it is considered as full toss.

3. Velocity: Tricky part using aproximation of depth vector computed using pre trained MiDaS model, at times it provides outliers(handled in outlier cleaning). Using the depth vector to find velocity along the sliding window context, getting the median to get better results.

This leaves us with reinforcement part and csv dataset of timesseries data with varying lengths of instances.

🕹️ Reinforcement Learning Agent:
The agent is trained using this posture and trajectory dataset to learn:

Patterns in the batsman's movement
Weaknesses or timing issues - manual annotation
Legal bowling strategies that reduce the quality of the shot or induce misses
The reinforcement learning algorithm used: Deep Deterministic policy gradients (DDPG) with LSTM(Long short term Memory) based actor network, which iterates through the varying length of timeseries data(this omits 2 problems: lossy data or padding with zeros-more parameters to train).

The reward system desinged penalizes the agent severly illegal deliveries like wide or noball, and penalizes if the batsman succeeds. The agent is positively rewarded only when the batsman misses the ball and is legal delivery. 
Different reward systems tested in the python notebooks.

### Results
The DDPG model outperforms, random action picker and supervised models like behavior clonning model(LSTM based model) and reward biased behavior clonning model.
Note: The issue faced was with limitation in recorded data, this study should be online learning instead it had to be simulated as an online learning experiment given time and facility constraints.

🔧 Tools & Technologies
Computer Vision: OpenCV, Mediapipe (for pose estimation), Yolov8 (for ball tracking and stump detection)
Depth estimation: MiDaS (depth approximation neural network)
Reinforcement Learning: DDPG
Data Processing: Pandas, NumPy
Environment Simulation : Custom cricket simulation setup

💡 Future Scope
1. Integration of the DDPG agent with a bowling machine, to automate the ball feeding and delivery planning process.
2. Real-time feedback system for bowlers using wearable sensors
3. Expansion to 1-on-1 AI-powered cricket training systems
4. Modeling batter’s skill progression and adapting RL difficulty dynamically
5. Integration with smart cameras for fully automated coaching
________________________________________________________________
_________________________________________________________________
## Dependencies
Download the requirements specified in the requirements.txt file which includes mediapipe, opencv, ultralytics, tensorflow, pandas and numpy to name a few.
________________________________________________________________
## License
_________________________________________________________________
## Author
- Name: Srikumar A
- contact: arivlagansrikumar@gmail.com
