# LunarLander PPO Agent

## 📌 Purpose
This project implements a **Proximal Policy Optimization (PPO)** agent to solve the **LunarLander-v3** environment from OpenAI Gymnasium.  

- **Training:** The `train_agent.py` script trains the PPO agent and saves the best policy as `best_policy.npy`.  
- **Evaluation:** The `evaluate_agent.py` script loads the saved policy and evaluates it over multiple episodes.  
- **Policy Handling:** The `my_policy.py` script defines how to load the saved policy and determine actions for given states.  

This setup is useful for **reinforcement learning experiments**, showcasing policy training, saving in NumPy format, and reloading for inference.

---

## 🛠 Tech Stack
- **Programming Language:** Python 3.8+  
- **Machine Learning Library:** [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO implementation)  
- **Environment:** [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (LunarLander-v3)  
- **Deep Learning:** PyTorch  
- **Data Handling:** NumPy  

---

## 🚀 How to Run the Project

### 1️⃣ Prerequisites
Install Python dependencies:
```bash
pip install gymnasium[box2d] stable-baselines3 torch numpy
```

---
 ### **2️⃣ Training the Agent**
 ```bash
python train_agent.py
```
This will:

- Train the PPO agent on LunarLander-v3.
- Save the trained model (ppo_trained_lander).
- Save the best policy weights as best_policy.npy.

  
### **3️⃣ Evaluating the Agent**
```bash
python evaluate_agent.py --filename best_policy.npy --policy_module my_policy
```
This will:

- Load the saved policy.
- Evaluate it over 100 episodes (rendering the first 5).
- Print per-episode rewards and the average score.
  
---

## ⚙️ How it works 

- `train_agent.py` uses stable-baselines3 PPO to train on LunarLander-v3. After training it extracts agent_model.policy.state_dict() and stores all arrays in a 0-D structured NumPy array, then saves best_policy.npy.

- `my_policy.py` reads best_policy.npy, converts tensors to PyTorch tensors, constructs a forward pass (MLP) and returns a discrete action by argmax.

- `evaluate_agent.py` loads the .npy policy, imports the policy module by name, and repeatedly runs episodes using the provided policy_action function.


---

## 💻 Running on Another Computer
To run on another machine:

- Install Python and dependencies listed above.
- Copy all project files (train_agent.py, evaluate_agent.py, my_policy.py, best_policy.npy).
- Ensure best_policy.npy is in the correct path or update --filename accordingly.
- Run evaluation or retraining as needed.

---
## 📂 Project Files
- `train_agent.py` → Trains the PPO agent and saves the policy.
- `evaluate_agent.py` → Loads and evaluates the saved policy.
- `my_policy.py` → Defines the policy loading and action selection logic.
- `best_policy.npy` → NumPy file storing trained policy weights (structured array format).
  
---
 
## LICENSE
- Use / modify the code for learning and research. Attribution appreciated but not required.

 
