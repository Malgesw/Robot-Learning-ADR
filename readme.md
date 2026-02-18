# Robot Learning Project – Domain Randomization and ADR

## 1. Description

This repository contains the starting code and project extensions for the Polytechnic University of Turin **Robot Learning (01HFNOV)** course.  

The project builds upon the provided template and explores **policy transfer in Reinforcement Learning** with the **Mujoco Hopper environment**, using **Domain Randomization (DR)** techniques.  

In addition to the baseline setup, the work extends the project by implementing **Automatic Domain Randomization** (*OpenAI, Akkaya et al., "Solving Rubik’s Cube with a Robot Hand", 2019*) both on the standard Hopper environment and on a **harder variant including an obstacle**, in order to test robustness and generalization capabilities.

### ADR (Automatic Domain Randomization)

**ADR** adaptively changes the distribution of environment parameters during training so the agent is exposed to progressively challenging variations. At each step, the algorithm samples parameters from the current distribution, collects rollouts and updates the policy, evaluates performance, then expands, shifts or contracts the parameter ranges based on success thresholds.

<img src="./images/ADR_code.png" width=300>

### Results preview (ADR + moving obstacle, Hopper torso mass shifted)

![adr moving obs](./ADR_Moving_Obstacle.gif)

For full details and results, see the [project report (PDF)](./Lab4ReportPlusExtension.pdf).

---

## 2. Requirements

This code is designed and tested for **Linux** with **Python 3.7**.  
If you are on Windows, note that **mujoco-py is not well supported**. Running the project reliably requires WSL2 (for a lighter installation) or a Linux virtual machine.

### Dependencies

1. Install **MuJoCo** and the Python MuJoCo interface following the [official mujoco-py instructions](https://github.com/openai/mujoco-py).  
2. Install other Python dependencies (```gym``` and ```stable-baselines3```):  
   ```bash
   pip install -r requirements.txt
   ```

### Test your installation
Check your installation by running ```python test_random_policy.py```
