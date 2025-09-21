# Robot Learning Project – Domain Randomization and ADR

This repository contains the starting code and project extensions for the **Robot Learning (01HFNOV)** course.  

The project builds upon the provided template and explores **policy transfer in Reinforcement Learning** with the **Mujoco Hopper environment**, using **Domain Randomization (DR)** techniques.  

In addition to the baseline setup, the work extends the project by implementing **Automatic Domain Randomization (ADR)** both on the standard Hopper environment and on a **harder variant including an obstacle**, in order to test robustness and generalization capabilities.

For full details and results, see the [project report (PDF)](./Lab4ReportPlusExtension.pdf).

---

## Requirements

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
