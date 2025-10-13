# InstructFlow: Adaptive Symbolic Constraint-Guided Code Generation for Long-Horizon Planning
This is the official implementation of our work "InstructFlow: Adaptive Symbolic Constraint-Guided Code Generation for Long-Horizon Planning" (NeurIPS 2025). 
![image](https://github.com/Chi-haotian/InstructFlow/blob/main/figures/Overview.png)

# Setup
<pre lang="md">conda create -n "instructflow" python=3.10
conda activate instructflow
python -m pip install -e . </pre>

# Example commands
<pre lang="md">python eval_policy.py --config-dir . --config-name=instructflow_draw_star.yaml</pre>

# Acknowledgement
Our method is developed based on [PRoC3S](https://github.com/Learning-and-Intelligent-Systems/proc3s), and we gratefully acknowledge their invaluable code and prompt. Please refer to the original PRoC3S publication for environment setup and further implementation details.
