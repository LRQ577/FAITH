# FAITH :Frequency-domain Attention In Two Horizon for TSF


Time series forecasting is a critical demand for real applications. Enlighted by the classic time series analysis and stochastic process theory, we propose the FAITH as a general series forecasting model [paper](https://arxiv.org/pdf/2405.13300) .

FAITH  captures inter-channel relationships and temporal global information in the sequence. Extensive experiments on 6 benchmarks for long-term forecasting and 5 benchmarks for short-term forecasting demonstrate that FAITH outperforms existing models in many fields, such as electricity, weather and traffic, proving its effectiveness and superiority both in long-term and short-term time series forecasting tasks.

# The overall FAITH framework

![fig22](https://github.com/LRQ577/FAITH/assets/119293404/5844097a-ef86-4c25-bb45-c31fbfa15c2d)

# FCTEM of FAITH
![fig44](https://github.com/user-attachments/assets/ffc74c52-5edc-421d-b162-25e2c5205b24)

# Experiment
![image](https://github.com/user-attachments/assets/ac23ab86-7879-4b3b-bea4-445e67c10e35)

![image](https://github.com/user-attachments/assets/639bb785-ad70-40c0-99b1-2ff5d9e9d92c)

![image](https://github.com/user-attachments/assets/ff6fd590-45c2-4587-9ac3-477522881505)

# Start
1. ```pip install -r requirement.txt ```
2. Dataset. All the six long-term benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view) .
3. Reproducibility. We provide the experiment scripts under the folder ./scripts. You can reproduce the experiments results by:
   ```
   bash run_all.sh
   ```


# Contact 
If you have any questions, please contact 1284042551@qq.com. Welcome to discuss together.

# Citation
If you find this repo useful, please cite our paper
```
@article{FAITH,
  title={FAITH :Frequency-domain Attention In Two Horizon for TSF},
  year={2024}
}
```
# Acknowleddgement
We appreciate the following github repos a lot for their valuable code base or datasets:

https://zhuanlan.zhihu.com/p/603468264

https://github.com/MAZiqing/FEDformer

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://github.com/Zero-coder/MLGN?tab=readme-ov-file
