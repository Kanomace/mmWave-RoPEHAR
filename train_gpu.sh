#!/bin/bash
#SBATCH --job-name=mmWaveHAR          # 作业名称
#SBATCH --gres=gpu:6000ada:1          # 请求1块RTX6000 ADA GPU
#SBATCH --time=0-02:00:00             # 最大运行时间（2小时）
#SBATCH --output=output_%j.log        # 输出日志
#SBATCH --error=error_%j.log          # 错误日志

# 加载所需模块
module load miniforge3
source activate

# 激活你的conda环境（假设你的环境名为'Py_mmWave_Roformer'）
conda activate Py_mmWave_Roformer

# 运行你的Python脚本
python /tmp/pycharm_project_491/mmWaveHAR250905.py