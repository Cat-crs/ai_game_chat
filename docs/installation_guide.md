# AI游戏+聊天功能安装指南

本文档提供了安装和配置AI游戏+聊天功能项目的详细步骤。

## 系统要求

- Python 3.8+
- CPU环境（无需GPU）
- 至少4GB内存
- 至少500MB磁盘空间

## 安装步骤

### 1. 克隆项目仓库

```bash
git clone https://github.com/Cat-crs/ai_game_chat.git
cd ai_game_chat
```

### 2. 创建虚拟环境

```bash
# 使用venv创建虚拟环境
python -m venv ai_game_chat_env

# 激活虚拟环境
# Windows
ai_game_chat_env\Scripts\activate
# Linux/Mac
source ai_game_chat_env/bin/activate
```

### 3. 安装依赖库

```bash
# 安装所有依赖
pip install -r requirements.txt
```

主要依赖库包括：
- torch>=1.9.0：PyTorch深度学习框架
- numpy>=1.20.0：数值计算库
- pygame>=2.0.1：游戏界面库
- nltk>=3.6.2：自然语言处理库
- tqdm>=4.61.0：进度条库
- psutil：系统资源监控库
- matplotlib：数据可视化库（可选，仅用于性能测试）

### 4. 验证安装

安装完成后，可以运行测试脚本验证安装是否成功：

```bash
# 运行功能测试
python -m tests.test_functionality

# 运行性能测试（可选）
python -m tests.test_performance
```

## 常见问题

### PyTorch安装问题

如果在安装PyTorch时遇到问题，可以访问[PyTorch官方网站](https://pytorch.org/get-started/locally/)获取适合您系统的安装命令。

### Pygame安装问题

在某些Linux系统上，可能需要先安装一些系统依赖：

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
```

### 内存不足

如果在运行时遇到内存不足的问题，可以尝试：
1. 减小MCTS的模拟次数
2. 使用较小的棋盘大小（如9x9而不是19x19）
3. 减小神经网络的大小

## 更新项目

要更新到最新版本，请执行：

```bash
git pull
pip install -r requirements.txt
```

## 联系支持

如果您在安装过程中遇到任何问题，请通过以下方式联系我们：
- 提交GitHub Issue
- 发送邮件至：ac.catmore3055@gmail.com

