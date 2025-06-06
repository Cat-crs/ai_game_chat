#!/bin/bash

# 打包脚本，用于将项目打包为可分发的格式

# 确保在项目根目录下运行
cd "$(dirname "$0")"

# 创建虚拟环境（如果不存在）
if [ ! -d "ai_game_chat_env" ]; then
    echo "创建虚拟环境..."
    python3 -m venv ai_game_chat_env
fi

# 激活虚拟环境
source ai_game_chat_env/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt
pip install wheel setuptools

# 清理旧的构建文件
echo "清理旧的构建文件..."
rm -rf build/ dist/ *.egg-info/

# 构建包
echo "构建包..."
python setup.py sdist bdist_wheel

# 创建发布目录
echo "创建发布目录..."
mkdir -p release

# 复制文件到发布目录
echo "复制文件到发布目录..."
cp -r dist/* release/
cp README.md LICENSE requirements.txt release/
cp -r docs release/

# 创建模型目录
mkdir -p release/models

# 打包发布目录
echo "打包发布目录..."
cd release
tar -czvf ai_game_chat-1.0.0.tar.gz *
cd ..

echo "打包完成！发布文件位于 release/ai_game_chat-1.0.0.tar.gz"

# 退出虚拟环境
deactivate

