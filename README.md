# AI游戏+聊天功能

这是一个结合了AI游戏（五子棋/围棋）和聊天功能的Python项目。AI使用PyTorch实现，具有自我学习能力，可以在CPU环境下流畅运行。

## 主要特点

- **五子棋和围棋游戏**：实现了完整的游戏规则和图形界面
- **基于PyTorch的AI**：使用卷积神经网络和蒙特卡洛树搜索算法
- **自我学习功能**：AI可以通过自我对弈不断提升棋力
- **聊天功能**：可以回答围棋和五子棋相关问题，并提供下棋指导
- **CPU优化**：针对CPU环境进行了优化，无需GPU也能流畅运行
- **自适应性能**：根据系统资源自动调整参数，以获得最佳性能

## 安装

详细的安装步骤请参考[安装指南](docs/installation_guide.md)。

简要步骤：

```bash
# 克隆仓库
git clone https://github.com/yourusername/ai_game_chat.git
cd ai_game_chat

# 创建虚拟环境
python -m venv ai_game_chat_env
source ai_game_chat_env/bin/activate  # Linux/Mac
# 或
ai_game_chat_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行应用
python -m ai_game_chat.main
```

## 使用方法

启动应用后，您将看到主菜单界面，可以选择以下功能：

- **五子棋**：开始一局五子棋游戏
- **围棋**：开始一局围棋游戏
- **聊天**：与AI聊天，询问围棋和五子棋相关问题
- **AI自我学习**：让AI通过自我对弈提升棋力
- **设置**：调整游戏和AI参数

详细的使用方法请参考[用户手册](docs/user_manual.md)。

## 项目结构

```
ai_game_chat/
├── game/       # 游戏逻辑模块
├── ai/         # AI模型和算法
├── chat/       # 聊天功能
├── gui/        # 图形界面
├── utils/      # 工具函数
├── tests/      # 测试模块
├── docs/       # 文档
└── models/     # 模型保存目录
```

详细的项目结构和开发指南请参考[开发文档](docs/developer_guide.md)。

## 性能表现

在普通CPU环境下（双核处理器，4GB内存）：

- 模型推理时间：1-2毫秒/步
- MCTS决策时间（100次模拟）：约2秒/步
- 自我对弈速度：约5步/秒
- 内存使用：约200MB

## 贡献

欢迎贡献代码、报告问题或提出改进建议！请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 详情请参见[LICENSE](LICENSE)文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- GitHub Issues
- 电子邮件：ac.catmore3055@gmail.com

