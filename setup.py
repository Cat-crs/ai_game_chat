from setuptools import setup, find_packages

setup(
    name="ai_game_chat",
    version="1.0.0",
    author="Cat",
    author_email="ac.catmore3055@gmail.com",
    description="AI游戏（五子棋/围棋）和聊天功能",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Cat-crs/ai_game_chat",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "pygame>=2.0.1",
        "nltk>=3.6.2",
        "tqdm>=4.61.0",
        "psutil",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "ai_game_chat=ai_game_chat.main:main",
        ],
    },
)

