# Connect 4 Deep Learning

## Overview

This is a Connect 4 game coded in Python. It comes with a GUI that two users can use to play against
each other as well as an AI that you can train and play against (using the GUI as well). The AI is
trained using Deep Q-Learning and my own implementation of multi-agent reinforcement learning.

## Setup Instructions

1. Download the repository and `cd` into it:

    `git clone https://github.com/yuvasaro/connect4-deep-learning`

    `cd connect4-deep-learning`

2. Create a virtual environment and activate it:

    `python3 -m venv venv`
    
    `source venv/bin/activate`

3. Install all required libraries:

    `pip install -r requirements.txt`

4. Play Connect 4! Here are the options available:

    `python3 main.py`: Play Connect 4 with two human players (click on a column to place a coin).
    
    `python3 main.py play_ai`: Play Connect 4 against the trained AI.

    `python3 main.py train_ai`: Train the AI with deep Q-learning (takes a few minutes).

    `python3 main.py simulate`: Simulate 1000 games against an opponent that plays random moves.

    `python3 main.py console`: Play Connect 4 on the console (terminal).
