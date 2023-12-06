# Connect 4 AI

## Status: In Progress

I am currently trying to tweak the neural network and reward system to see if I can get the AI to
improve at the game. Currently, it can beat a random opponent (that plays random moves) around
60-85% of the time, but doesn't seem to be that great at the game when I myself play against it.
My own implementation of training the model using two agents that are playing against each other 
might not be as useful as I hoped in training the model. I'm currently researching ways to
improve the model.

## Overview

This is a Connect 4 game coded in Python. It comes with a GUI that two users can use to play against
each other as well as an AI that you can train and play against (using the GUI as well). The AI is
trained using Deep Q-Learning and my own implementation of multi-agent reinforcement learning.

## Setup Instructions

1. Download the repository and `cd` into it:

    `git clone https://github.com/yuvasaro/connect4ai`

    `cd connect4ai`

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
