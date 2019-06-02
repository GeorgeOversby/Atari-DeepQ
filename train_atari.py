#!/usr/bin/env python3
import argparse
from src.parameters import AtariParameters
from src.runner import Experience
from src.setup import setup_for_train


def train_dqn(params):
    agent,runner = setup_for_train(params)

    experience = Experience(params.replay_size)
    while runner.tracker.frame_count < 50_000_000:
        experience += runner.run(1)
        if len(experience) > params.replay_initial:
            batch = experience.get_minibatch(params.batch_size)
            agent.train(batch)
            runner.step()
    runner.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("atari_game", type=str, help="Name of Atari game to train")
    parser.add_argument("--dueling", action="store_true", help="Use dueling network architecture")
    parser.add_argument("--double", action="store_true", help="Use double Q-learning")
    parser.add_argument("--big", action="store_true", help="Use Bigger Conv Net")
    args = parser.parse_args()
    print(args)

    params = AtariParameters(args.atari_game + "Deterministic-v4")
    params.apply_args(args)
    train_dqn(params)