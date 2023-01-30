import argparse
import configparser
import os
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from stable_baselines3 import DQN

from utils.actors import CyclicAgent
from utils.configuration import generate_env_from_config
from utils.logger import log_episode_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help="experiment config path")
    parser.add_argument('--cyclic-agent', type=bool, default=False, required=False, help="whether to use a cyclic actor")
    parser.add_argument('--interaction-interval', type=int, default=3, required=False, help="interaction interval of cyclic actor")
    parser.add_argument('--out-csv-name', type=str, required=False, help="output file")
    parser.add_argument('--run-id', type=str, required=False, help="id of the mlflow run")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)
    eval_config, reward_config = (config['EVAL_CONFIG'], config['REWARD_CONFIG'])

    env = generate_env_from_config(eval_config, reward_config)
    env.out_csv_name = getattr(args, 'out_csv_name', None)
    env.record_trip_info = True

    if not args.cyclic_agent:
        training_run = mlflow.get_run(args.run_id)
        artifacts = [f.path for f in MlflowClient().list_artifacts(args.run_id)][0]
        model_path = os.path.join(unquote(urlparse(training_run.info.artifact_uri).path), artifacts)

        agent = DQN.load(model_path)
    else:
        agent = CyclicAgent(n_actions=2, interaction_interval=args.interaction_interval)

    # tracking of episode rewards
    rewards = []
    episode = 0
    ep_reward = 0

    obs = env.reset()
    mlflow.set_experiment('eval')
    with mlflow.start_run(run_name=args.run_id):
        for step in range(eval_config.getint('eval_timesteps')):
            if args.cyclic_agent:
                # cyclic agent needs the step to calculate the current action
                action = agent.predict(step)
            else:
                action, state = agent.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            ep_reward += r
            mlflow.log_metric('reward', r, step)

            if done:
                print(f"Total episode reward is: {ep_reward}")
                obs = env.reset()
                log_episode_metrics(env, ep_reward, episode)
                rewards.append(ep_reward)

                episode += 1
                ep_reward = 0

        print(f'Mean episode reward over {len(rewards)} episodes is : {np.mean(rewards)}')
