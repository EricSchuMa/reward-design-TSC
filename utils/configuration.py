import configparser

from stable_baselines3 import DQN
from reward.reward_model import get_reward_fn, get_normalization_ranges
from sumo_rl import SumoEnvironment


def generate_agent(environment, agent_config, net_arch):
    if agent_config.get('agent') == 'DQN':
        return DQN("MlpPolicy",
                   environment,
                   verbose=0,
                   gamma=agent_config.getfloat('gamma'),
                   max_grad_norm=agent_config.getfloat('max_grad_norm'),
                   batch_size=agent_config.getint('batch_size'),
                   train_freq=agent_config.getint('train_freq'),
                   learning_starts=agent_config.getint('learning_starts'),
                   learning_rate=agent_config.getfloat('learning_rate'),
                   buffer_size=agent_config.getint('buffer_size'),
                   exploration_final_eps=agent_config.getfloat('exploration_final_eps'),
                   exploration_fraction=agent_config.getfloat('exploration_fraction'),
                   policy_kwargs=dict(net_arch=[int(size) for size in net_arch.values()])
                   )
    else:
        raise NotImplementedError(f"Supported agent is DQN not {agent_config.get('agent')}")


def generate_env_from_config(config, reward_config):
    return SumoEnvironment(net_file=config.get('net_file'),
                           route_file=config.get('route_file'),
                           single_agent=config.getboolean('single_agent'),
                           reward_fn=get_reward_fn(config, reward_config),
                           reward_norm_ranges=get_normalization_ranges(config),
                           use_gui=config.getboolean('use_gui'),
                           delta_time=config.getint('delta_time'),
                           yellow_time=config.getint('yellow_time'),
                           min_green=config.getint('min_green'),
                           observation_c=config.getint('observation_c'),
                           observation_fn=config.get('observation_fn'),
                           num_seconds=config.getint('num_seconds'),
                           reward_config=reward_config,
                           norm_config=config)


def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.config_path)
    overwrite_reward_config(args, config)
    return (config['AGENT_CONFIG'],
            config['ENV_CONFIG'],
            config['REWARD_CONFIG'],
            config['NET_ARCH'],
            config['TRAIN_CONFIG'])


def overwrite_reward_config(args, config):
    if hasattr(args, "queue") and args.queue:
        config.set("REWARD_CONFIG", "queue", args.queue)
    if hasattr(args, "brake") and args.brake:
        config.set("REWARD_CONFIG", "brake", args.brake)
    if hasattr(args, "speed") and args.speed:
        config.set("REWARD_CONFIG", "average_speed", args.speed)
    if hasattr(args, "diff_wait") and args.diff_wait:
        config.set("REWARD_CONFIG", "diff_waiting_time", args.diff_wait)
    if hasattr(args, "wait") and args.wait:
        config.set("REWARD_CONFIG", "wait", args.wait)
