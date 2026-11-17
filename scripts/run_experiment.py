"""
simulate cs navigation
"""

"""
銝璅???典sumo ???亦??隞?
"""
import os, sys
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
# os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
# sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
# os.environ["PATH"] = os.environ["PATH"] + ";" + os.path.join(os.environ["SUMO_HOME"], "bin")

import traci
from traci import constants as tc
from pathlib import Path
from sumolib import checkBinary
import time
from sumolib.net import readNet
from css.agents.charging import (
    CompactReplayBuffer,
    ContextualLinUCBAgent,
    DiscreteSACAgent,
    PaperBaselineAgent,
    base_DQN,
    base_eDQN,
)
from css.charging import Env, require_route_model
from css.experiment.config import RUN_DIR, cfg_path, net_path, ROUTE_PLANNING_MODE
from css.experiment.scenarios import require_higher_evaluation_assets, validate_environment_assets
from css.experiment.environments import ENVIRONMENT_NAMES, resolve_environment
from css.experiment.training import DRL_train, evaluation, higher_evaluation, save_training_results
from css.graph import prepare_static
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def plot_train_process(total_reward, total_random = None, total_shortest = None, labelname = 'Total Reward', savename = 'results.png'):
    plt.figure()
    plt.plot(total_reward, label=labelname)
    if total_random:
        plt.plot(total_random, label="Min Waiting Reward")
    if total_shortest:
        plt.plot(total_shortest, label="Shortest Reward")
    plt.title('Reward trace')
    plt.xlabel('Episode')
    plt.ylabel('Reward(min)')
    plt.legend()
    plt.savefig(savename)

def fmt_triplet(x):
    return f"({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})"


AGENT_CHOICES = ('linucb', "sacd", "paper_baseline", "plain_dqn", "dqn", "edqn")
MODE_CHOICES = ("train", "eval", "both")


def default_route_mode_for_agent(agent_name: str) -> str:
    if agent_name == 'linucb':
        return ROUTE_PLANNING_MODE[0]
    if agent_name == "paper_baseline":
        return ROUTE_PLANNING_MODE[2]
    if agent_name == "sacd":
        return ROUTE_PLANNING_MODE[0]
    if agent_name in {"dqn", "edqn"}:
        return ROUTE_PLANNING_MODE[3]
    if agent_name == "plain_dqn":
        return ROUTE_PLANNING_MODE[1]
    raise ValueError(f"Unknown agent: {agent_name}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate CS-selection agents.")
    parser.add_argument("--env", choices=ENVIRONMENT_NAMES, default="env", help="Select the road network and its data splits.")
    parser.add_argument("--check-data", action="store_true", help="Validate selected input assets and exit without SUMO or training.")
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="sacd", help="CS-selection agent to use.")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="train", help="Run training, evaluation, or both.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--training-start-episode", type=int, default=None, help="Learning start (default: 0 for LinUCB, 250 otherwise).")
    parser.add_argument("--route-mode", choices=ROUTE_PLANNING_MODE, default=None, help="Route-planning backend. Defaults depend on --agent.")
    parser.add_argument(
        "--route-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint for the selected route-planning backend. "
            "This is separate from --checkpoint, which belongs to the charging-station agent."
        ),
    )
    parser.add_argument("--replay-buffer-size", type=int, default=100000, help="Replay buffer size for SACD.")
    parser.add_argument('--linucb-alpha', type=float, default=1.0, help='LinUCB exploration strength.')
    parser.add_argument('--linucb-ridge', type=float, default=1.0, help='LinUCB ridge regularization.')
    parser.add_argument('--linucb-linear-only', action='store_true', help='Disable squared LinUCB features.')
    parser.add_argument("--no-train-evaluation", action="store_true", help="Disable evaluation during training.")
    evaluation_group = parser.add_mutually_exclusive_group()
    evaluation_group.add_argument(
        "--higher-evaluation",
        action="store_true",
        help="Use the optional higher-evaluation dataset (must be installed).",
    )
    evaluation_group.add_argument(
        "--standard-evaluation",
        action="store_true",
        help="Use the standard testing dataset (the default).",
    )
    parser.add_argument("--use-gui", action="store_true", help="Use sumo-gui instead of sumo.")
    parser.add_argument('--sumo-trace', action='store_true', help='Record replayable TraCI commands for SUMO crash diagnosis.')
    parser.add_argument('--show-sumo-warnings', action='store_true', help='Show non-fatal SUMO warnings (hidden by default).')
    parser.add_argument('--seed', type=int, default=42, help='Seed Python, NumPy, PyTorch, and LinUCB.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path to save during training and load during evaluation.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    environment = resolve_environment(args.env)
    if args.higher_evaluation:
        require_higher_evaluation_assets(environment.root)
    data_summary = validate_environment_assets(environment)
    if args.check_data:
        print(json.dumps(data_summary, ensure_ascii=False, indent=2))
        return 0
    cfg_path, net_path = str(environment.cfg_path), str(environment.net_path)
    RUN_DIR = environment.root
    requested_route_mode = args.route_mode or default_route_mode_for_agent(args.agent)
    route_checkpoint = args.route_checkpoint
    if route_checkpoint is None:
        route_checkpoint = require_route_model(requested_route_mode, environment.name)
    else:
        if requested_route_mode == "dijkstra":
            raise ValueError("--route-checkpoint cannot be used with --route-mode dijkstra.")
        route_checkpoint = route_checkpoint.expanduser().resolve()
        if not route_checkpoint.is_file():
            raise FileNotFoundError(f"Route-planning checkpoint not found: {route_checkpoint}")
    if environment.manifest_based and args.checkpoint is None:
        args.checkpoint = str(environment.model_dir / f"{args.agent}_route_{requested_route_mode}.pt")
    if args.mode == "eval" and args.checkpoint is not None and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Charging-selection checkpoint not found: {args.checkpoint}")
    if args.checkpoint is not None and args.mode in {"train", "both"}:
        Path(args.checkpoint).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # # 1) ?曉 SUMO ??tools
    # if "SUMO_HOME" not in os.environ:
    #     sys.exit("Please declare environment variable 'SUMO_HOME'")
    # tools = Path(os.environ["SUMO_HOME"]) / "tools"
    # sys.path.append(str(tools))
    # # 2) 靽??具?撠楝敺? sumocfg嚗?極雿??銝?湛?
    # HERE = Path(__file__).resolve().parent
    # PARENT_HERE = HERE.parent ## ?望???冗雿蔭銝?嚗?頝臬?隤踹??餃??祉?


    # # 2.1) ??啣?嚗?雿輻small env
    # """
    # ### testing small env
    # """
    # using_env = "small_env"
    # RUN_DIR = PARENT_HERE / "small_env"
    # cfg_path = str(RUN_DIR / "osm.sumocfg")
    # net_path = str(RUN_DIR / "small_env.net.xml")

    # 3) ??checkBinary ?曉甇?Ⅱ??sumo-gui ?臬銵?嚗??PATH ??嚗?
    sumoBinary = checkBinary("sumo-gui" if args.use_gui else "sumo")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = environment.log_dir / f'run_{stamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    error_log = log_dir / 'sumo_error.log'
    stdout_log = log_dir / 'sumo_stdout.log'
    trace_log = log_dir / 'traci_trace.py'

    # Keep SUMO diagnostics instead of discarding stdout. A TraCI trace records
    # state-changing commands and can be replayed even when SUMO exits natively.
    cmd = [
        sumoBinary,
        '-c',
        cfg_path,
        '--step-length',
        '0.5',
        '--error-log',
        str(error_log),
    ]
    if not args.show_sumo_warnings:
        cmd.append('--no-warnings')
    sumo_stdout_handle = stdout_log.open('w', encoding='utf-8')
    try:
        start_kwargs = {'stdout': sumo_stdout_handle}
        if args.sumo_trace:
            start_kwargs.update({'traceFile': str(trace_log), 'traceGetters': False})
        traci.start(cmd, **start_kwargs)
        print(f'[sumo] error log: {error_log}')
        print(f'[sumo] stdout log: {stdout_log}')
        if args.sumo_trace:
            print(f'[sumo] TraCI trace: {trace_log}')

        net = readNet(net_path, withInternal=True)

        static_pack = prepare_static(net, prefer_controlled_tls=environment.prefer_controlled_tls)


        """
        閮剖?CS雿蔭嚗?閮毀route optimal??璅?
        """
        using_gnn = True
        # cs_loc = ["42496394", "cluster_1275778430_42504174_9744131154", "42907440"] ### for small env

        cs_loc = list(environment.stations)
        env = Env(cs_loc, static_pack, net, environment_name=environment.name,
                  route_mode=requested_route_mode, route_checkpoint=route_checkpoint)
        save_floder = str(RUN_DIR / "training_data_power")

        # """
        # ???駁? (??銝甈∪停憟?
        # ?桀??芣???training??
        # 鋆?testing??
        # """
        # def init_power():
        #     filename = str(RUN_DIR / "training_data_goal")
        #     folder = Path(filename)
        #     for file in folder.iterdir():
        #         if file.is_file() and file.suffix == ".json":
        #             with file.open("r", encoding="utf-8") as rf:
        #                 data = json.load(rf)   # ?湔??json 頧? Python ?拐辣
        #                 ev_num = len(data)
        #                 ev_power = np.round(np.random.uniform(0.2,0.4,ev_num),2)
        #                 save_data = ev_power.tolist()
        #                 with open(f"{save_floder}/power_{file.name[5:]}", "w", encoding="utf-8") as wf:
        #                     json.dump(save_data, wf, ensure_ascii=False, indent=4)

        # init_power()
        # sys.exit()

        """
        ??閮毀
        1. ?遣蝡?蝡?agent
        """ 
        ### state_dim & action_dim閬???銝?蝢抵?靘?
        if environment.manifest_based:
            state_dim, action_dim = env.state_dim, env.action_dim
            tensorboard_dir = str(PROJECT_ROOT / 'runs' / environment.name / f'{args.agent}_route_{requested_route_mode}' / stamp)
            if args.agent == 'linucb':
                bandit_agent = ContextualLinUCBAgent(env, alpha=args.linucb_alpha, ridge=args.linucb_ridge,
                                                   replay_capacity=args.replay_buffer_size,
                                                   quadratic_features=not args.linucb_linear_only,
                                                   seed=args.seed, log_dir=tensorboard_dir)
            elif args.agent == 'sacd':
                agent = DiscreteSACAgent(state_dim, action_dim, log_dir=tensorboard_dir)
                replay_buffer = CompactReplayBuffer(args.replay_buffer_size)
            elif args.agent == 'paper_baseline':
                paper_agent = PaperBaselineAgent(env, log_dir=tensorboard_dir)
            elif args.agent == 'plain_dqn':
                agent_dql = base_DQN(state_dim, action_dim, log_dir=tensorboard_dir)
            else:
                agent_edqn = base_eDQN(state_dim, action_dim)
                from torch.utils.tensorboard import SummaryWriter
                agent_edqn.writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            state_dim = env.state_dim  
            action_dim = env.action_dim
            agent = DiscreteSACAgent(state_dim, action_dim)
            REPLAY_BUFFER_SIZE = args.replay_buffer_size
            replay_buffer = CompactReplayBuffer(REPLAY_BUFFER_SIZE)
            bandit_agent = ContextualLinUCBAgent(
                env,
                alpha=args.linucb_alpha,
                ridge=args.linucb_ridge,
                replay_capacity=REPLAY_BUFFER_SIZE,
                quadratic_features=not args.linucb_linear_only,
                seed=args.seed,
            )

            agent_dql = base_DQN(state_dim, action_dim)
            agent_edqn = base_eDQN(state_dim, action_dim)
            paper_agent = PaperBaselineAgent(env)
        USE_AGENT = args.agent
        TRAIN_START_EPISODE = args.training_start_episode
        if TRAIN_START_EPISODE is None:
            TRAIN_START_EPISODE = 0 if USE_AGENT == 'linucb' else 250
        TRAIN_EPISODES = args.episodes
        RUN_EVALUATION = not args.no_train_evaluation
        RUN_HIGHER_EVALUATION = args.higher_evaluation

        ACTIVE_ROUTE_MODE = requested_route_mode
        print(
            f"[config] env={environment.name} data={environment.root} stations={cs_loc} mode={args.mode} agent={USE_AGENT} route_mode={ACTIVE_ROUTE_MODE} "
            f"episodes={TRAIN_EPISODES} training_start={TRAIN_START_EPISODE} "
            f"train_eval={RUN_EVALUATION} higher_eval={RUN_HIGHER_EVALUATION} use_gui={args.use_gui} "
            f"route_checkpoint={route_checkpoint} charging_checkpoint={args.checkpoint}"
        )

        if USE_AGENT == 'linucb':
            active_agent = bandit_agent
            active_buffer = bandit_agent.memory
        elif USE_AGENT == "paper_baseline":
            active_agent = paper_agent
            active_buffer = paper_agent.memory
        elif USE_AGENT in {"dqn", "edqn"}:
            active_agent = agent_edqn
            active_buffer = agent_edqn.memory
        elif USE_AGENT == "plain_dqn":
            active_agent = agent_dql
            active_buffer = agent_dql.memory
        elif USE_AGENT == "sacd":
            active_agent = agent
            active_buffer = replay_buffer
        else:
            raise ValueError(f"Unknown USE_AGENT: {USE_AGENT}")







        if args.mode in {"train", "both"}:
            results = DRL_train(
                env,
                active_agent,
                active_buffer,
                episodes=TRAIN_EPISODES,
                training_start_episode=TRAIN_START_EPISODE,
                run_evaluation=RUN_EVALUATION,
                using_min_waiting=True,
                route_planning_mode=ACTIVE_ROUTE_MODE,
                show_sumo_warnings=args.show_sumo_warnings,
                checkpoint_filename=args.checkpoint,
                environment=environment,
                schedule_seed=args.seed,
            )

            total_reward, total_eval_rewards, min_waiting_eval_rewards, shortest_eval_rewards, best_data, base_data_1, base_data_2 = results
            print("best eval ", best_data)
            print("min waiting ", base_data_1)
            print("shortest ", base_data_2)

            if args.mode == "train":
                print("end of training")
                return 0

        # plot_train_process(total_reward, savename = "total_rewards")
        # plot_train_process(total_eval_rewards, min_waiting_eval_rewards,shortest_eval_rewards, savename = "total eval rewards")

        # ??銝oute planning?券????迤撣?
        print("testing ========================================")
        if args.checkpoint is None:
            active_agent.load_para()
        else:
            active_agent.load_para(args.checkpoint)
        # eval_cost, _, data, rand_data = evaluation(env, agent, route_planning_mode = ROUTE_PLANNING_MODE[0]) # ?牡qngnn ?楝敺???
        # eval_cost, rand_cost, shortest_cost, eval_data_bag, rand_data_bag, shortest_data_bag = \
        #     evaluation(env, active_agent, using_min_waiting=True, using_shortest = True, route_planning_mode = ROUTE_PLANNING_MODE[0])

        if RUN_HIGHER_EVALUATION:
            eval_cost, min_waiting_cost, shortest_cost, eval_data_bag, min_waiting_data_bag, shortest_data_bag, detail = \
                higher_evaluation(env, active_agent, using_min_waiting=True, using_shortest=True, route_planning_mode=ACTIVE_ROUTE_MODE, return_details=True, environment=environment)
            eval_detail_filename = f"higher_evaluation_detail_{USE_AGENT}_{ACTIVE_ROUTE_MODE}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        else:
            eval_cost, min_waiting_cost, shortest_cost, eval_data_bag, min_waiting_data_bag, shortest_data_bag, detail = \
                evaluation(env, active_agent, using_min_waiting=True, using_shortest=True, route_planning_mode=ACTIVE_ROUTE_MODE, return_details=True, environment=environment)
            eval_detail_filename = f"evaluation_detail_{USE_AGENT}_{ACTIVE_ROUTE_MODE}_{time.strftime('%Y%m%d-%H%M%S')}.json"

        eval_output_dir = environment.output_dir
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_detail_path = eval_output_dir / eval_detail_filename
        save_training_results(detail, eval_detail_path)

        print(f"agent {fmt_triplet(eval_data_bag)}")
        print(f"min waiting {fmt_triplet(min_waiting_data_bag)}")
        print(f"shortest {fmt_triplet(shortest_data_bag)}")
        print(f"detail saved to {eval_detail_path}")

        # # eval_cost, _, data, rand_data = evaluation(env, agent, route_planning_mode = ROUTE_PLANNING_MODE[2]) # ?牡ijkstra ?楝敺???
        # eval_cost, rand_cost, shortest_cost, eval_data_bag, rand_data_bag, shortest_data_bag = \
        #     evaluation(env, agent_dql, using_random = True, using_shortest = True, route_planning_mode = ROUTE_PLANNING_MODE[2])

        # print(f"dqngnn {eval_data_bag}")
        # print(f"rand {rand_data_bag}")
        # print(f"shortest {shortest_data_bag}")

        return 0
    finally:
        try:
            traci.close()
        except Exception:
            pass
        sumo_stdout_handle.close()





if __name__ == "__main__":
    raise SystemExit(main())
