import copy
import glob
import os
import time
import csv
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # 固定随机种子 args.seed=
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    #torch.manual_seed(args.seed)
    # 为所有的GPU设置种子
    #torch.cuda.manual_seed_all(args.seed)
    # 没有使用GPU的时候设置的固定生成的随机数
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False #为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
        torch.backends.cudnn.deterministic = True #固定每次返回的卷积算法

    log_dir = os.path.expanduser(args.log_dir) #把Linux ~展开
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir) #在指定路径建立文件，若已存在，则删除后重新创建。
    utils.cleanup_log_dir(eval_log_dir) #在指定路径建立文件，若已存在，则删除后重新创建。

    torch.set_num_threads(1) #限制Pytorch占用过多CPU资源。
    device = torch.device("cuda:0" if args.cuda else "cpu") # 选择CPU 或 GPU

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})#main属性：卷积网络，用于卷积输入图像；
                                                         #critic_linear属性：全连接网络，可能用于记录Q值。
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower())) #./ gail_experts / trajs_halfcheetah.pt
        # ------------start - ---------------
        # args.gail_experts_dir:./ gail_experts
        # trajs_{}.pt.format(args.env_name.split('-')[0].lower())): trajs_halfcheetah.pt
        # args.env_name.split('-'): ['HalfCheetah', 'v2']
        # args.env_name.split('-')[0].lower(): halfcheetah
        #file_name:./ gail_experts / trajs_halfcheetah.pt
        # - ------------end - ----------------
        # print("------------start----------------")
        # print("args.gail_experts_dir:", args.gail_experts_dir)
        # print("trajs_{}.pt.format(args.env_name.split('-')[0].lower())):", "trajs_{}.pt".format(
        #         args.env_name.split('-')[0].lower()))
        # print("args.env_name.split('-'):", args.env_name.split('-'))
        # print("args.env_name.split('-')[0].lower():", args.env_name.split('-')[0].lower())
        #print("file_name:", file_name)
        # print("-------------end-----------------")

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    episode_len = 0
    episode_return = 0
    episode_num = 0
    total_steps = 0

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            #print("value:", value, "action:", action, "action_log_prob:", action_log_prob, "recurrent_hidden_stes:", recurrent_hidden_states)
            # value: tensor([[-1.6006]], device='cuda:0')
            # action: tensor([[0.2846, 0.4442, 0.1657, -1.0094, -1.7039, 0.6084]],
            #                device='cuda:0')
            # action_log_prob: tensor([[-7.2011]], device='cuda:0')
            # recurrent_hidden_stes: tensor([[0.]], device='cuda:0')
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            episode_len += 1
            episode_return += reward
            total_steps += 1


            if done:
                data = [episode_return, episode_len, total_steps]
                cav_path = "data/csv/GAIL_pytorch_Ant.csv"
                with open(cav_path, "a+", newline='') as f:
                    # print("-----------{Ant_sam.csv} added a new line!------------".format())
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(data)
                episode_return, episode_len = 0, 0

                episode_len = 0
                episode_num += 1
                episode_return = 0

            #print("obs:", obs,"rewards:", reward,"donne:", done,"infos:", infos)
#           obs: tensor([[-0.2471,  0.5996, -0.4484,  1.2435,  0.1895,  1.3830, -0.6217,  0.7217,
#                         -0.6454,  2.1233, -0.8465, -0.5543,  1.2418, -0.9192,  2.0461, -0.7358,
#                         0.9339]], device='cuda:0')
#            rewards: tensor([[-0.0649]]) donne: [False]
#            infos: [{'reward_run': -0.6749512241805, 'reward_ctrl': -0.7481081485748291}]
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
