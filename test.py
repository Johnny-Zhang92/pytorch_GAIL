

#python main.py --env-name "HalfCheetah-v2"
# --algo ppo
# --use-gae
# --log-interval 1
# --num-steps 2048
# --num-processes 1
# --lr 3e-4
# --entropy-coef 0
# --value-loss-coef 0.5
# --ppo-epoch 10
# --num-mini-batch 32
# --gamma 0.99
# --gae-lambda 0.95
# --num-env-steps 10000000
# --use-linear-lr-decay
# --use-proper-time-limits
# --gail
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--sparse', action='store_true', default=True, help='GAT with sparse version or not.')
# parser.add_argument('--seed', type=int, default=72, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
#
# args = parser.parse_args()
#
# print(args.sparse)
# print(args.seed)
# print(args.epochs)
import torch
import numpy as np
import random
if __name__ == '__main__':
#     torch.manual_seed(1)
#     #torch.cuda.manual_seed_all(args.seed)
#     np.random.seed(1)
#     for i in range(2):
#         # for j in range(2):
#         #     a = np.random.rand(3)
#         #     print(a)
#         a = np.random.rand(3)
#         print(a)
#     print("--------1----------")
#     a = np.random.rand(3)
#     print(a)
#     print("--------2----------")
#     a = np.random.rand(3)
#     print(a)
#         #print("------------------")
#     print("------*------------")
#     print("[4.17022005e-01 7.20324493e-01 1.14374817e-04]\
# [0.30233257 0.14675589 0.09233859]\
# [0.18626021 0.34556073 0.39676747]\
# [0.53881673 0.41919451 0.6852195 ]")#
 #    print("[4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01\
 # 1.46755891e-01]")
 #    for _ in range(4):
 #        np.random.seed(1)
 #        b = np.random.choice(a)
 #        print(b)

    # w = torch.empty(3, 5)
    # print(w)
    # print(torch.nn.init.orthogonal_(w))
    # a = 3 / 1
    # b = 3 // 1
    # print("a:", a, "b:", b)
    # a = [1, 2, 3, 4, 5, 6, 7, 8]
    # print(a[:8])
    # torch.manual_seed(1)
    # a = torch.randint(1, 100, (1, 9, 2))
    # #b = torch.rand(1, ).long()
    # # print(b)
    # # b1 = torch.tensor([1, 1, 0]).long()
    # # b2 = torch.tensor([1]).long()
    # # c1 = a[b1]
    # # c2 = a[b2]
    # print("a:", a)
    # # print("b1:", b1)random
    # # print("b2:", b2)
    # # print("c1:", c1)
    # # print("c2:", c2)
    # # b = a[0, 0::4]
    # b = a // 3
    # print("b:", b)
    # a = torch.tensor([1, 2, 3])
    # d = {"A": a}
    # e = d["A"]
    # f = d["A"].sum()
    # g = d["A"].sum().item()
    # print("e:", e)
    # print("f:", f)
    # print("g:", g)
    # for i, j in d.items():
    #     print("i:", i, "j:", j)
    # b = 2 > 1
    # print("b:", b)
    a = np.random.rand(3, 4, 6)
    b = np.random.rand(2, 5)
    np.savez('/home/johnny/Document/Python/baselines/data/test.npz', a=a, b=b)
    print("a:", a, "\n", "b:", b)
    data = np.load('/home/johnny/Document/Python/baselines/data/test.npz')
    print("data:", data)
    print("data['a']:", data['a'], "\n", "data['b']:", data['b'])
    print("len(data['a']):", len(data['a']))
    c = data['a'][:len(data['a'])]
    print("c:", c)
