# Gym stuff
import gym
import gym_anytrading
#from gym_anytrading.envs import StocksEnv   #Change it to local Class After Change
from stable_baselines3.common.noise import NormalActionNoise

from StockEnv import StocksEnv

from stable_baselines3 import TD3, PPO
from stable_baselines3 import DQN


# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C


# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from finta import TA

df = pd.read_csv('data/IBM_Testing_Data 1 month.txt')

df.sort_values("Date", ascending=True, inplace=True)

print(df.head())
df["Open"] = pd.array(df["Open"], 'float64')
df["High"] = pd.array(df["High"], 'float64')
df["Low"] = pd.array(df["Low"], 'float64')
df["Close"] = pd.array(df["Close"], 'float64')
df["Volume"] = df["Volume"]             \
    # .apply(lambda x: float(x.replace(",","")))


df['Date'] = pd.to_datetime(df['Date'])
# df['']
print(df.dtypes)

df.set_index('Date', inplace=True)
df["SMA"] = TA.SMA(df, 12)
df["RSI"] = TA.RSI(df)
df.fillna(0,inplace=True)
print(df.head())

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, "Low"].to_numpy()[start:end]
    signal_features = env.df.loc[:, ["Low", "Volume", "SMA", "RSI"]].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals

env = MyCustomEnv(df=df, frame_bound=(5,200), window_size=5)
print("Environment: ",env.signal_features)


print("action_space: ", env.action_space)

state = env.reset()
while True:
    action = env.action_space.sample()
    print(action)
    n_state, reward, done, info = env.step(action)
    if done:
        print("info", info)
        print("Final position: ", env.last_position)
        break

plt.figure(figsize=(15, 6))
plt.cla()
env.render_all()
plt.show()

env_maker = lambda: env
env = DummyVecEnv([env_maker])

print("Len: ", len(env.envs))
print("env: ", env.envs[0])
profits_and_rewards = []

highest_profit = 0
highest_loss = 0
highest_profited_model = None

for x in range(100):

    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000,)
    model.save("a2c_cartpole")


    # n_actions = 5
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    #
    # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    # model.learn(total_timesteps=1000, log_interval=10)
    #
    # env = MyCustomEnv(df=df, frame_bound=(90,190), window_size=5)

    # model = DQN("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=5)
    # model.save("dqn_cartpole")

    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=100000)
    # model.save("ppo_cartpole")

    env = MyCustomEnv(df=df, frame_bound=(190,210), window_size=5)

    obs = env.reset()
    print(obs)

    while True:
        print(obs.shape)
        print(obs)
        print("Loop 1")
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("info", info)
            print("Final position: ", env.last_position)
            print("State: ", _states)
            action, _states = model.predict(obs)
            print("Next prediction: ", action)
            profits_and_rewards.append([env.total_profit, env.total_reward])
            if highest_profit < env.total_profit:
                sec_profited_mod = highest_profited_model
                highest_profit = env.total_profit
                highest_profited_model = model
            break

profits_and_rewards = np.asarray(profits_and_rewards)
profit_count = 0
loss_count = 0
equal_count = 0
highest_profit = 0
highest_loss = 1

print(profits_and_rewards.shape)
with open('data/profits.txt', 'a') as f:
    for i in range(len(profits_and_rewards)):

        if profits_and_rewards[i][0] > 1:
            if highest_profit < profits_and_rewards[i][0]:
                highest_profit = profits_and_rewards[i][0]
            profit_count += 1
        elif profits_and_rewards[i][0] < 1:
            if highest_loss > profits_and_rewards[i][0]:
                highest_loss = profits_and_rewards[i][0]
            loss_count += 1
        else: equal_count += 1

        f.write(str(profits_and_rewards[i][0]))
        f.write(", ")
        f.write(str(profits_and_rewards[i][1]))
        f.write("\n")

print("Profits: ", profit_count, "Losses: ", loss_count, "Equals: ", equal_count)
print("Highest profit: ", highest_profit, "Highest loss: ", highest_loss)

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

model = highest_profited_model

obs = env.reset()
while True:

    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        print("Final position: ", env.last_position)
        print("State: ", _states)
        action, _states = model.predict(obs)
        print("Next prediction: ", action)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

env = MyCustomEnv(df=df, frame_bound=(190, 210), window_size=5)
obs = env.reset()
while True:

    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        print("Final position: ", env.last_position)
        print("State: ", _states)
        action, _states = model.predict(obs)
        print("Next prediction: ", action)
        break;

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

print("Highest profit: ", highest_profit, "Highest loss: ", highest_loss)

trade_start = 190
trade_end = 210

# for times in range(100):
#
#     trade_start += 1
#     trade_end += 1
#     next_env = MyCustomEnv(df=df, frame_bound=(trade_start, trade_end), window_size=5)
#     print(next_env.observation)
#     obs = env.set_for_next_env(next_env)
#     print(obs)
#
#     while True:
#
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         if done:
#             print("info", info)
#             print("Final position: ", env.last_position)
#             print("State: ", _states)
#             action, _states = model.predict(obs)
#             print("Next prediction: ", env.last_position)
#             break
#     if trade_start % 5 == 0:
#         plt.figure(figsize=(15,6))
#         plt.cla()
#         env.render_all()
#         plt.show()
# print("Highest profit: ", highest_profit, "Highest loss: ", highest_loss)

trade_end += 100

env = MyCustomEnv(df=df, frame_bound=(trade_start, trade_end), window_size=5)
obs = env.reset()
while True:

    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        print("Final position: ", env.last_position)
        print("State: ", _states)
        action, _states = model.predict(obs)
        print("Next prediction: ", action)
        break;

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()