# Gym stuff
# import datareader as datareader
# import gym
# import gym_anytrading
# #from gym_anytrading.envs import StocksEnv   #Change it to local Class After Change
# from stable_baselines3.common.noise import NormalActionNoise

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
import pandas_datareader.data as web
import datetime as dt

from function import FunctionMaker

slice_len = 1500
df = pd.read_csv('data/IBM_Testing_Data 1 month.txt')

# start = dt.datetime(2010,1,1,8,30)
# end = dt.datetime(2022,1,17,5,00)
# df = web.DataReader("AXP", "yahoo", start, end)
# df.sort_values("Date", ascending=True, inplace=True)
# print(df.head())

print(df.head())
df["Open"] = pd.array(df["Open"], 'float64')
df["High"] = pd.array(df["High"], 'float64')
df["Low"] = pd.array(df["Low"], 'float64')
df["Close"] = pd.array(df["Close"], 'float64')
df["Volume"] = df["Volume"]             \
    # .apply(lambda x: float(x.replace(",","")))
# ****************************************************************************************
prices = df["Low"]
print(prices[0])
window = 5
curr = 0
last_price = None
price_trends = []
for i in range(len(prices)):
    slice = []
    ups = 0
    downs = 0
    status = None

    for j in range(window):
        if curr < 0:
            slice.append(None)
        if i+j > len(prices) - window - 1:
            break
        curr_price = prices[i + j]
        # print("Last_Price: ", last_price)
        # print("curr_Price: ", curr_price)
        # print("dif: ", last_price - curr_price)
        if last_price is not None:
            if last_price > curr_price:
                downs += 1
                slice.append(0)
            else:
                ups += 1
                slice.append(1)
        last_price = curr_price
    if ups == window * 0.8 and downs == window * 0.2:
        status = 2
    elif ups == window * 0.2 and downs == window * 0.8:
        status = 0
    else:
        status = 1
    print("Trend Shape: ", slice, "Status: ", status)
    price_trends.append(status)
# ****************************************************************************************

df['Status'] = price_trends
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
    signal_features = env.df.loc[:, ["High", "Volume", "SMA", "RSI", "Status"]].to_numpy()[start:end]
    return prices, signal_features

trend_window = 5

class MyCustomEnv(StocksEnv):
     _process_data = add_signals




env = MyCustomEnv(df=df, frame_bound=(5,slice_len), window_size=5)
function = FunctionMaker()
unique_frames = env.makeUniqueTrendFrame(window=trend_window)
function.new(unique_frames[0], unique_frames[1], env.makeTrendFrame(window=trend_window))
env.setConditions(function.conditions)
print("Environment: ", env.signal_features)
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
# steps = len(function.indexes)
# for i in range(steps):
#     function.updateCondition(env.total_profit, env.num_trades)
#     env = MyCustomEnv(df=df, frame_bound=(5, slice_len), window_size=5)
#     env.setConditions(function.conditions)
#     print("Environment: ", env.signal_features)
#     print("action_space: ", env.action_space)
#
#     state = env.reset()
#     while True:
#         action = env.action_space.sample()
#         print(action)
#         n_state, reward, done, info = env.step(action)
#         if done:
#             print("info", info)
#             print("Final position: ", env.last_position)
#             break
#     print("Profits: ", function.profit, "Trades: ", function.trade)
#     if i == steps-1:
#         break
#
#
#
# # function.profit = env.total_profit
# # function.Trades = env.num_trades
#
# function.updateCondition(env.total_profit,env.num_trades)
# env = MyCustomEnv(df=df, frame_bound=(5,slice_len), window_size=5)
# env.setConditions(function.conditions)
# print("Environment: ", env.signal_features)
# print("action_space: ", env.action_space)
#
# state = env.reset()
# while True:
#     action = env.action_space.sample()
#     print(action)
#     n_state, reward, done, info = env.step(action)
#     if done:
#         print("info", info)
#         print("Final position: ", env.last_position)
#         break
# function.profit = env.total_profit
# function.Trades = env.num_trades
# print("Profits: ", function.profit, "Trades: ", function.trade)
# plt.figure(figsize=(15, 6))
# plt.cla()
# env.render_all()
# plt.show()

env_maker = lambda: env
env = DummyVecEnv([env_maker])

print("Len: ", len(env.envs))
print("env: ", env.envs[0])
profits_and_rewards = []

highest_profit = 0
highest_loss = 0
highest_profited_model = None

for x in range(10):

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

    env = MyCustomEnv(df=df, frame_bound=(slice_len + 10, slice_len + 80), window_size=5)
    env.setConditions(function.conditions)

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

env = MyCustomEnv(df=df, frame_bound=(slice_len + 10, slice_len + 80), window_size=5)
env.setConditions(function.conditions)
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

trade_start = slice_len + 10
trade_end = slice_len + 80

print("Chart Loop Starting")
# env = MyCustomEnv(df=df, frame_bound=(trade_start, trade_end), window_size=5)
# obs = env.reset()

for times in range(100):

    trade_start += 1
    trade_end += 1
    print("Obs: ",obs.shape)
    next_env = MyCustomEnv(df=df, frame_bound=(trade_start, trade_end), window_size=5)
    print(next_env.observation)
    obs = env.set_for_next_env(next_env)
    print(obs)

    while True:

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("info", info)
            print("Final position: ", env.last_position)
            print("State: ", _states)
            action, _states = model.predict(obs)
            print("Next prediction: ", env.last_position)
            break
    if trade_start % 5 == 0:
        plt.figure(figsize=(15,6))
        plt.cla()
        env.render_all()
        plt.show()
print("Highest profit: ", highest_profit, "Highest loss: ", highest_loss)

trade_start = slice_len + 10
trade_end = slice_len + 430

env = MyCustomEnv(df=df, frame_bound=(trade_start, trade_end), window_size=5)
env.setConditions(function.conditions)
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