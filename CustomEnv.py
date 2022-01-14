import random as random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Actions(Enum):
    Sell = 0
    Buy = 1
    ShortBuy = 2
    ShortSell = 3
    Hold = 4


class Positions(Enum):
    Short = 0
    Long = 1
    ShortSell = 2
    LongSell = 3
    Hold = 4
    def swap(self):
        return Positions.Long if self == Positions.Short else Positions.Short

    def opposite(self):
        # Reverses the buy and sell status

        chance = random.random()

        if self == Positions.Hold:
            if chance > 0.50:
                chance = random.random()
                return Positions.Long if random.random() > 0.50 else Positions.Short
            else:
                return Positions.Hold
        elif self == Positions.Short:
            return Positions.ShortSell
        elif self == Positions.Long:
            return Positions.LongSell
        elif self == Positions.LongSell:
            return Positions.Long
        else:
            return Positions.Short

        # return Positions.Short if self == Positions.ShortSell else Positions.ShortSell


"""
Conditions
    Case Long Buy:
    Action(Buy)
    Case Long Sell:
    Action(Sell)
    Case Short Buy:
    Action(Buy)
    Case Short Sell:
    Action(Sell)
    default:
    Action(Hold)
"""






class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = 0
        self._last_trade_tick = 0
        self._position = None
        self._position_history = None
        self.position_history = None
        self._total_reward = 0
        self._total_profit = 1
        self._first_rendering = None
        self.history = None
        self.lastBuyLong = None
        self.lastBuyShort = None
        self.onBuy = False
        self.onBuyShort = False
        self.sell_interval = 0
        self.short_sell_interval = 0
        self.rewards_given = 0
        self.reward_multiplier = 1
        self.penalty_multiplier = 1
        self.total_profit = None
        self.current_action = Positions.Hold
        self.observation = None
        self.long_hold_reward_accumulator = 0
        self.short_hold_reward_accumulator = 0
        self.long_order_completed = False
        self.short_order_completed = False
        self.long_acc = 0.0
        self.short_acc = 0.0



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Hold
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.0
        self.rewards_given = 0
        self.reward_multiplier = 1
        self.penalty_multiplier = 1
        self._total_profit = 1.0  # unit
        self.long_hold_reward_accumulator = 0
        self.short_hold_reward_accumulator = 0
        self._first_rendering = True
        self.history = {}
        self.current_action = Positions.Hold
        self.long_order_completed = False
        self.short_order_completed = False
        self.long_acc = 0
        self.short_acc = 0


        return self._get_observation()

    def set_for_next_env(self, env):

        self.seed()
        self.df = env.df
        self.window_size = env.window_size
        self.prices, self.signal_features = env.prices, env.signal_features
        self.shape = (self.window_size, self.signal_features.shape[1])

        self._done = False
        # self._current_tick = self._start_tick
        self._current_tick -= 1
        # self._last_trade_tick = self._current_tick - 1
        self._last_trade_tick -= 1

        if isinstance(self.lastBuyShort, int):
            self.lastBuyShort -= 1
        if isinstance(self.lastBuyLong, int):
            self.lastBuyLong -= 1

        # self._position = Positions.Hold
        # self._position_history = (self.window_size * [None]) + [self._position]
        his = self.position_history
        del his[5]
        self._position_history = his
        self._position = env.last_position

        # self._total_reward = 0.0
        # self.rewards_given = 0
        # self.reward_multiplier = 1
        # self.penalty_multiplier = 1
        # self._total_profit = 1.0  # unit
        self._first_rendering = True
        # self.history = {}
        # self.current_action = Positions.Hold
        new_observation = env.signal_features[(self._current_tick - self.window_size):self._current_tick]

        return new_observation


    def step(self, action):
        print("Initially Action: ", action, " Position: ", self._position, " onBuy: ", self.onBuy, " onShortBuy: ", self.onBuyShort)
        print("Tick: ", self._current_tick, "Long Sell Interval: ", self.sell_interval, "Short Sell Interval: ", self.short_sell_interval)
        prices = self.prices
        long_profit = self.onBuy and self.is_long_profit(prices[self._current_tick], prices[self.lastBuyLong])
        long_stop = self.onBuy and self.is_long_stop(prices[self._current_tick], prices[self.lastBuyLong])
        short_profit = self.onBuyShort and self.is_short_profit(prices[self._current_tick], prices[self.lastBuyShort])
        short_stop = self.onBuyShort and self.is_short_stop(prices[self._current_tick], prices[self.lastBuyShort])
        print("is_Long_Profit: ", long_profit,"is_Long_Stop: ", long_stop,"is_Short_Profit: ", short_profit,"is_Short_Stop: ", short_stop,)

        long_sell = False
        short_sell = False
        wait_interval_min = 3
        wait_interval_max = 15
        # short_buy = False
        # long_buy = False

        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
            self.position_history = self._position_history
            print("Position History: ",self.position_history)
        if self._position == Positions.Hold and action != Actions.Hold.value:
            if action != Actions.Sell.value and action != Actions.ShortSell.value:
                self._position = self._position.opposite()
            # elif action == Actions.Buy and self.onBuyShort and not self.onBuy:
            #     self._position = Positions.Long
            # elif action == Actions.ShortBuy and not self.onBuyShort and self.onBuy:
            #     self._position = Positions.Short

        if self.short_sell_interval > 0:
            self.short_sell_interval -= 1
        if self.sell_interval > 0:
            self.sell_interval -= 1
        # Set Holds setting here
        # Initial Hold-> Buy/ ShortBuy ->
        """
        Conditions
            Case Long Buy:
            if(not self.onBuy) Action(Buy)
            self.onBuy = true
            Case Long Sell:
            if(self.onBuy) Action(Sell)
            self.onBuy = false

            Case Short Buy:
            self.onBuyShort = true            
            Case Short Sell:
            if(self.onBuyShort) Action(Sell)
            self.onBuyShort = false
            default:
            Action(Hold)
        """
        trade = (action == Actions.Buy.value and self._position == Positions.Long) or \
                (action == Actions.Sell.value and self._position == Positions.LongSell) or \
                (action == Actions.ShortBuy.value and self._position == Positions.Short) or \
                (action == Actions.ShortSell.value and self._position == Positions.ShortSell) or \
                (self.short_sell_interval < 1) or \
                (self.sell_interval < 1) or \
                (long_profit or long_stop) or \
                (short_profit or short_stop)

        print("Action: ", action, "Position: ", self._position, " Trade: ", trade)

        if trade:

            if long_profit or long_stop:
                print("Selling Regular Action Occurred")
                print("Sold at ", self.prices[self._current_tick])
                self._position = Positions.LongSell
                self._last_trade_tick = self._current_tick
                long_sell = True
                long_buy = False
                self.onBuy = False
                action = Actions.Sell.value
                self.long_order_completed = True
            elif short_profit or short_stop:
                print("Short Regular Selling Action Occurred")
                print("Sold at ", self.prices[self._current_tick])
                self._position = Positions.ShortSell
                short_sell = True
                short_buy = False
                self.onBuyShort = False
                self._last_trade_tick = self._current_tick
                action = Actions.ShortSell.value
                self.short_order_completed = True
            elif action == Actions.Buy.value:
                print("Long buy action check triggered")
                if not self.onBuy:
                    print("Buying Action Occurred")
                    print("Bought at ", self.prices[self._current_tick])
                    self._position = Positions.Long
                    long_buy = True
                    self._last_trade_tick = self._current_tick
                    self.lastBuyLong = self._current_tick
                    self.onBuy = True
                    self.sell_interval = random.randint(wait_interval_min, wait_interval_max)

                elif self.onBuy and self._position == Positions.Long:
                    print("Middle Hold")
                    self._position = Positions.Hold

                elif self.onBuyShort and self._position == Positions.Long:
                    if not self.onBuy:
                        print("Buying Action Occurred 2")
                        print("Bought at ", self.prices[self._current_tick])
                        self._position = Positions.Long
                        long_buy = True
                        self._last_trade_tick = self._current_tick
                        self.lastBuyLong = self._current_tick
                        self.onBuy = True
                        self.sell_interval = random.randint(wait_interval_min, wait_interval_max)
                    else:
                        print("Inner Hold")
                        self._position = Positions.Hold

                else:
                    print("Middle Hold")
                    self._position = Positions.Hold
            elif action == Actions.Sell.value:
                print("Long sell action check triggered")
                if self.onBuy and self.sell_interval < 1:

                    print("Selling Regular Action Occurred")
                    print("Sold at ", self.prices[self._current_tick])
                    self._position = Positions.LongSell
                    self._last_trade_tick = self._current_tick
                    long_sell = True
                    long_buy = False
                    self.onBuy = False
                else:
                    print("Middle Hold")
                    self._position = Positions.Hold
            elif action == Actions.ShortBuy.value:
                print("Short buy action check triggered")
                if not self.onBuyShort:
                    print("Short Buying Action Occurred")
                    print("Bought at ", self.prices[self._current_tick])
                    self._position = Positions.Short
                    short_buy = True
                    self._last_trade_tick = self._current_tick
                    self.lastBuyShort = self._current_tick
                    self.onBuyShort = True
                    self.short_sell_interval = random.randint(wait_interval_min,wait_interval_max)
                elif self.onBuy and self._position == Positions.Long:
                    if not self.onBuyShort:
                        print("Short Buying Action Occurred 2")
                        print("Bought at ", self.prices[self._current_tick])
                        self._position = Positions.Short
                        short_buy = True
                        self._last_trade_tick = self._current_tick
                        self.lastBuyShort = self._current_tick
                        self.onBuyShort = True
                        self.short_sell_interval = random.randint(wait_interval_min,wait_interval_max)
                    else:
                        print("Inner Hold")
                        self._position = Positions.Hold
                # elif self.onBuy and self._position == Positions.Short:
                #     self._position = Positions.Hold
                else:
                    print("Middle Hold")
                    self._position = Positions.Hold
            elif action == Actions.ShortSell.value:
                print("Short sell action check triggered")
                if self.onBuyShort and (self.short_sell_interval < 1):
                    print("Short Regular Selling Action Occurred")
                    self._position = Positions.ShortSell
                    short_sell = True
                    short_buy = False
                    self.onBuyShort = False
                    self._last_trade_tick = self._current_tick
                else:
                    print("Middle Hold")
                    self._position = Positions.Hold
            else:
                print("Was set onHold 2nd Top")
                self._position = Positions.Hold
            #hold
        if not trade:
            print("No Trade Hold Top")
            self._position = Positions.Hold

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        # long_hold = False
        # short_hold = True
        # long = False

        self._update_profit(action)
        print("Position: ", self._position)
        self._position_history.append(self._position)

        if short_sell:
            self._position = Positions.Short
            if random.random() > 0.50:
                self._position = self._position.swap()
                print("Chance: ", random.random())
            short_sell = False
        if long_sell and short_sell:
            self._position = Positions.Long
            if random.random() > 0.50:
                self._position = self._position.swap()
                print("Chance: ", random.random())
            long_sell = False

        observation = self._get_observation()
        self.total_profit = self._total_profit
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
        )
        self._update_history(info)

        if not self.onBuy:
            self.sell_interval = 0
        if not self.onBuyShort:
            self.short_sell_interval = 0


        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            elif position == Positions.ShortSell:
                color = 'blue'
            elif position == Positions.LongSell:
                color = 'orange'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        red_patch = mpatches.Patch(color='red', label='Long')
        green_patch = mpatches.Patch(color='green', label='LongSell')
        blue_patch = mpatches.Patch(color='blue', label='Short')
        orange_patch = mpatches.Patch(color='orange', label='ShortSell')
        plt.legend(handles=[red_patch,green_patch,blue_patch,orange_patch])

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit +
            " Short - Blue" +
            " ShortSell - Orange" +
            " Long - Red" +
            " LongSell - Green"

        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        print(len(window_ticks))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        short_sell_ticks = []
        long_sell_ticks = []
        print(self._position_history)

        for x, y in enumerate(window_ticks):
            y = self.prices[x]
            if self._position_history[x] == Positions.Short:
                plt.plot(x, y, color='blue', linewidth=3,
                        marker='o', markerfacecolor='blue', markersize=12)
            elif self._position_history[x] == Positions.Long:
                plt.plot(x, y, color='red', linewidth=3,
                        marker='o', markerfacecolor='red', markersize=12)
            elif self._position_history[x] == Positions.ShortSell:
                plt.plot(x, y, color='orange', linewidth=3,
                        marker='o', markerfacecolor='orange', markersize=12)
            elif self._position_history[x] == Positions.LongSell:
                plt.plot(x, y, color='green', linewidth=3,
                        marker='o', markerfacecolor='green', markersize=12)
            elif self._position_history[x] == Positions.Hold:
                plt.plot(x, y, color='black', linewidth=3,
                        marker='o', markerfacecolor='black', markersize=12)
            else:
                plt.plot(x, y, color='red', linewidth=3)
        # for i, tick in enumerate(window_ticks):
        #     if self._position_history[i] == Positions.Short:
        #         short_ticks.append(tick)
        #
        #         long_ticks.append(-1)
        #         short_sell_ticks.append(-1)
        #         long_sell_ticks.append(-1)
        #
        #     elif self._position_history[i] == Positions.Long:
        #         long_ticks.append(tick)
        #         short_ticks.append(-1)
        #         short_sell_ticks.append(-1)
        #         long_sell_ticks.append(-1)
        #     elif self._position_history[i] == Positions.ShortSell:
        #         short_sell_ticks.append(tick)
        #         long_ticks.append(-1)
        #         short_ticks.append(-1)
        #         long_sell_ticks.append(-1)
        #
        #     elif self._position_history[i] == Positions.LongSell:
        #         long_sell_ticks.append(tick)
        #         long_ticks.append(-1)
        #         short_sell_ticks.append(-1)
        #         short_ticks.append(-1)
        #
        # print("Size: ", len(short_ticks), "Short_Ticks: ", short_ticks)
        # print("Size: ", len(short_sell_ticks),"ShortSell_Ticks: ", short_sell_ticks)
        # print("Size: ", len(long_ticks),"Long_Ticks: ", long_ticks)
        # print("Size: ", len(long_sell_ticks),"LongSell_Ticks: ", long_sell_ticks)
        #
        # # plotting the points
        # # plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
        # #          marker='o', markerfacecolor='blue', markersize=12)
        #
        # plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        # plt.plot(long_ticks, self.prices[long_ticks], 'go')
        # plt.plot(short_sell_ticks, self.prices[short_ticks], 'bo')
        # plt.plot(long_sell_ticks, self.prices[long_ticks], 'yo')
        red_patch = mpatches.Patch(color='red', label='Long')
        green_patch = mpatches.Patch(color='green', label='LongSell')
        blue_patch = mpatches.Patch(color='blue', label='Short')
        orange_patch = mpatches.Patch(color='orange', label='ShortSell')
        black_patch = mpatches.Patch(color='black', label='Hold')

        plt.legend(handles=[red_patch,green_patch,blue_patch,orange_patch,black_patch])

        self.total_profit = self._total_profit
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError

    @property
    def last_position(self):
        return self._position

    @property
    def total_reward(self):
        return self._total_reward

