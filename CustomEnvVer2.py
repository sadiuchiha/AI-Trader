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
        self.trend_stat = None



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
        print("is_Long_Profit: ", long_profit,"is_Long_Stop: ", long_stop,"is_Short_Profit: ", short_profit,"is_Short_Stop: ", short_stop)


        long_sell = False
        short_sell = False
        wait_interval_min = 3
        wait_interval_max = 5
        # short_buy = False
        # long_buy = False

        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
            self.position_history = self._position_history
            print("Position History: ", self.position_history)
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

        #Pass prices
        self.trend_stat = self.checkTrendType(5, self._current_tick)
        isUptrend = self.trend_stat == "UpTrend"
        isDowntrend = self.trend_stat == "Downtrend"
        isNoTrend = self.trend_stat == "No Trend Up" or "No Trend Down" or "No Trend lvl"

        trade = (action == Actions.Buy.value and self._position == Positions.Long) or \
                (action == Actions.Sell.value and self._position == Positions.LongSell) or \
                (action == Actions.ShortBuy.value and self._position == Positions.Short) or \
                (action == Actions.ShortSell.value and self._position == Positions.ShortSell) or \
                (self.short_sell_interval < 1) or \
                (self.sell_interval < 1) or \
                (long_profit or long_stop) or \
                (short_profit or short_stop) or isUptrend or isDowntrend    #UpTrend and DownTrend not used.

        print("Action: ", action, "Position: ", self._position, " Trade: ", trade)

        if trade:

            # if self.onBuy and isUptrend and prices[self.lastBuyLong] < prices[self._current_tick]:
            #     print("Selling Action on UpTrend Occurred")
            #     print("Sold at ", self.prices[self._current_tick])
            #     self._position = Positions.LongSell
            #     self._last_trade_tick = self._current_tick
            #     long_sell = True
            #     long_buy = False
            #     self.onBuy = False
            #     action = Actions.Sell.value
            # elif self.onBuyShort and isDowntrend and prices[self.lastBuyLong] > prices[self._current_tick]:
            #     print("Short Selling Action on DownTrendOccurred")
            #     print("Sold at ", self.prices[self._current_tick])
            #     self._position = Positions.ShortSell
            #     short_sell = True
            #     short_buy = False
            #     self.onBuyShort = False
            #     self._last_trade_tick = self._current_tick
            #     action = Actions.ShortSell.value

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

            #Check if Downtrend Result
            if not isUptrend:
                pass
                #Set Chance as per ratio
            chance = random.random()
            if random.random() > 0.50:
                self._position = self._position.swap()
                print("Chance: ", random.random())
            short_sell = False
        if long_sell and not short_sell:
            self._position = Positions.Long
            if isUptrend:
                pass
            chance = random.random()
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

        print("Current Trend: ", self.trend_stat)
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

        red_patch = mpatches.Patch(color='red', label='Long')
        green_patch = mpatches.Patch(color='green', label='LongSell')
        blue_patch = mpatches.Patch(color='blue', label='Short')
        orange_patch = mpatches.Patch(color='orange', label='ShortSell')
        black_patch = mpatches.Patch(color='black', label='Hold')

        plt.legend(handles=[red_patch,green_patch,blue_patch,orange_patch,black_patch])
        self.showUniquePatterns(5)
        self.total_profit = self._total_profit
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    # Take prices and put it on a loop and a loop of window to save ups and downs
    # Count ups and downs in a fixed window  More ups is uptrend else if downtrend else no trend
    # Return whether Not Ready, Uptrend, Downtrend or No Trend

    def checkTrendType(self, window, index):
        prices = self.prices
        curr = index
        print("curr: ", curr)
        ups = 0
        downs = 0
        lvl = 0
        last_price = None
        initial_window_price = prices[curr - window]

        for i in range(window):
            if curr < 0:
                return "Not Ready"
            else:
                last_index = curr - window + i - 1
                curr_price = prices[curr - window + i]
                # print("Last_Price: ", last_price)
                print("curr_Price: ", curr_price)
                if last_price is None:
                    last_price = prices[last_index]
                    print("Last_Price: ", last_price)
                if last_price > curr_price:
                    downs += 1
                elif last_price < curr_price:
                    ups += 1
                else:
                    lvl += 1
                print("dif: ", last_price - curr_price)
                last_price = curr_price

        print("Ups: ", ups, "Downs: ", downs, "Lvl: ", lvl)
        price_difference = (curr_price - last_price) / 2
        if ups >= window * 0.8:
            if initial_window_price < curr_price:
                print("UpTrend set Followed")
                return "UpTrend"
            else:
                print("UpTrend set Followed but Dropped")
                return "UpTrend Dropped"

        elif downs >= window * 0.8:
            if initial_window_price > curr_price:
                print("DownTrend set Followed")
                return "DownTrend"
            else:
                print("DownTrend set Followed but Rose")
                return "DownTrend Rose"
        else:

            if lvl == 0:
                if ups >= window * 0.6 and downs <= window * 0.4:
                    if initial_window_price < curr_price:
                        print("No Trend Up Followed set")
                        return "No Trend Up"
                    else:
                        print("No Trend Up set Dropped")
                        return "No Trend Up Dropped"
                elif ups <= window * 0.4 and downs >= window * 0.6:
                    if initial_window_price > curr_price:
                        print("No Trend Down Followed set")
                        return "No Trend Down"
                    else:
                        print("No Trend Down set rose")
                        return "No Trend Down Rose"
            if lvl < window * 0.8:
                if ups > downs:
                    if initial_window_price < curr_price:
                        print("No Trend Up set")
                        return "No Trend Up"
                    else:
                        print("No Trend Up set Followed Dropped")
                        return "No Trend Up Dropped"
                elif ups < downs:
                    if initial_window_price > curr_price:
                        print("No Trend Down set")
                        return "No Trend Down"
                    else:
                        print("No Trend Down set Followed rose")
                        return "No Trend Down Rose"
                else:
                    if initial_window_price > curr_price:
                        print("No Trend lvl set Dropped")
                        return "No Trend lvl Dropped"
                    elif initial_window_price > curr_price:
                        print("No Trend lvl set rose")
                        return "No Trend lvl Rose"
                    else:
                        print("No Trend lvl set")
                        return "No Trend lvl"

            if lvl <= window:
                print("No Trend lvl set")
                return "No Trend lvl"


        # Figure pattern

        # for i in range(len(prices)):
        #     for j in range(window):

    #
    # Another function saves it in an array with the number of ud dataset

    def showPricesTrends(self, window):
        prices = self.prices
        curr = 0
        last_price = None
        price_trends = []
        window_initial_price = 0.0

        for i in range(len(prices)):
            curr = i
            slice = []
            ups = 0
            downs = 0
            lvl = 0
            status = None
            for j in range(window):
                if j == 0:
                    window_initial_price = prices[i - window + j]
                last_index = i - window + j - 1
                if curr < 0:
                    slice.append(None)
                curr_price = prices[i - window + j]
                if last_price is None:
                    last_price = prices[last_index]

                print("Last_Price: ", last_price)
                print("curr_Price: ", curr_price)
                print("dif: ", last_price - curr_price)
                if last_price > curr_price:
                    downs += 1
                    slice.append("D")
                elif last_price < curr_price:
                    ups += 1
                    slice.append("U")
                else:
                    lvl += 1
                    slice.append("L")

                last_price = curr_price
            price_difference = (curr_price - window_initial_price) * 0.5
            print("List Ups: ", ups, "Downs: ", downs, "Lvl: ", lvl)
            print("Initial_Price: ", window_initial_price, "Current Price: ", curr_price)
            if ups >= window * 0.8:
                if window_initial_price < curr_price:
                    print("UpTrend Followed")
                    status = "UpTrend"
                else:
                    print("UpTrend Followed but Dropped")
                    status = "UpTrend Dropped"
            elif downs >= window * 0.8:
                if window_initial_price > curr_price:
                    print("DownTrend Followed")
                    status = "DownTrend"
                else:
                    print("DownTrend Followed but rose")
                    status = "DownTrend Rose"
            else:
                if lvl == 0:
                    if ups >= window * 0.6 and downs <= window * 0.4:
                        if window_initial_price + price_difference < curr_price:
                            print("No Trend Up Followed")
                            status = "No Trend Up"
                        else:
                            print("No Trend Up Followed but Dropped")
                            status = "No Trend Up but Dropped"
                    elif ups <= window * 0.4 and downs >= window * 0.6:
                        if window_initial_price > curr_price:
                            print("No Trend Down Followed")
                            status = "No Trend Down"
                        else:
                            print("No Trend Down Followed but rose")
                            status = "No Trend Down rose"
                elif lvl < window * 0.8:
                    if ups > downs:
                        if window_initial_price < curr_price:
                            print("No Trend Up Followed")
                            status = "No Trend Up"
                        else:
                            print("No Trend Up Followed but Dropped")
                            status = "No Trend Up Dropped"
                    elif ups < downs:
                        if window_initial_price > curr_price:
                            print("No Trend Down Followed")
                            status = "No Trend Down"
                        else:
                            print("No Trend Down Followed but rose")
                            status = "No Trend Down Rose"
                    else:
                        if window_initial_price > curr_price:
                            print("No Trend lvl Followed but rose")
                            status = "No Trend lvl Rose"
                        elif window_initial_price < curr_price:
                            print("No Trend lvl Followed but dropped")
                            status = "No Trend lvl Dropped"
                        else:
                            print("No Trend Lvl Followed")
                            status = "No Trend lvl"
                elif lvl <= window:
                    print("No Trend Lvl Followed")
                    status = "No Trend lvl"
                print("Status Set")
            trend = [slice, status]
            price_trends.append(trend)
        return price_trends

    # Note all unique patterns in a window or range
    # List all uniques possibilities of trend pattern (box of ups and downs)
    # Match with trend prices from last function
    # print the patterns details

    def showUniquePatterns(self, window):
        patterns = self.showPricesTrends(window)
        uni_pattern = []

        for i in range(len(patterns)):
            pattern = patterns[i][0]
            status = patterns[i][1]
            if len(uni_pattern) == 0:
                uni_pattern.append([pattern, 1, status])
            matched = False
            for j in range(len(uni_pattern)):
                if pattern == uni_pattern[j][0]:
                    matched = True
                    uni_pattern[j][1] += 1

            if not matched:
                uni_pattern.append([pattern, 1, status])

        for i in range(len(uni_pattern)):
            print("Pattern: ", uni_pattern[i][0], " Status: ", uni_pattern[i][2], " Count: ", uni_pattern[i][1])

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

