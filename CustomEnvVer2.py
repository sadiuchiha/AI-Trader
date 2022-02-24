import random as random
from operator import itemgetter
import numpy as np
from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from function import FunctionMaker


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
        self.num_trades = 0
        self.trend_window = 5
        self.patternShortSellCondition = None
        self.patternLongSellCondition = None
        self.short_sell_alert_counter = -1
        self.nextShortSellPattern = None
        self.long_sell_alert_counter = -1
        self.nextSellPattern = None
        self.short_alert_counter = -1
        self.long_alert_counter = -1
        self.onShortSellAlert = None
        self.onSellAlert = None
        self.nextShortBuyPattern = None
        self.nextBuyPattern = None
        self.nextPattern = None
        self.patternLongBuyCondition = False
        self.patternShortBuyCondition = False
        self.created_pdf = False
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
        self.trend_status = None
        function = FunctionMaker()
        unique_frames = self.makeUniqueTrendFrame(self.trend_window)
        function.new(unique_frames[0], unique_frames[1], self.makeTrendFrame(self.trend_window))
        self.conditions = function.conditions


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.num_trades = 0
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
        self.trend_stat = None


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
        curr_price = self.prices[self._current_tick]
        long_price = prices[self.lastBuyLong]
        short_price = prices[self.lastBuyShort]
        long_profit = self.onBuy and self.is_long_profit(curr_price, long_price)
        long_stop = self.onBuy and self.is_long_stop(curr_price, long_price)
        short_profit = self.onBuyShort and self.is_short_profit(curr_price, short_price)
        short_stop = self.onBuyShort and self.is_short_stop(curr_price, short_price)
        print("is_Long_Profit: ", long_profit,"is_Long_Stop: ", long_stop,"is_Short_Profit: ", short_profit,"is_Short_Stop: ", short_stop)

        long_sell = False
        short_sell = False
        wait_interval_min = 3
        wait_interval_max = 5
        # short_buy = False
        # long_buy = False

        self._done = False
        self._current_tick += 1
        if self.long_alert_counter > -1:
            if self.onBuy and self.long_alert_counter >= 1:
                self.long_alert_counter -= 1
            elif self.long_alert_counter == 0:
                if not self.onBuy:
                    self.long_alert_counter -= 1

        if self.short_alert_counter > -1:
            if self.onBuyShort and self.short_alert_counter >= 1:
                self.short_alert_counter -= 1
            elif self.short_alert_counter == 0:
                if not self.onBuyShort:
                    self.short_alert_counter -= 1
        if self.long_sell_alert_counter > -1:
            if self.onBuy and self.long_sell_alert_counter >= 1:
                self.long_sell_alert_counter -= 1
            elif self.long_sell_alert_counter == 0:
                if not self.onBuy:
                    self.long_sell_alert_counter -= 1
        if self.short_sell_alert_counter > -1:
            if self.onBuyShort and self.short_sell_alert_counter >= 1:
                self.short_sell_alert_counter -= 1
            elif self.short_sell_alert_counter == 0:
                if not self.onBuyShort:
                    self.short_sell_alert_counter -= 1

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

        # if self.short_sell_interval > 0:
        #     self.short_sell_interval -= 1
        # if self.sell_interval > 0:
        #     self.sell_interval -= 1
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
        self.trend_stat = self.checkTrendType(self.trend_window, self._current_tick)
        isUptrend = self.trend_stat == "UpTrend"
        isDowntrend = self.trend_stat == "Downtrend"
        isNoTrendUp = self.trend_stat == "No Trend Up"
        isNoTrendDown = self.trend_stat == "No Trend Down"
        isNoTrend = self.trend_stat == "No Trend Up" or "No Trend Down" or "No Trend lvl"

        # Check buy conditions here
        current_pattern = self.currentTrendFrame(self.trend_window, self._current_tick)
        pattern_condition = self.checkPatternMatch(self.trend_window,current_pattern)
        custom_action = pattern_condition[1]
        next_pattern = pattern_condition[2]
        next_window = pattern_condition[3]
        print("custom_action: ", custom_action)
        print("next_window: ", next_window)
        print("self.long_alert_counter: ", self.long_alert_counter)
        print("self.short_alert_counter: ", self.short_alert_counter)
        print("self.long_sell_alert_counter: ", self.long_sell_alert_counter)
        print("self.short_sell_alert_counter: ", self.short_sell_alert_counter)
        pattern_condition = pattern_condition[0]

        # if custom_action == "Hold":
        #     pattern_condition = False
        if custom_action == "Buy" and not self.onBuy:
            print("pattern_condition on Buy: ", pattern_condition)
            self.nextBuyPattern = next_pattern
            pattern_condition = True
        if custom_action == "ShortBuy" and not self.onBuyShort:
            print("pattern_condition on ShortBuy: ", pattern_condition)
            self.nextShortBuyPattern = next_pattern
            pattern_condition = True
        if custom_action == "Sell" and self.onBuy:
            print("pattern_condition on Sell: ", pattern_condition)
            self.nextSellPattern = next_pattern
            pattern_condition = True
        if custom_action == "ShortSell" and self.onBuyShort:
            print("pattern_condition Short Sell: ", pattern_condition)
            self.nextShortSellPattern = next_pattern
            pattern_condition = True
        if self.nextBuyPattern == current_pattern or self.long_alert_counter == 0:
            print("pattern_condition on buy for sell: ", pattern_condition)
            self.onSellAlert = True
            pattern_condition = True
        if self.nextShortBuyPattern == current_pattern or self.short_alert_counter == 0:
            print("pattern_condition on shortBuy for sell: ", pattern_condition)
            self.onShortSellAlert = True
            pattern_condition = True

        print("pattern_condition: ", pattern_condition)

        if self._current_tick % 50 == 0:
            self.trend_status = self.checkTrendType(50, self._current_tick)
            print("Overall Status: ", self.trend_status)

        print("Current Trend Stat: ", self.trend_stat)

        long_pattern_buy_completed = False
        long_pattern_sell_completed = False
        short_pattern_buy_completed = False
        short_pattern_sell_completed = False
        trade = (action == Actions.Buy.value and self._position == Positions.Long) or \
                (action == Actions.Sell.value and self._position == Positions.LongSell) or \
                (action == Actions.ShortBuy.value and self._position == Positions.Short) or \
                (action == Actions.ShortSell.value and self._position == Positions.ShortSell) or \
                (long_profit or long_stop) or \
                (short_profit or short_stop) \
                or pattern_condition
                # or isUptrend or isDowntrend or isNoTrendUp or isNoTrendDown


        print("Action: ", action, "Position: ", self._position, " Trade: ", trade)

        if trade:
            if long_profit or long_stop:
                self.doLongProfitOrStop(curr_price)
            elif short_profit or short_stop:
                self.doShortProfitOrStop(curr_price)
            elif pattern_condition:
                if self.onSellAlert:
                    print("Initiating long sell")
                    self.patternLongBuyCondition = True
                    long_sell = self.doLongSell(isUptrend, isNoTrendUp, curr_price, long_price, long_sell, pattern_condition, next_window)
                    if self._position == Positions.Hold:
                        action = Actions.Hold.value
                    else:
                        action = Actions.Sell.value
                elif self.long_sell_alert_counter == 0:
                    print("Initiating long sell")
                    self.patternLongSellCondition = True
                    long_sell = self.doLongSell(isUptrend, isNoTrendUp, curr_price, long_price, long_sell, pattern_condition, next_window)
                    if self._position == Positions.Hold:
                        action = Actions.Hold.value
                    else:
                        action = Actions.Sell.value
                elif self.onShortSellAlert:
                    print("Initiating short sell")
                    self.patternShortBuyCondition = True
                    short_sell = self.doShortSell(isDowntrend, isNoTrendDown, curr_price, short_price, short_sell, pattern_condition, next_window)
                    if self._position == Positions.Hold:
                        action = Actions.Hold.value
                    else:
                        action = Actions.ShortSell.value
                elif self.short_sell_alert_counter == 0:
                    print("Initiating short sell")
                    self.patternShortSellCondition = True
                    short_sell = self.doShortSell(isDowntrend, isNoTrendDown, curr_price, short_price, short_sell, pattern_condition, next_window)
                    if self._position == Positions.Hold:
                        action = Actions.Hold.value
                    else:
                        action = Actions.ShortSell.value

                elif custom_action == "Buy" and not self.onBuy:
                    print("Initiating Long Buy")
                    self.doLongBuy(pattern_condition, next_window)
                    if self._position == Positions.Hold:
                        action = Actions.Hold.value
                    else:
                        action = Actions.Buy.value
                        long_pattern_buy_completed = True

                elif custom_action == "ShortBuy" and not self.onBuyShort:
                    print("Initiating Short Buy")
                    self.doShortBuy(pattern_condition, next_window)
                    if self._position == Positions.Hold:
                        action = Actions.Hold.value
                    else:
                        action = Actions.ShortBuy.value
                        short_pattern_buy_completed = True

                # elif custom_action == "Sell" and self.onBuy:
                #     long_sell = self.doLongSell(isUptrend, isNoTrendUp, curr_price, long_price, pattern_condition)
                #     action = Actions.Sell.value
                # elif custom_action == "ShortSell" and self.onBuyShort:
                #     short_sell = self.doShortSell(isDowntrend, isNoTrendDown, curr_price, short_price, pattern_condition)
                #     action = Actions.ShortSell.value
                elif custom_action == "Hold":
                    self._position = Positions.Hold
                    action = Actions.Hold.value

            # sell will only occur on short_sell_alert_counter, sell alert_counter
            # elif action == Actions.Buy.value:
            #     self.doLongBuy(False, next_window)
            # elif action == Actions.Sell.value:
            #     long_sell = self.doLongSell(isUptrend, isNoTrendUp, curr_price, long_price,False, next_window)
            # elif action == Actions.ShortBuy.value:
            #     self.doShortBuy(False, next_window)
            # elif action == Actions.ShortSell.value:
            #     short_sell = self.doShortSell(isDowntrend, isNoTrendDown, curr_price, short_price,False, next_window)
            else:
                print("Was set onHold 2nd Top")
                self._position = Positions.Hold
                action = Actions.Hold.value
        print("After Trade True, Action: ", action, "Position: ", self._position, " Trade: ", trade)

        if not trade or self._position == Positions.Hold:
            print("No Trade Hold Top")
            self._position = Positions.Hold
            action = Actions.Hold.value
        print("After Trade False, Action: ", action, "Position: ", self._position, " Trade: ", trade)

        if self._position == Positions.Long:
            if self.onBuy and self.lastBuyLong == self._current_tick:
                action = Actions.Buy.value
            else:
                action = Actions.Hold.value
                self._position = Positions.Hold
        elif not self.patternLongBuyCondition and long_pattern_buy_completed:
            self._position == Positions.LongSell
            action = Actions.sell.value
        elif not self.patternLongSellCondition and long_pattern_sell_completed:
            self._position == Positions.LongSell
            action = Actions.sell.value
        elif self._position == Positions.Short:
            if self.onBuyShort and self.lastBuyShort == self._current_tick:
                action = Actions.ShortBuy.value
            else:
                action = Actions.Hold.value
                self._position = Positions.Hold
        elif not self.patternShortBuyCondition and short_pattern_buy_completed:
            self._position == Positions.ShortSell
            action = Actions.ShortSell.value
        elif not self.patternShortSellCondition and short_pattern_sell_completed:
            self._position == Positions.ShortSell
            action = Actions.ShortSell.value
        elif self._position == Positions.LongSell:
            action = Actions.Sell.value
        elif self._position == Positions.ShortSell:
            action = Actions.ShortSell.value

        print("Final Change, Action: ", action, "Position: ", self._position, " Trade: ", trade)

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        # self._update_profit(action)
        print("Position: ", self._position)
        self._position_history.append(self._position)

        # If short sell occured it will check trend Status and swap short to long accordingly
        # if short_sell:
        #     self._position = Positions.Short
        #     chance = random.random()
        #     if self.trend_status == "No Trend Down":
        #         chance = 0
        #     if chance < .5:
        #         self._position = self._position.swap()
        #         print("Chance: ", chance)
        #     short_sell = False

        # if long_sell and not short_sell:
        #     self._position = Positions.Long
        #     chance = random.random()
        #     if self.trend_status == "No Trend Up":
        #         chance = 0
        #     if chance < .5:
        #         self._position = self._position.swap()
        #         print("Chance: ", chance)
        #     long_sell = False

        observation = self._get_observation()
        self.total_profit = self._total_profit
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
        )
        self._update_history(info)

        if self._position == Positions.Long or self._position == Positions.Short:
            self._position = Positions.Hold
            print("Changed into Hold From either Long or Short Buy")

        # if not self.onBuy:
        #     self.sell_interval = 0
        # if not self.onBuyShort:
        #     self.short_sell_interval = 0

        print("Current Trend: ", self.trend_stat)
        if pattern_condition:
            print("Pattern Condition Matched")
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
                        marker='o', markerfacecolor='blue', markersize=8)
            elif self._position_history[x] == Positions.Long:
                plt.plot(x, y, color='red', linewidth=3,
                        marker='o', markerfacecolor='red', markersize=8)
            elif self._position_history[x] == Positions.ShortSell:
                plt.plot(x, y, color='orange', linewidth=3,
                        marker='o', markerfacecolor='orange', markersize=8)
            elif self._position_history[x] == Positions.LongSell:
                plt.plot(x, y, color='green', linewidth=3,
                        marker='o', markerfacecolor='green', markersize=8)
            elif self._position_history[x] == Positions.Hold:
                plt.plot(x, y, color='black', linewidth=3,
                        marker='h', markerfacecolor='black', markersize=4)
            else:
                plt.plot(x, y, color='red', linewidth=3)

        red_patch = mpatches.Patch(color='red', label='Long')
        green_patch = mpatches.Patch(color='green', label='LongSell')
        blue_patch = mpatches.Patch(color='blue', label='Short')
        orange_patch = mpatches.Patch(color='orange', label='ShortSell')
        black_patch = mpatches.Patch(color='black', label='Hold')

        plt.legend(handles=[red_patch,green_patch,blue_patch,orange_patch,black_patch])
        self.showOverallPatterns(50)
        self.showAllPatterns(5)
        self.showTrendFrame(5)
        # self.showUniquePatterns(5)
        print("Number of trades: ", self.num_trades)

        self.total_profit = self._total_profit
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        # self.makeUniqueTrendFrame(5)


    # It will Return Current Trend frame of price point
    # Last 5 trends in a list
    #
    def currentTrendFrame(self, window, current_tick):
        trends = self.makeTrendFrame(window)
        prices = self.prices
        trend_frame = []

        for i in range(len(trends)):
            if current_tick == i:
                return trends[i][0]
        return "None"
        #     slice = []
        #     initial_window_price = prices[i]
        #     initial_index = i
        #     for j in range(window):
        #         # Add Overall Trend Here
        #         index = i + j
        #         if index < len(trends):
        #             slice.append(trends[index][1])
        #         else:
        #             index -= 1
        #             break
        #     last_price = prices[index]
        #     # print("initial_window_price: ", initial_window_price, "last_price: ", last_price)
        #     dif = last_price - initial_window_price
        #     # print("Slice: ", slice)
        #     trend_frame.append([slice, dif, [initial_window_price, initial_index], [last_price, index]])
        # if len(trend_frame) == 0:
        #     return "None"
        # else:
        #     last_index = len(trend_frame)-1
        #     return trend_frame[last_index][0]

    #Collection of trends in a window
    # Structure example: { Uptrend, DownTrend, Uptrend, DownTrend, DownTrend }
    #
    # Secondary requirement:
    # Store overall trend of each trend
    def makeTrendFrame(self, window):
        trends = self.showPricesTrends(window, -1)
        prices = self.prices
        trend_frame = []
        # print(trends)
        # print(trends[0][1])
        # print(trends[0][1])
        # print(trends[1][1])
        # print(trends[2][1])

        for i in range(len(trends)):
            slice = []
            initial_window_price = prices[i]
            initial_index = i
            for j in range(window):
                # Add Overall Trend Here
                index = i+j
                if index < len(trends):
                    slice.append(trends[index][1])
                else:
                    index -= 1
                    break
            last_price = prices[index]
            # print("initial_window_price: ", initial_window_price, "last_price: ", last_price)
            dif = last_price - initial_window_price
            # print("Slice: ", slice)
            trend_frame.append([slice, dif, [initial_window_price, initial_index], [last_price, index]])

        return trend_frame

    # Show Trend Frame of a certain window
    def showTrendFrame(self, window):
        trend_frame = self.makeTrendFrame(window)
        for i in range(len(trend_frame)):
            print("TF ", i, ": ", trend_frame[i][0], "dif: ", trend_frame[i][1], "Ini_Price (", trend_frame[i][2][1], "): ", trend_frame[i][2][0], "curr: (", trend_frame[i][3][1], "): ", trend_frame[i][3][0])


    #Shows Unique trends
    def makeUniqueTrendFrame(self, window):
        trend_frames = self.makeTrendFrame(window)
        uni_frame = []
        order_pos = []
        previous_frame = None
        for i in range(len(trend_frames)):
            initial_index = trend_frames[i][2][1]
            last_index = trend_frames[i][3][1]
            dif = trend_frames[i][1]
            frame = trend_frames[i][0]
            if i > window:
                previous_frame = trend_frames[i-1][0]
            else:
                previous_frame = "None"
            indexes = [[initial_index, last_index, dif, previous_frame]]

            if len(uni_frame) == 0:
                uni_frame.append([frame, 0, indexes])
            matched = False
            for j in range(len(uni_frame)):
                if frame == uni_frame[j][0]:
                    matched = True
                    uni_frame[j][1] += 1
                    uni_frame[j][2].append(indexes)
                    indexes = uni_frame[j][2]
                    uni_frame[j][2] = indexes
            if not matched:
                uni_frame.append([frame, 1, indexes])

        for i in range(len(uni_frame)):
            order_pos.append([i, uni_frame[i][1]])

        print("Sorting Started")
        print(order_pos)
        order_pos = sorted(order_pos, key=itemgetter(1), reverse=True)
        print(order_pos)
        for i in range(len(order_pos)):
            index = order_pos[i][0]
            # print("Status: ", uni_frame[index][0], " Count: ", uni_frame[index][1], " Indexes: ", uni_frame[index][2])
            print("Status: ", uni_frame[index][0], " Count: ", uni_frame[index][1])

        if self.created_pdf == True:
            ch = "n"
        else:
            ch = input("Create Pdfs of chart? y/n ")
            if ch == "n":
                self.created_pdf = True
        if ch == "y":
            self.showPatternChart(window, initial_index, uni_pattern=uni_frame, indexes=order_pos)

        return [uni_frame, order_pos]


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
                # print("curr_Price: ", curr_price)
                if last_price is None:
                    last_price = prices[last_index]
                    # print("Last_Price: ", last_price)
                if last_price > curr_price:
                    downs += 1
                elif last_price < curr_price:
                    ups += 1
                else:
                    lvl += 1
                # print("dif: ", last_price - curr_price)
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
    def showOverallPricesTrends(self, window):
        prices = self.prices
        curr = 0
        last_price = None
        price_trends = []
        window_initial_price = 0.0

        for i in range(len(prices)):
            if i % window == 0: i += window
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
                if last_index >= len(prices):
                    last_index = len(prices) - 1
                if curr < 0:
                    slice.append(None)
                # print()
                # print("Current Price index: ", last_index, "Size of Price: ", len(prices))
                curr_price = prices[last_index]

                if last_price is None:
                    last_price = prices[last_index]

                # print("Last_Price: ", last_price)
                # print("curr_Price: ", curr_price)
                # print("dif: ", last_price - curr_price)
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
            price_difference = (curr_price - window_initial_price)
            # print("List Ups: ", ups, "Downs: ", downs, "Lvl: ", lvl)
            # print("Initial_Price: ", window_initial_price, "Current Price: ", curr_price)
            if ups >= window * 0.8:
                if window_initial_price < curr_price:
                    # print("UpTrend Followed")
                    status = "UpTrend"
                else:
                    # print("UpTrend Followed but Dropped")
                    status = "UpTrend Dropped"
            elif downs >= window * 0.8:
                if window_initial_price > curr_price:
                    # print("DownTrend Followed")
                    status = "DownTrend"
                else:
                    # print("DownTrend Followed but rose")
                    status = "DownTrend Rose"
            else:
                if lvl == 0:
                    if ups >= window * 0.6 and downs <= window * 0.4:
                        if window_initial_price < curr_price:
                            # print("No Trend Up Followed")
                            status = "No Trend Up"
                        else:
                            # print("No Trend Up Followed but Dropped")
                            status = "No Trend Up but Dropped"
                    elif ups <= window * 0.4 and downs >= window * 0.6:
                        if window_initial_price > curr_price:
                            # print("No Trend Down Followed")
                            status = "No Trend Down"
                        else:
                            # print("No Trend Down Followed but rose")
                            status = "No Trend Down rose"
                elif lvl < window * 0.8:
                    if ups > downs:
                        if window_initial_price < curr_price:
                            # print("No Trend Up Followed")
                            status = "No Trend Up"
                        else:
                            # print("No Trend Up Followed but Dropped")
                            status = "No Trend Up Dropped"
                    elif ups < downs:
                        if window_initial_price > curr_price:
                            # print("No Trend Down Followed")
                            status = "No Trend Down"
                        else:
                            # print("No Trend Down Followed but rose")
                            status = "No Trend Down Rose"
                    else:
                        if window_initial_price > curr_price:
                            # print("No Trend lvl Followed but rose")
                            status = "No Trend lvl Rose"
                        elif window_initial_price < curr_price:
                            # print("No Trend lvl Followed but dropped")
                            status = "No Trend lvl Dropped"
                        else:
                            # print("No Trend Lvl Followed")
                            status = "No Trend lvl"
                elif lvl <= window:
                    # print("No Trend Lvl Followed")
                    status = "No Trend lvl"
                # print("Status Set")
            if i % window == 0:
                price_dif_per = price_difference / window_initial_price
                trend = [slice, status, price_dif_per]
                price_trends.append(trend)
        return price_trends

    def showPatternChart(self, window, current_tick, uni_pattern, indexes):

        plt.show()
        prices = self.prices
        for i in range(len(indexes)):

            pdfFile = PdfPages("outputs/patterns/Pattern Charts"+str(i)+".pdf")

            curr_index = indexes[i][0]

            for z in range(uni_pattern[curr_index][1]):

                print(uni_pattern[curr_index][2][z][0])

                if not isinstance(uni_pattern[curr_index][2][z][0], int):
                    current_tick = uni_pattern[curr_index][2][z][0][0]
                    prev_status = uni_pattern[curr_index][2][z][0][3]

                else:
                    current_tick = uni_pattern[curr_index][2][z][0]
                    prev_status = uni_pattern[curr_index][2][z][3]


                print("Current_Tick: ", current_tick)

                x = []
                y = []
                x1 = []
                y1 = []

                for j in range(window):

                    status = uni_pattern[curr_index][0]

                    past_window_curr = current_tick - window + j
                    curr = current_tick + j
                    print(curr)

                    print("Ploting")

                    x.append(curr)
                    y.append(prices[curr])
                    if past_window_curr > 0:
                        x1.append(past_window_curr)
                        y1.append(prices[past_window_curr])

                print(x)
                print(y)
                statuses = ""
                prev_statuses = ""
                for c in range(len(status)):
                    c_status = status[c]
                    if c_status is None:
                        c_status = "None"
                    statuses += str(c_status) + " "
                for c in range(len(prev_status)):
                    p_status = prev_status[c]
                    if p_status is None:
                        p_status = "None"
                    prev_statuses += str(p_status) + " "

                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(x, y, color="red")
                ax.set_title("Status: " + statuses + "\n Previous Status: " + prev_statuses)
                pdfFile.savefig(fig)
                # plt.show()

            pdfFile.close()

    def showPricesTrends(self, window, current_tick):
        prices = self.prices
        last_price = None
        price_trends = []
        window_initial_price = 0.0

        for i in range(len(prices)):
            slice = []
            ups = 0
            downs = 0
            lvl = 0
            status = None

            if i > current_tick != -1:
                break

            for j in range(window):
                curr = i - window + j
                if j == 0:
                    window_initial_price = prices[curr]
                last_index = i - window + j - 1
                if i < 0:
                    slice.append(None)
                curr_price = prices[curr]
                if last_price is None:
                    last_price = prices[last_index]

                # print("Last_Price: ", last_price)
                # print("curr_Price: ", curr_price)
                # print("dif: ", last_price - curr_price)
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
            price_difference = (curr_price - window_initial_price)
            # print("List Ups: ", ups, "Downs: ", downs, "Lvl: ", lvl)
            # print("Initial_Price: ", window_initial_price, "Current Price: ", curr_price)
            if ups >= window * 0.8:
                if window_initial_price < curr_price:
                    # print("UpTrend Followed")
                    status = "UpTrend"
                else:
                    # print("UpTrend Followed but Dropped")
                    status = "UpTrend Dropped"
            elif downs >= window * 0.8:
                if window_initial_price > curr_price:
                    # print("DownTrend Followed")
                    status = "DownTrend"
                else:
                    # print("DownTrend Followed but rose")
                    status = "DownTrend Rose"
            else:
                if lvl == 0:
                    if ups >= window * 0.6 and downs <= window * 0.4:
                        if window_initial_price < curr_price:
                            # print("No Trend Up Followed")
                            status = "No Trend Up"
                        else:
                            # print("No Trend Up Followed but Dropped")
                            status = "No Trend Up but Dropped"
                    elif ups <= window * 0.4 and downs >= window * 0.6:
                        if window_initial_price > curr_price:
                            # print("No Trend Down Followed")
                            status = "No Trend Down"
                        else:
                            # print("No Trend Down Followed but rose")
                            status = "No Trend Down rose"
                elif lvl < window * 0.8:
                    if ups > downs:
                        if window_initial_price < curr_price:
                            # print("No Trend Up Followed")
                            status = "No Trend Up"
                        else:
                            # print("No Trend Up Followed but Dropped")
                            status = "No Trend Up Dropped"
                    elif ups < downs:
                        if window_initial_price > curr_price:
                            # print("No Trend Down Followed")
                            status = "No Trend Down"
                        else:
                            # print("No Trend Down Followed but rose")
                            status = "No Trend Down Rose"
                    else:
                        if window_initial_price > curr_price:
                            # print("No Trend lvl Followed but rose")
                            status = "No Trend lvl Rose"
                        elif window_initial_price < curr_price:
                            # print("No Trend lvl Followed but dropped")
                            status = "No Trend lvl Dropped"
                        else:
                            # print("No Trend Lvl Followed")
                            status = "No Trend lvl"
                elif lvl <= window:
                    # print("No Trend Lvl Followed")
                    status = "No Trend lvl"
                # print("Status Set")
            trend = [slice, status]
            price_trends.append(trend)
        return price_trends

    # Note all unique patterns in a window or range
    # List all uniques possibilities of trend pattern (box of ups and downs)
    # Match with trend prices from last function
    # print the patterns details

    # Check list of conditions and apply the conditions as action and position set
    def checkPatternMatch(self, window, pattern):
        print("Checking Pattern")
        conditions = self.conditions
        single_cond = conditions[0]
        multi_cond = conditions[1]
        prev_match = False
        next_match_pattern, next_match = None, None
        action = None
        # Check single conditions and set probability to full for action
        for i in range(len(single_cond)):
            next_match_pattern = single_cond[i][2]
            # print("single_cond[i][0]: ", single_cond[i][0])
            # print("pattern: ", pattern)

            if single_cond[i][0] == pattern:
                print("Single condition matched")
                prev_match = True
                ratio = single_cond[i][5] / single_cond[i][4]
                action = single_cond[i][1]
                # random_choosen = random.random()
                # if ratio < random_choosen:
                #     action = single_cond[i][1]
                # else:
                #     action = "Hold"
                # print("Action Set single: ", action, " ratio: " + str(single_cond[i][5]) + "/" + str(single_cond[i][4]), "random_choosen: ", random_choosen)
                return [prev_match, action, next_match_pattern, window]

        if not prev_match:
            for i in range(len(multi_cond)):
                next_match_pattern = multi_cond[i][1]
                # print("multi_cond[i][0: ", multi_cond[i][0])
                # print("pattern: ", pattern)
                if multi_cond[i][0] == pattern:
                    print("Multi condition matched")
                    actions = multi_cond[i][1]
                    ratio = []
                    random_choosen = random.randint(0,len(actions)-1)
                    # sum = 0

                    prev_match = True
                    next_match = next_match_pattern[random_choosen]
                    action = actions[random_choosen][0]
                    return [prev_match, action, next_match, window]

                    # for j in range(len(actions)):
                    #     # Create probability per ratio
                    #     # if j > len(actions) - 1:
                    #     next_match = next_match_pattern[j]
                    #     action = actions[j][0]
                    #     print("actions[j][2]: ", actions[j][2])
                    #     print("actions[j][3]: ", actions[j][3])
                    #     print("random_choosen: ", random_choosen)
                    #     sum += actions[j][3] / actions[j][2]
                    #     # ratio.append(actions[j][2] / actions[j][3])
                    #     if sum > random_choosen:
                    #         print("Action Set multi: ", action,
                    #               " ratio: " + str(actions[j][3]) + "/" + str(actions[j][2]),
                    #               "random_choosen: ", random_choosen)
                    #     if sum == actions[j][3]:
                    #         break
                    #     # Check Multiple conditions and set probability as per ratio of the count of the prev pattern
        # Shuffle action
        #
        print("Action Set default: ", action)
        return [prev_match, action, next_match, window]

    # Check Trend of a pattern by matching initial price and final price
    def checkTrendInPattern(self):
        pass

    def showUniquePatterns(self, window):
        initial_index = 0
        patterns = self.showPricesTrends(window, -1)
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


    def showAllPatterns(self, window):
        patterns = self.showPricesTrends(window, -1)

        for i in range(len(patterns)):
            print("Pattern: ", patterns[i][0], " Status: ", patterns[i][1])

    def showOverallPatterns(self, window):
        patterns = self.showOverallPricesTrends(window)
        overall_trends = []

        for i in range(len(patterns)):
            past_trend = None
            if i != 0: past_trend = patterns[i-1][0]
            pattern = patterns[i][0]
            status = patterns[i][1]
            overall_trends.append([pattern, status, past_trend])

        for i in range(len(overall_trends)):
            print("Pattern", i, ":", overall_trends[i][0], " Status: ", overall_trends[i][1], "Past Pattern: ", overall_trends[i][2])
        print("Length of Pattern: ", len(overall_trends))

    def partition(self, arr, low, high):
        i = (low - 1)  # index of smaller element
        pivot = arr[high][1]  # pivot

        for j in range(low, high):

            # If current element is smaller than or
            # equal to pivot
            if arr[j][1] <= pivot:
                # increment index of smaller element
                i = i + 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    # The main function that implements QuickSort
    # arr[] --> Array to be sorted,
    # low  --> Starting index,
    # high  --> Ending index

    # Function to do Quick sort

    def quickSort(self, arr, low, high):
        if len(arr) == 1:
            return arr
        if low < high:
            # pi is partitioning index, arr[p] is now
            # at right place
            pi = self.partition(arr, low, high)

            # Separately sort elements before
            # partition and after partition
            self.quickSort(arr, low, pi - 1)
            self.quickSort(arr, pi + 1, high)


    def sortFrames(self, trend_frames):

        highest_order = []

        for i in range(len(trend_frames)):
            curr_trend_count = trend_frames[i][1]
            trend = trend_frames[i]
            if i == 0 and curr_trend_count > 0:
                highest_order.append(trend)
                len_order = len(highest_order)
                last_order = highest_order
            for j in range(len_order):
                if curr_trend_count > highest_order[j][1]:
                    highest_order = [trend] + last_order
                    break
                elif curr_trend_count <= highest_order[j][1]:
                    highest_order.insert(trend, j)
                    break

        return highest_order

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig("Plot.png")

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


    def doLongBuy(self, pattern_match,next_window):
        print("Doing Long buy 1")
        if pattern_match:
            print("Doing Long buy 2")
            if not self.onBuy:
                print("Doing Long buy 3")
                print("Buying Action on Pattern Occurred")
                print("Bought at ", self.prices[self._current_tick])
                self._position = Positions.Long
                long_buy = True
                self._last_trade_tick = self._current_tick
                self.lastBuyLong = self._current_tick
                self.onBuy = True
                self.patternLongBuyCondition = True
                self.long_alert_counter = next_window

            else:
                self._position = Positions.Hold
        # Apply Pattern Match code
            # self.patternLongBuyCondition is not True   Check
            # apply Long Buy action
            # self.patternLongBuyCondition
        else:
            self._position = Positions.Hold
            # print("Long buy action check triggered")
            # if not self.onBuy:
            #     print("Buying Action Occurred")
            #     print("Bought at ", self.prices[self._current_tick])
            #     self._position = Positions.Long
            #     long_buy = True
            #     self._last_trade_tick = self._current_tick
            #     self.lastBuyLong = self._current_tick
            #     self.onBuy = True
            #     # self.sell_interval = random.randint(wait_interval_min, wait_interval_max)
            #
            # elif self.onBuy and self._position == Positions.Long:
            #     print("Middle Hold")
            #     self._position = Positions.Hold
            #
            # elif self.onBuyShort and self._position == Positions.Long:
            #     if not self.onBuy:
            #         print("Buying Action Occurred 2")
            #         print("Bought at ", self.prices[self._current_tick])
            #         self._position = Positions.Long
            #         long_buy = True
            #         self._last_trade_tick = self._current_tick
            #         self.lastBuyLong = self._current_tick
            #         self.onBuy = True
            #         # self.sell_interval = random.randint(wait_interval_min, wait_interval_max)
            #     else:
            #         print("Inner Hold")
            #         self._position = Positions.Hold
            #
            # else:
            #     print("Middle Hold")
            #     self._position = Positions.Hold

    def doLongSell(self, isUptrend, isNoTrendUp, curr_price, long_price, long_sell, pattern_match,next_window):
        print("Doing Sell 1")
        if pattern_match:
            print("Doing Sell 2")
            if self.onBuy and self.patternLongBuyCondition or self.patternLongSellCondition:
                print("Doing Sell 3")
                if self.patternLongBuyCondition:
                    print("Doing Sell 4")
                    if self.long_alert_counter == 0:
                        print("Doing Sell 5")
                        print("Selling Pattern Action Occurred")
                        print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyLong])
                        self._position = Positions.LongSell
                        self._last_trade_tick = self._current_tick
                        long_sell = True
                        long_buy = False
                        self.onBuy = False
                        self.patternLongBuyCondition = False
                        self.onSellAlert = False
                        self.num_trades += 1
                        self._update_profit(Actions.Sell.value)

                    else:
                        print("Middle Hold")
                        self._position = Positions.Hold
                elif self.patternLongSellCondition:
                    if self.long_sell_alert_counter == 0:
                        print("Selling Pattern 2 Action Occurred")
                        print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyLong])
                        self._position = Positions.LongSell
                        self._last_trade_tick = self._current_tick
                        long_sell = True
                        long_buy = False
                        self.onBuy = False
                        self.patternLongSellCondition = False
                        self.onSellAlert = False
                        self.num_trades += 1
                        self._update_profit(Actions.Sell.value)

                    else:
                        print("Middle Hold")
                        self._position = Positions.Hold
            else:
                print("Middle Hold")
                self._position = Positions.Hold
        # Apply Pattern Match code
            # self.patternLongBuyCondition is True   Check
            # apply Long Sell action by pattern_match condition
            # self.patternLongBuyCondition
        else:
            print("Long sell action check triggered")
            if self.onBuy and isUptrend and curr_price > long_price:
                print("Selling isUptrend Action Occurred")
                print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyLong])
                self._position = Positions.LongSell
                self._last_trade_tick = self._current_tick
                long_sell = True
                long_buy = False
                self.onBuy = False
            elif self.onBuy and isNoTrendUp and curr_price > long_price:
                print("Selling isNoTrendUp Action Occurred")
                print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyLong])
                self._position = Positions.LongSell
                self._last_trade_tick = self._current_tick
                long_sell = True
                long_buy = False
                self.onBuy = False

            else:
                print("Middle Hold")
                self._position = Positions.Hold

        return long_sell
    def doShortSell(self,isDowntrend, isNoTrendDown, curr_price, short_price, short_sell, pattern_match, next_window):
        print("Doing Short Sell 1")

        if pattern_match:
            print("Doing Short Sell 2")

            if self.onBuyShort and self.patternShortBuyCondition or self.patternShortSellCondition:
                print("Doing Short Sell 3")

                if self.patternShortBuyCondition:
                    print("Doing Short Sell 4")

                    if self.short_alert_counter == 0:
                        print("Doing Short Sell 5")
                        print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyShort])
                        print("Short Pattern Selling Action Occurred")
                        self._position = Positions.ShortSell
                        short_sell = True
                        short_buy = False
                        self.onBuyShort = False
                        self._last_trade_tick = self._current_tick
                        self.patternShortBuyCondition = False
                        self.onShortSellAlert = False
                        self.num_trades += 1
                        self._update_profit(Actions.ShortSell.value)

                    else:
                        print("Middle Hold")
                        self._position = Positions.Hold
                elif self.patternShortSellCondition:
                    print("Doing Short Sell 4")

                    if self.short_sell_alert_counter == 0:
                        print("Doing Short Sell 5")
                        print("Short Pattern Selling 2 Action Occurred")
                        print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyShort])

                        self._position = Positions.ShortSell
                        short_sell = True
                        short_buy = False
                        self.onBuyShort = False
                        self._last_trade_tick = self._current_tick
                        self.patternShortSellCondition = False
                        self.onShortSellAlert = False
                        self.num_trades += 1
                        self._update_profit(Actions.ShortSell.value)

                    else:
                        print("Middle Hold")
                        self._position = Positions.Hold
            else:
                print("Middle Hold")
                self._position = Positions.Hold
            # Apply Pattern Match code
            # self.patternShortBuyCondition is True   Check
            # apply Long Sell action by pattern_match condition
            # self.patternLongBuyCondition
        else:
            print("Short sell action check triggered")
            if self.onBuyShort and isDowntrend and short_price > curr_price:
                print("Short isDowntrend Selling Action Occurred")
                print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyShort])

                self._position = Positions.ShortSell
                short_sell = True
                short_buy = False
                self.onBuyShort = False
                self._last_trade_tick = self._current_tick
            elif self.onBuyShort and isNoTrendDown and short_price > curr_price:
                print("Short isNoTrendDown Selling Action Occurred")
                print("Sold at ", curr_price, "Buy price: ", self.prices[self.lastBuyShort])

                self._position = Positions.ShortSell
                short_sell = True
                short_buy = False
                self.onBuyShort = False
                self._last_trade_tick = self._current_tick
            else:
                print("Middle Hold")
                self._position = Positions.Hold

        return short_sell
    def doShortBuy(self,pattern_match,next_window):
        print("Doing Short buy 1")

        if pattern_match:
            print("Doing Short buy 1")

            if not self.onBuyShort:
                print("Doing Short buy 1")

                print("Short Buying Action Occurred")
                print("Bought at ", self.prices[self._current_tick])
                self._position = Positions.Short
                short_buy = True
                self._last_trade_tick = self._current_tick
                self.lastBuyShort = self._current_tick
                self.onBuyShort = True
                self.short_alert_counter = next_window
            # Apply Pattern Match code
            # self.patternShortBuyCondition is not True   Check
            # apply Short Buy action
            # self.patternShortBuyCondition
            else:
                self._position = Positions.Hold
        else:
            self._position = Positions.Hold
            # print("Short buy action check triggered")
            # if not self.onBuyShort:
            #     print("Short Buying Action Occurred")
            #     print("Bought at ", self.prices[self._current_tick])
            #     self._position = Positions.Short
            #     short_buy = True
            #     self._last_trade_tick = self._current_tick
            #     self.lastBuyShort = self._current_tick
            #     self.onBuyShort = True
            #     # self.short_sell_interval = random.randint(wait_interval_min,wait_interval_max)
            # elif self.onBuy and self._position == Positions.Long:
            #     if not self.onBuyShort:
            #         print("Short Buying Action Occurred 2")
            #         print("Bought at ", self.prices[self._current_tick])
            #         self._position = Positions.Short
            #         short_buy = True
            #         self._last_trade_tick = self._current_tick
            #         self.lastBuyShort = self._current_tick
            #         self.onBuyShort = True
            #         # self.short_sell_interval = random.randint(wait_interval_min,wait_interval_max)
            #     else:
            #         print("Inner Hold")
            #         self._position = Positions.Hold
            # # elif self.onBuy and self._position == Positions.Short:
            # #     self._position = Positions.Hold
            # else:
            #     print("Middle Hold")
            #     self._position = Positions.Hold

    def doLongProfitOrStop(self,curr_price):
        print("Selling Regular Action Occurred")
        print("Sold at ", curr_price)
        self._position = Positions.LongSell
        self._last_trade_tick = self._current_tick
        long_sell = True
        long_buy = False
        self.onBuy = False
        action = Actions.Sell.value
        self.long_order_completed = True

    def doShortProfitOrStop(self, curr_price):
        print("Short Regular Selling Action Occurred")
        print("Sold at ", self.prices[self._current_tick], "Buy price: ", self.prices[self.lastBuyLong])
        self._position = Positions.ShortSell
        short_sell = True
        short_buy = False
        self.onBuyShort = False
        self._last_trade_tick = self._current_tick
        action = Actions.ShortSell.value
        self.short_order_completed = True
