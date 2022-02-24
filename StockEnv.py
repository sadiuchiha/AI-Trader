import numpy as np

from CustomEnvVer2 import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)
        self.reward_multiplier = 1
        self.penalty_multiplier = 1

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _update_profit(self, action):

        isTrade = action == Actions.Buy.value  or \
                  action == Actions.Sell.value  or \
                  action == Actions.ShortBuy.value  or \
                  action == Actions.ShortSell.value

        if isTrade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            fee_ask = 1 - self.trade_fee_ask_percent
            fee_bid = (1 - self.trade_fee_bid_percent)
            print("Current Sell: ", current_price)
            sell = current_price + (current_price * self.trade_fee_bid_percent)
            print("With Fee Current Sell: ", current_price)

            if self._position == Positions.LongSell:
                print("Long Sell Profit/Loss calculation")
                print("Sell price: ", self.prices[self.lastBuyLong])
                buy = self.prices[self.lastBuyLong] + (self.prices[self.lastBuyLong] * self.trade_fee_ask_percent)
                print("Sell price: ", buy)
                if sell > buy:
                    print("Sell Profit Occured!")
                    print("Total Profit: ", self._total_profit)
                    print("Current Sell: ", current_price)
                    print("Buy: ", buy, "Sell: ", sell)
                    profit = (sell - buy) / buy
                    print("Profit: ", profit)
                    self._total_profit += self._total_profit * profit
                    print("Total Profit: ", self._total_profit)

                elif sell < buy:
                    print("Sell Loss Occured!")
                    print("Total Profit: ", self._total_profit)
                    print("Current Sell: ", current_price)
                    print("Buy: ", buy, "Sell: ", sell)
                    loss = (sell - buy) / buy
                    print("Loss: ", loss)
                    self._total_profit += self._total_profit * loss
                    print("Total Profit: ", self._total_profit)

            elif self._position == Positions.ShortSell:
                print("Short Sell Profit/Loss calculation")
                # profit_loss = current_price - (1 - (self.prices[self.lastBuyShort]/current_price) - shra -)
                buy = self.prices[self.lastBuyShort] + (self.prices[self.lastBuyShort] * self.trade_fee_ask_percent)
                if sell > buy:
                    print("Short Sell Loss Occured!")
                    print("Total Profit: ", self._total_profit)
                    print("Current Sell: ", current_price)
                    print("Buy: ", buy, "Sell: ", sell)
                    loss = (sell - buy) / buy
                    print("Loss: ", loss)
                    self._total_profit -= self._total_profit * loss
                    print("Total Profit: ", self._total_profit)

                elif sell < buy:
                    print("Short Sell Profit Occured!")
                    print("Total Profit: ", self._total_profit)
                    print("Current Sell: ", current_price)
                    print("Buy: ", buy, "Sell: ", sell)
                    profit = -(sell - buy) / buy
                    print("Profit: ", profit)
                    self._total_profit += self._total_profit * profit
                    print("Total Profit: ", self._total_profit)

    def is_long_profit(self, current_price, buy_price):
        profit = ((current_price - buy_price) / buy_price)
        print("onBuy: ", self.onBuy, "profit: ", profit)
        if not isinstance(profit, float):
            print("Array changed to float")
            profit = profit[0][len(profit[0]) - 1]
        if self.lastBuyLong is None:
            return False
        elif profit < 0.02:
            return False
        else:
            return True

    def is_short_profit(self, current_price, buy_price):
        print("Current and bought price: ", current_price, buy_price)
        profit = ((buy_price - current_price) / buy_price)
        print("onBuyShort: ", self.onBuyShort, "profit: ", profit)
        if not isinstance(profit, float):
            print("Array changed to float")
            profit = profit[0][len(profit[0]) - 1]
        if self.lastBuyShort is None:
            return False
        elif profit < 0.02:
            return False
        else:
            return True

    def is_long_stop(self, current_price, buy_price):
        stop = ((current_price - buy_price) / buy_price)
        print("onBuy: ", self.onBuy, "profit: ", stop)
        if not isinstance(stop, float):
            print("Array changed to float")
            stop = stop[0][len(stop[0]) - 1]
        if self.lastBuyLong is None:
            return False
        elif stop > -0.010:
            return False
        else:
            return True

    def is_short_stop(self, current_price, buy_price):
        stop = ((buy_price - current_price) / buy_price)
        print("onBuyShort: ", self.onBuyShort, "profit: ", stop)
        if not isinstance(stop, float):
            print("Array changed to float")
            stop = stop[0][len(stop[0]) - 1]
        if self.lastBuyShort is None:
            return False
        elif stop > -0.010:
            return False
        else:
            return True

    def _calculate_reward(self, action):

        # prices = self.prices
        # long_profit = self.is_long_profit(prices[self._current_tick], prices[self.lastBuyLong])
        # long_stop = self.is_long_stop(prices[self._current_tick], prices[self.lastBuyLong])
        # short_profit = self.is_short_profit(prices[self._current_tick], prices[self.lastBuyLong])
        # short_stop = self.is_short_stop(prices[self._current_tick], prices[self.lastBuyLong])

        step_reward = 0
        profit_made = False
        isTrade = self._position == Positions.ShortSell or Positions.LongSell
        # Conditions for a Buy/Sell/Hold action

        if isTrade:
            current_price = self.prices[self._current_tick]
            if self._position == Positions.LongSell:
                last_trade_price = self.prices[self.lastBuyLong]
                price_diff = current_price - last_trade_price
                step_reward += price_diff * self.long_acc

            if self._position == Positions.ShortSell:
                last_trade_price = self.prices[self.lastBuyShort]
                price_diff = last_trade_price - current_price
                step_reward += price_diff * self.short_acc

        # add hold functionality later

        current_price = self.prices[self._current_tick]

        if self._position == Positions.Long:
            step_reward += 1.5
        elif self._position == Positions.LongSell:
            price_diff = current_price - self.prices[self.lastBuyLong]
            dif_price_ratio = (price_diff / current_price)
            hold_reward = self.long_hold_reward_accumulator
            print("price difference: ", dif_price_ratio)
            if dif_price_ratio > 0:
                if not self.long_order_completed:
                    print("regular interval long_order_completed ", )
                    step_reward += 1 + (self.long_acc * dif_price_ratio)
                    print("Step Reward: ", step_reward)
                    self.long_acc = 0
                    profit_made = True
                else:
                    print("long_order_completed ", )
                    self.long_order_completed = False
                    step_reward += 1.5 + (self.long_acc * dif_price_ratio)
                    self.long_acc = 0
                    profit_made = True
                    print("Step Reward: ", step_reward)

            else:
                if not self.long_order_completed:
                    print("regular interval long_order_completed ", )
                    step_reward -= 1 - (self.long_acc * dif_price_ratio)
                    print("Step Reward: ", step_reward)
                    self.long_acc = 0

                else:
                    print("long_order_completed ", )
                    self.long_order_completed = False
                    step_reward -= 1.5 - (self.long_acc * dif_price_ratio)
                    print("Step Reward: ", step_reward)
                    self.long_acc = 0

        elif self._position == Positions.Short:
            step_reward += 1.5
        elif self._position == Positions.ShortSell:
            price_diff = self.prices[self.lastBuyShort] - current_price
            dif_price_ratio = (price_diff / current_price)
            hold_reward = self.short_hold_reward_accumulator
            print("price difference: ", dif_price_ratio)
            if dif_price_ratio < 0:
                if not self.short_order_completed:
                    print("regular interval short_order_completed " )
                    step_reward -= 1 - (self.short_acc * dif_price_ratio)
                    self.short_acc = 0
                    profit_made = True
                    print("Step Reward: ", step_reward)

                else:
                    print("short_order_completed " )
                    step_reward -= 1.5 - (self.short_acc * dif_price_ratio)
                    self.short_order_completed = False
                    self.short_acc = 0
                    profit_made = True
                    print("Step Reward: ", step_reward)

            else:
                if not self.short_order_completed:
                    print("regular interval short_order_completed " )
                    step_reward += 1 + self.short_acc * dif_price_ratio
                    self.short_acc = 0
                    print("Step Reward: ", step_reward)

                else:
                    print("short_order_completed " )
                    step_reward += 1.5 + self.short_acc * dif_price_ratio
                    self.short_order_completed = False
                    self.short_acc = 0
                    print("Step Reward: ", step_reward)

        elif self._position == Positions.Hold:
            step_reward = 0.0
            # self.long_acc += 1
            # self.short_acc += 1

            if self.onBuy or self.onBuyShort:
                reward = 0.0
                long_prit = 0.0
                short_prit = 0.0

                if self.onBuy:
                    price_dif = current_price - self.prices[self.lastBuyLong]
                    if self.sell_interval < 1:
                        print("reducing reward extended hold long interval less than 1")
                        reward = step_reward + (price_dif/current_price)
                        if action == Actions.Sell.value and self._position != Positions.LongSell:
                            reward = (2 * (step_reward + (price_dif / current_price)))
                            self.long_acc += reward
                            print("self.long_acc: ", reward)
                    else:
                        reward = step_reward + (price_dif / current_price)
                        self.long_acc += reward
                        print("self.long_acc: ", reward)

                long_prit = reward
                self.long_hold_reward_accumulator = reward

                if self.onBuyShort:
                    price_dif = self.prices[self.lastBuyLong] - current_price
                    if self.short_sell_interval < 1:
                        print("reducing reward extended hold short interval less than 1")
                        reward = step_reward + (price_dif / current_price)
                        if action == Actions.ShortSell.value and self._position != Positions.ShortSell:
                            reward = (2 * (step_reward + (price_dif / current_price)))
                            self.short_acc += reward
                            print("self.short_acc: ", reward)
                    else:
                        reward = step_reward + (price_dif / current_price)
                        self.short_acc += reward
                        print("self.short_acc: ", reward)

                short_prit = reward
                self.short_hold_reward_accumulator = reward

                print("Long Priority: ", long_prit,"Short Priority: ", short_prit)
                profits = self._total_profit
                isFloat_long_prit = isinstance(long_prit, np.float)
                isFloat_short_prit = isinstance(short_prit, np.float)

                if not isFloat_long_prit:
                    list = np.asarray(long_prit)
                    print("Shape: ", list.shape)
                    print("List: ", list)
                    long_prit = long_prit[0][len(long_prit[0]) - 1]

                if not isFloat_short_prit:
                    list = np.asarray(short_prit)
                    print("Shape: ", list.shape)
                    print("List: ", list)
                    short_prit = short_prit[0][len(short_prit[0]) - 1]

                if long_prit != short_prit:
                    if long_prit > short_prit:
                        step_reward = long_prit
                    else:
                        step_reward = short_prit
            else:
                step_reward = -0.4
                print("Accumulator reset")
                if not self.onBuyShort: self.short_hold_reward_accumulator = 0
                if not self.onBuy: self.long_hold_reward_accumulator = 0

        print("total profit: ", self._total_profit)
        print("total profit type: ", type(self._total_profit))

        profits = self._total_profit
        rewards = self._total_reward
        if not isinstance(profits,np.float):
            list = np.asarray(self._total_profit)
            print("Shape: ", list.shape)
            print("List: ", list)
            self._total_profit = list[0][len(list[0])-1]
        if not isinstance(rewards,np.float):
            list = np.asarray(rewards)
            print("Shape: ", list.shape)
            print("List: ", list)
            self._total_reward = list[0][len(list[0])-1]

        if isTrade:
            if self._total_profit < 1.0:
                # if profit_made: self.penalty_multiplier += 1
                if self._total_profit < 0.97:
                    self._total_reward -= self._total_reward * 0.03
                if self._total_profit < 0.94:
                    self._total_reward -= self._total_reward * 0.06
                if self._total_profit < 0.91:
                    self._total_reward -= self._total_reward * 0.09
                if self._total_profit < 0.88:
                    self._total_reward -= self._total_reward * 0.12
                if self._total_profit < 0.85:
                    self._total_reward -= self._total_reward * 0.15
                if self._total_profit < 0.82:
                    self._total_reward -= self._total_reward * 0.18
                if self._total_profit < 0.79:
                    self._total_reward -= self._total_reward * 0.21
                if self._total_profit < 0.76:
                    self._total_reward -= self._total_reward * 0.24

            else:
                # if profit_made: self.reward_multiplier += 0.4
                if self._total_profit > 1.03:
                    self._total_reward += self._total_reward * 0.03
                if self._total_profit > 1.06:
                    self._total_reward += self._total_reward * 0.06
                if self._total_profit > 1.09:
                    self._total_reward += self._total_reward * 0.09
                if self._total_profit > 1.12:
                    self._total_reward += self._total_reward * 0.12
                if self._total_profit > 1.15:
                    self._total_reward += self._total_reward * 0.15
                if self._total_profit > 1.18:
                    self._total_reward += self._total_reward * 0.18
                if self._total_profit > 1.21:
                    self._total_reward += self._total_reward * 0.21
                if self._total_profit > 1.24:
                    self._total_reward += self._total_reward * 0.24
        print("Final Step reward: ", step_reward)
        is_number = isinstance(step_reward, float)
        print("step reward is array? ", is_number)
        if not is_number:
            print(step_reward)
            rw = step_reward[0]
            step_reward = rw[len(rw)-1]
        return step_reward

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit

    # def predict_next(self, ):


