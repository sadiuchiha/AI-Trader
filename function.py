
# Will contain frames and select the highest to lowest order count of frames and also will show prev frames
# Will Compare count of selected frame, and will bring the highest unique count of previous frames
# Choose the previous frame then set Action (Buy, Sell, ShortBuy, ShortSell)
# Will Create action based on previous frame as a condition in a condition_list
# on CustomEnvVer2 step() will compare each Trend Frame of current tick and match pattern and perform that action.
from operator import itemgetter

import numpy as np


class FunctionMaker():
    uni_frame = []
    indexes = []
    conditions = []
    trend_frames = []

    def new(self, uni_frame, indexes, trend_frames):
        self.uni_frame = uni_frame
        self.indexes = indexes
        self.trend_frames = trend_frames
        self.conditions = self.createConditions()

    def createConditions(self):
        uni_pattern = self.uni_frame
        indexes = self.indexes
        conditions = []

        ch = input("Set conditions enter y: ")

        if ch == "y":
            combinations = input("Enter combination format( count=n action = 1/3 or skip= -1) "
                                 "\nExample: 1 3 -1\n=")
            combinations = combinations.replace(" ", "")
            combination_actions = []
            combinations_exist = False
            if combinations != "n":
                combinations_exist = True
                skip = False
                skip_trig = 0
                for t in range(len(combinations)):
                    print("Combination: ", t)
                    count3 = combinations[t]
                    print("count3: ", count3)
                    action = "-1"

                    if skip_trig + 1 < t:
                        skip = False

                    if not skip:
                        if combinations[t] == "?":
                            count3 = "?"
                        elif combinations[t] != "-" and "?":
                            action = combinations[t + 1]
                            skip = True
                            skip_trig = t
                        # elif combinations[t] == "-":
                        else:
                            count3 = "-1"
                            skip = True
                            skip_trig = t

                    print("Combination: ", t, "count3: ", count3, "action: ", action)
                    if skip_trig == t:
                        print("Combination: ", t, "Added")
                        combination_actions.append([count3, int(action)])
            # self.setCombination()

            for i in range(len(indexes)):

                curr_index = indexes[i][0]
                count = uni_pattern[curr_index][1]
                print("Looking at Pattern: ", uni_pattern[curr_index][0])
                prev_patterns = []
                current_pattern = uni_pattern[curr_index][0]



                for z in range(uni_pattern[curr_index][1]):

                    matched = False
                    if not isinstance(uni_pattern[curr_index][2][z][0], int):
                        current_tick = uni_pattern[curr_index][2][z][0][0]
                        prev_status = uni_pattern[curr_index][2][z][0][3]

                    else:
                        current_tick = uni_pattern[curr_index][2][z][0]
                        prev_status = uni_pattern[curr_index][2][z][3]


                    for v in range(len(prev_patterns)):
                        # print(prev_status, prev_patterns[v][0])
                        if prev_patterns[v][0] == prev_status:
                            # print("matched")
                            prev_patterns[v][1] += 1
                            matched = True

                    if not matched:
                        prev_patterns.append([prev_status, 1])
                        # print("Not matched")

                    # Sort prev per count
                order_pos = []
                for j in range(len(prev_patterns)):
                    order_pos.append([j, prev_patterns[j][1]])
                order_pos = sorted(order_pos, key=itemgetter(1), reverse=True)
                ch1 = "y"

                if combinations_exist:
                    print("i: ", i)
                    print("combination_actions: ", combination_actions[i])
                    # combination_actions[z]
                    count3 = combination_actions[i][0]
                    action = combination_actions[i][1]
                    if count3 != "?":
                        count3 = int(count3)
                    pattern_index = count3

                    if count3 == "?":

                        while ch1 != "n":
                            self.showPrevPatterns(prev_patterns, order_pos)

                            pattern_index = input("Choose number of previous pattern with high count: ")

                            if pattern_index[0] == "=":
                                pattern_index = int(pattern_index[1:]) - 1
                                ch = input("Choose action: (0-Hold, 1-Buy, 2-Sell, 3-Short, 4-ShortSell)")

                                index = order_pos[pattern_index][0]
                                print("Appending Pattern ", prev_patterns[index][0])
                                conditions.append(self.makeCondition(prev_patterns[index][0], ch, current_pattern, count))
                            elif "to" in pattern_index:
                                from_point = int(pattern_index.rpartition("to")[0])
                                till_point = int(pattern_index.rpartition("to")[2])
                                ch = input("Choose action: (0-Hold, 1-Buy, 2-Sell, 3-Short, 4-ShortSell)")

                                for l in range(till_point):
                                    if l == 0:
                                        l = from_point - 1
                                    index = order_pos[l][0]
                                    print("Appending Pattern ", prev_patterns[index][0])
                                    conditions.append(self.makeCondition(prev_patterns[index][0], ch, current_pattern, count))
                            else:
                                pattern_index = int(pattern_index)
                                if 0 < pattern_index <= len(order_pos):

                                    ch = input("Choose action: (0-Hold, 1-Buy, 2-Sell, 3-Short, 4-ShortSell)")

                                    for l in range(pattern_index):
                                        index = order_pos[l][0]
                                        print("Appending Pattern ", prev_patterns[index][0])
                                        conditions.append(self.makeCondition(prev_patterns[index][0], ch, current_pattern, count))
                                else:
                                    print("Pattern not added to condition")
                                    # ch = input("Add another pattern? y/n ")
                            ch1 = input("Add patterns from current pattern? y/n ")

                    else:
                #         add combination
                        for l in range(pattern_index):
                            index = order_pos[l][0]
                            print("Appending Pattern ", prev_patterns[index][0])
                            conditions.append(self.makeCondition(prev_patterns[index][0], action, current_pattern, count))
                    if i == len(combination_actions)-1:
                        combinations_exist = False
                else:
                    while ch1 != "n":
                    # add elements to previous_pattern list / (=index, x to y, x)
                        self.showPrevPatterns(prev_patterns,order_pos)
                        pattern_index = input("Choose number of previous pattern with high count: ")
                        if pattern_index[0] == "=":
                            pattern_index = int(pattern_index[1:]) - 1
                            ch = input("Choose action: (0-Hold, 1-Buy, 2-Sell, 3-Short, 4-ShortSell)")

                            index = order_pos[pattern_index][0]
                            print("Appending Pattern ", prev_patterns[index][0])
                            conditions.append(self.makeCondition(prev_patterns[index][0], ch, current_pattern, count))
                        elif "to" in pattern_index:
                            from_point = int(pattern_index.rpartition("to")[0])
                            till_point = int(pattern_index.rpartition("to")[2])
                            ch = input("Choose action: (0-Hold, 1-Buy, 2-Sell, 3-Short, 4-ShortSell)")

                            for l in range(till_point):
                                if l == 0:
                                    l = from_point - 1
                                index = order_pos[l][0]
                                print("Appending Pattern ", prev_patterns[index][0])
                                conditions.append(self.makeCondition(prev_patterns[index][0], ch, current_pattern, count))
                        else:
                            pattern_index = int(pattern_index)
                            if 0 < pattern_index <= len(order_pos):

                                ch = input("Choose action: (0-Hold, 1-Buy, 2-Sell, 3-Short, 4-ShortSell)")

                                for l in range(pattern_index):
                                    index = order_pos[l][0]
                                    print("Appending Pattern ", prev_patterns[index][0])
                                    conditions.append(self.makeCondition(prev_patterns[index][0], ch, current_pattern, count))
                            else:
                                print("Pattern not added to condition")
                                # ch = input("Add another pattern? y/n ")
                        ch1 = input("Add patterns from current pattern? y/n ")
                stop = "n"
                if not combinations_exist:
                    stop = input("Stop making multiple condtions y/n")
                if stop == "y":
                    break
        #         Do operations
        print("Conditions: ", conditions)
        conditions = self.organizeConditions(conditions)
        return conditions

    def showPrevPatterns(self,prev_patterns,order_pos):
        print("Pattern List: \n")
        for i in range(len(order_pos)):
            index = order_pos[i][0]
            pattern_count = self.getPatternCount(prev_patterns[index][0], -1)
            if pattern_count == 0:
                chance = 0
            else:
                chance = (prev_patterns[index][1] / pattern_count) * 100
            print("Pattern " + str(i+1) + ": ", prev_patterns[index][0], " Count: ", prev_patterns[index][1], "Ratio: " + str(prev_patterns[index][1]) +"/"+str(pattern_count), " Chance appearance: " + str("%.2f" % chance) + "%")

    def makeCondition(self, pattern, ch, curr, count):
        print("Ch: ", ch)
        ch = int(ch)
        action = "Not Set"
        if ch == 0:
            action = "Hold"
        elif ch == 1:
            action = "Buy"
        elif ch == 2:
            action = "Sell"
        elif ch == 3:
            action = "ShortBuy"
        elif ch == 4:
            action = "ShortSell"
        return [pattern, action, curr, count]

    def getPatternCount(self, pattern, total_count):

        count = 0
        for i in range(len(self.trend_frames)):
            if pattern == self.trend_frames[i][0]:
                count += 1

        if total_count == -1:
            return count
        else:
            if count == total_count:
                return 1
            else:
                return count

    # Conditions with count > 1 will have n times the ratio of the probability
    def organizeConditions(self, conditions):

        single_cond = []
        multi_cond = []
        # conditions[0] = [pattern, action, curr, count]
        for i in range(len(conditions)):
            print("Condition " + str(i) + ": ", conditions[i])
            previous = conditions[i][0]
            action_choosen = conditions[i][1]
            current = conditions[i][2]
            curr_count = conditions[i][3]
            matched = False
            if i == 0:
                total_count = self.getPatternCount(previous, curr_count)
                single_cond.append([previous, action_choosen, current, 1,curr_count,total_count])
            else:
                for j in range(len(single_cond)):
                    condition = single_cond[j]
                    if single_cond[j][0] == previous:
                        matched = True
                        condition[3] += 1
                        if condition[3] > 1:
                            action = condition[1]
                            curr = condition[2]
                            prev = condition[0]
                            count = condition[3]
                            past_count = condition[4]
                            total_prev_count = self.getPatternCount(prev,past_count)
                            multi_cond.append([prev,[[action,curr,past_count,total_prev_count],[action_choosen, current,curr_count,total_count]], count])
                            del single_cond[j]
                            break
                for z in range(len(multi_cond)):
                    if multi_cond[z][0] == previous and not matched:
                        action = action_choosen
                        curr = current
                        past_count = curr_count
                        total_prev_count = self.getPatternCount(previous,past_count)
                        cond_action = [action,curr, past_count, total_prev_count]
                        multi_cond[z][2] += 1
                        multi_cond[z][1].append(cond_action)
                        print("len: ", len(multi_cond[z][1]))
                        print("con: ", multi_cond[z][1])

                        matched = True
                        break

                if not matched:
                    total_count = self.getPatternCount(previous, curr_count)
                    single_cond.append([previous, action_choosen, current, 1, curr_count,total_count])
        list = np.asarray(multi_cond)
        print("Shape: " + str(list.shape))
        # print(list[0])
        # self.showConditions(single_cond, multi_cond)
        return [single_cond, multi_cond]

    def showConditions(self, single_cond, multi_cond):
        print("Single Conditions: \n")
        for i in range(len(single_cond)):
            print("Pattern: ", single_cond[i])
        print("\nMultiple Conditions: \n")

        for i in range(len(multi_cond)):
            print("Pattern: ", multi_cond[i][0], "Count: ", multi_cond[i][2])
            print(" ")
            actions = multi_cond[i][1]

            for j in range(len(actions)):
                print("Conditions " + str(j+1) + ": ", actions[j][1], "Action: ", actions[j][0])
                print(" ")

    def countPatternOccurence(self, prev, curr):
        # for i in range()
        pass

