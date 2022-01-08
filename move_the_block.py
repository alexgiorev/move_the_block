import sys
from pprint import pprint as pp
import itertools
import os.path
from collections import namedtuple, deque

Action = namedtuple("Action", "posn diff")
Node = namedtuple("Node", "state parent action")
class Board:
    HS, HM, HE = "hs", "hm", "he"
    VS, VM, VE = "vs", "vm", "ve"
    GS, GE = "gs", "ge"
    EMPTY = "__"
    HORIZONTALS = (HS,HM,HE,GS,GE)
    VERTICALS = (VS,VM,VE)

    def __init__(self, data):
        self.data = tuple(map(tuple, data))
    
    @classmethod
    def from_str(cls, text):
        data = [line.split(" ") for line in text.split("\n")]
        data.pop()
        return Board(data)

    def __hash__(self):
        return hash(self.data)
    
    def __eq__(self, other):
        return type(self) is type(other) and self.data == other.data

    def __getitem__(self, posn):
        row,col = posn
        return self.data[row][col]

    def __str__(self):
        matr = list(map(list, self.data))
        for row,col in self.posns:
            value = matr[row][col]
            if value == self.EMPTY:
                matr[row][col] = " "
            elif value in self.HORIZONTALS:
                matr[row][col] = "═"
            else:
                matr[row][col] = "║"
        pp(matr)
        lines = ["".join(row) for row in matr]
        top = "┌" + "─"*len(lines[0]) + "┐"
        bottom = "└" + "─"*len(lines[0]) + "┘"
        lines = [top] + ["│"+line+"│" for line in lines] + [bottom]
        return "\n".join(lines)

    def search_bfs(self):
        """Returns the sequence of actions which solve the problem"""
        root = Node(state=self, parent=None, action=None)
        frontier = deque([root])
        existing = set()
        count = 0
        while frontier:
            node = frontier.popleft()
            successors = node.state.successors
            count += 1
            for successor, action in successors:
                if successor.is_solution:
                    # From the sequence of actions and return it
                    actions = []
                    while action is not None:
                        actions.append(action)
                        node = node.parent
                        action = node.action
                    actions.reverse()
                    print(f"### count=={count}")
                    return actions
                else:
                    if successor not in existing:
                        child = Node(state=successor, parent=node, action=action)
                        frontier.append(child)
                        existing.add(successor)

    @property
    def is_solution(self):
        row,col = self.goal_block_position
        for col in range(col+2, 6):
            if not self[row,col] == self.EMPTY:
                return False
        return True

    @property
    def goal_block_position(self):
        for posn in self.posns:
            if self[posn] == self.GS:
                return posn

    @property
    def posns(self):
        return itertools.product(range(6), range(6))
        
    @property
    def successors(self):
        """Returns a list of pairs (SUCCESSOR, ACTION),
        where applying ACTION on SELF will result in SUCCESSOR"""
        return [(self.apply_action(action), action)
                for action in self.possible_actions]
            
    @property
    def possible_actions(self):
        actions = []
        for (row, col), size, direction in self.blocks:
            if direction == "horizontal":
                for diff in range(-1,-5,-1):
                    dcol = col+diff
                    if dcol >= 0 and self[row,dcol] == self.EMPTY:
                        action = Action(posn=(row,col), diff=diff)
                        actions.append(action)
                    else:
                        break
                for diff in range(1,5):
                    dcol = col+(size-1)+diff
                    if dcol <= 5 and self[row,dcol] == self.EMPTY:
                        action = Action(posn=(row,col), diff=diff)
                        actions.append(action)
                    else:
                        break
            elif direction == "vertical":
                # 100% analogous to the "horizontal" case
                for diff in range(-1,-5,-1):
                    drow = row+diff
                    if drow >= 0 and self[drow, col] == self.EMPTY:
                        action = Action(posn=(row,col), diff=diff)
                        actions.append(action)
                    else:
                        break
                for diff in range(1,5):
                    drow = row+(size-1)+diff
                    if drow <= 5 and self[drow, col] == self.EMPTY:
                        action = Action(posn=(row,col), diff=diff)
                        actions.append(action)
                    else:
                        break
        return actions
    
    def apply_action(self, action):
        new_data = list(map(list, self.data))
        (row,col),diff = action.posn, action.diff
        size = None
        if self[row,col] in (self.GS, self.HS):
            dr,dc = (0,1)
            size = 2 if self[row, col+1] in (self.HE, self.GE) else 3
        else:
            dr,dc = (1,0)
            size = 2 if self[row+1, col] == self.VE else 3
        # extract the components (start, middle, end)
        components = []
        for k in range(size):
            drow, dcol = row+k*dr, col+k*dc
            components.append(new_data[drow][dcol])
            new_data[drow][dcol] = self.EMPTY
        # place it at the correct position
        row,col = row+dr*diff, col+dc*diff
        for k in range(size):
            drow, dcol = row+k*dr, col+k*dc
            new_data[drow][dcol] = components[k]
        return Board(new_data)

    @property
    def blocks(self):
        """Returns a list of triples (POSN, SIZE, DIRECTION), where POSN is the
        beginning position of some block (the leftmost square for "horizontal"
        blocks and the topmost for "vertical" blocks) whose size is SIZE and
        whose direction is DIRECTION. A possible triple is ((2,2),3,"horizontal")"""
        result = []
        for row,col in self.posns:
            if self[row,col] == self.HS:
                if self[row,col+1] == self.HE:
                    result.append(((row,col),2,"horizontal"))
                else:
                    result.append(((row,col),3,"horizontal"))
            if self[row,col] == self.GS:
                result.append(((row,col),2,"horizontal"))
            elif self[row,col] == self.VS:
                if self[row+1,col] == self.VE:
                    result.append(((row,col),2,"vertical"))
                else:
                    result.append(((row,col),3,"vertical"))
        return result

#════════════════════════════════════════
# main

PATH = "board"
def get_board():
    text = open(PATH).read()
    return Board.from_str(text)
    
def main():
    board = get_board()
    pp(board.search_bfs())
