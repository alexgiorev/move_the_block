import sys
from pprint import pprint as pp
import itertools
import os.path
from collections import namedtuple, deque

Action = namedtuple("Action", "posn diff")
Node = namedtuple("Node", "state parent action")
Block = namedtuple("Block", "orient size token")

class Board:
    ACTION_COUNT = None
    
    def __init__(self, data, goal_block):
        self.data = tuple(map(tuple, data))
        self.goal_block = goal_block
    
    @classmethod
    def from_str(cls, text):
        tokens = [line.split(" ") for line in text.strip().split("\n")]
        return cls.from_tokens(tokens, "x", "_")

    @classmethod
    def from_tokens(cls, tokens, goal_token, empty_token):
        tokens = list(map(list, tokens))
        goal_block = None
        tokens_blocks = {}
        for row in range(6):
            for col in range(6):
                token = tokens[row][col]
                if token == empty_token:
                    tokens[row][col] = None
                elif token not in tokens_blocks:
                    try:
                        orient = "H" if tokens[row][col+1] == token else "V"
                    except IndexError:
                        orient = "V"
                    # compute size
                    if orient == "H":
                        size = 2 if col > 3 or tokens[row][col+2] != token else 3
                    else:
                        size = 2 if row > 3 or tokens[row+2][col] != token else 3
                    block = Block(orient,size,token)
                    tokens_blocks[token] = block
                    if token == goal_token:
                        goal_block = block
                    tokens[row][col] = block
                else:
                    tokens[row][col] = tokens_blocks[token]
        return cls(tokens, goal_block)
        
    def __hash__(self):
        return hash(self.data)
    
    def __eq__(self, other):
        return type(self) is type(other) and self.data == other.data

    def __getitem__(self, posn):
        row,col = posn
        return self.data[row][col]

    def output_char(self):
        matrix = list(map(list, self.data))
        for row,col in self.posns:
            block = matrix[row][col]
            if block is None:
                matrix[row][col] = "_"
            else:
                matrix[row][col] = block.token
        text = "\n".join(" ".join(row) for row in matrix)
        return text

    def search_bfs(self):
        """Returns the sequence of actions which solve the problem"""
        Board.ACTION_COUNT = 0
        root = Node(state=self, parent=None, action=None)
        frontier = deque([root])
        existing = set()
        count = 0
        while frontier:
            node = frontier.popleft()
            for successor, action in node.state.successors:
                if successor.is_solution:
                    # From the sequence of actions and return it
                    actions = []
                    while action is not None:
                        actions.append(action)
                        node = node.parent
                        action = node.action
                    actions.reverse()
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
            if self[row,col] is not None:
                return False
        return True

    @property
    def goal_block_position(self):
        for posn in self.posns:
            if self[posn] is self.goal_block:
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
        for (row, col), block in self.blocks:
            if block.orient == "H":
                for diff in range(-1,-5,-1):
                    dcol = col+diff
                    if dcol >= 0 and self[row,dcol] is None:
                        action = Action(posn=(row,col), diff=(0,diff))
                        actions.append(action)
                    else:
                        break
                for diff in range(1,5):
                    dcol = col+(block.size-1)+diff
                    if dcol <= 5 and self[row,dcol] is None:
                        action = Action(posn=(row,col), diff=(0,diff))
                        actions.append(action)
                    else:
                        break
            else:
                # analogous to the horizontal case
                for diff in range(-1,-5,-1):
                    drow = row+diff
                    if drow >= 0 and self[drow, col] is None:
                        action = Action(posn=(row,col), diff=(diff,0))
                        actions.append(action)
                    else:
                        break
                for diff in range(1,5):
                    drow = row+(block.size-1)+diff
                    if drow <= 5 and self[drow, col] is None:
                        action = Action(posn=(row,col), diff=(diff,0))
                        actions.append(action)
                    else:
                        break
        return actions
    
    def apply_action(self, action):
        Board.ACTION_COUNT += 1
        new_data = list(map(list, self.data))
        (row,col),(row_diff,col_diff) = action.posn, action.diff
        block = self[row,col]
        dr,dc = (0,1) if block.orient == "H" else (1,0)
        posns = [(row,col)]
        for _ in range(block.size-1):
            r,c = posns[-1]
            posns.append((r+dr,c+dc))
        for r,c in posns:
            new_data[r][c] = None
        for r,c in posns:
            r,c = r+row_diff,c+col_diff
            new_data[r][c] = block
        return Board(new_data, self.goal_block)

    @property
    def blocks(self):
        """Returns a list of pairs (POSN, BLOCK)"""
        result = []
        blocks = set()
        for row,col in self.posns:
            block = self[row,col]
            if block is not None and block not in blocks:
                result.append(((row,col),block))
                blocks.add(block)
        return result

#════════════════════════════════════════
# main

PATH = "board"
def get_board():
    text = open(PATH).read()
    return Board.from_str(text)
    
def main_char(cols):
    board = get_board()
    actions = board.search_bfs()
    print(f"### The solution has {len(actions)} actions")
    print(f"### It took {Board.ACTION_COUNT} actions to find this solution")
    texts = [board.output_char()]
    for action in actions:
        char = board[action.posn].token
        board = board.apply_action(action)
        text = board.output_char().replace(char, char.upper())
        texts.append(text)
    texts = [text.split("\n") for text in texts]
    board_rows = []
    i = 0
    while i < len(texts):
        row = []
        for j in range(min(len(texts)-i,cols)):
            row.append(texts[i])
            i += 1
        board_rows.append(row)
    empty_board = [" " * 11 for _ in range(6)]
    while len(board_rows[-1]) % cols != 0:
        board_rows[-1].append(empty_board)
    print(("═"*13+"╦") * (cols-1) + ("═"*13))
    text_rows = []
    for board_row in board_rows:
        lines = []
        for line_index in range(6):
            line = " " + " ║ ".join(board[line_index] for board in board_row) + " "
            lines.append(line)
        text_rows.append("\n".join(lines))
    row_separator = "\n" + ("═"*13+"╬") * (cols-1) + ("═"*13) + "\n"
    print(row_separator.join(text_rows))
    footer = ("═"*13+"╩") * (cols-1) + ("═"*13)
    print(footer)
        
