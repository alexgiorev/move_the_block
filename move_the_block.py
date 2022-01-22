import sys
import networkx
import itertools
import os.path
import math
import shutil
import pdb

from collections import namedtuple, deque
from pprint import pprint as pp
from PIL import Image

Action = namedtuple("Action", "posn diff")
Block = namedtuple("Block", "orient size token")

class Node:
    def __init__(self, state, parent, action, name="X"):
        self.state = state
        self.parent = parent
        self.action = action
        self.name = name
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def serialize_for_emacs(self):
        if not self.children:
            return self.name
        children = (child.serialize_for_emacs() for child in self.children)
        strings = itertools.chain([f"({self.name}"],children,[")"])
        return " ".join(strings)

class Board:
    ACTION_COUNT = 0
    
    def __init__(self, data, goal_block):
        self.data = tuple(map(tuple, data))
        self.goal_block = goal_block
    
    @classmethod
    def from_str(cls, text):
        tokens = [line for line in text.strip().split("\n")]
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

    def image(self):
        board_img = BoardImage()
        for (row,col),block in self.blocks:
            orient = "G" if block is self.goal_block else block.orient
            block_type = f"{orient}{block.size}"
            board_img.draw(block_type,(col,row))
        return board_img.board

    def state_space(self):
        graph = networkx.Graph()
        graph.add_nodes_from([(self,{"initial":True})])
        state_id = 0
        frontier = deque([self])
        expanded = set()
        while frontier:
            state = frontier.popleft()
            if state in expanded:
                continue
            successors = list(state.successors)
            #════════════════════
            # expand
            graph.nodes[state]["id"] = state_id
            graph.nodes[state]["is_solution"] = state.is_solution
            state_id += 1
            for successor, action in successors:
                graph.add_edges_from([(state,successor)])
            expanded.add(state)
            #════════════════════
            # BFS
            for successor, action in successors:
                frontier.append(successor)
        return graph
    
    def search_bfs(self):
        """Returns the sequence of actions which solve the problem"""
        Board.ACTION_COUNT = 0
        root = Node(state=self, parent=None, action=None)
        tree_serializations = []
        frontier = deque([root])
        existing = set()
        count = 0
        while frontier:
            node = frontier.popleft()
            node.name = "X"
            for successor, action in node.state.successors:
                if successor.is_solution:
                    # Add the child for the sake of serialization and serialize
                    child = Node(state=successor, parent=node, action=action,name="S")
                    tree_serializations.append(root.serialize_for_emacs())                    
                    # From the sequence of actions and return it
                    actions = [action]
                    while node.action is not None:
                        actions.append(node.action)
                        node = node.parent
                    with open("serializations","w") as f:
                        f.write("(")
                        f.write("\n".join(tree_serializations))
                        f.write(")")
                    actions.reverse()
                    return actions
                else:
                    if successor not in existing:
                        child = Node(state=successor, parent=node, action=action,name="F")
                        tree_serializations.append(root.serialize_for_emacs())
                        frontier.append(child)
                        existing.add(successor)


    def search_best_first(self, ef):
        """EF is an "evaluation function". It accepts a Node as an argument and
        returns the priority of the node."""
        raise NotImplementedError

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

# For image output
class BoardImage:
    DIR = "images"
    BOARD = Image.open(os.path.join(DIR, "board.tiff"))
    SQUARE = Image.open(os.path.join(DIR, "square.tiff"))
    UNIT_LENGTH = SQUARE.width
    BLOCK_H2 = Image.open(os.path.join(DIR, "2x1.tiff"))
    BLOCK_H3 = Image.open(os.path.join(DIR, "3x1.tiff"))
    BLOCK_V2 = Image.open(os.path.join(DIR, "1x2.tiff"))
    BLOCK_V3 = Image.open(os.path.join(DIR, "1x3.tiff"))
    BLOCK_GOAL = Image.open(os.path.join(DIR, "goal.tiff"))
    TYPE_TO_IMG = {"H2": BLOCK_H2,
                   "H3": BLOCK_H3,
                   "G2": BLOCK_GOAL,
                   "V2": BLOCK_V2,
                   "V3": BLOCK_V3}

    @classmethod
    def create_board_image(cls):
        square = Image.open("square.jpg")
        unit_length = square.width
        side_length = unit_length * 6 - 5
        board = Image.new("RGB", (side_length,side_length))
        for row,col in itertools.product(range(6),range(6)):
            xy = cls.get_pixel_xy((col,row))
            board.paste(square, (x,y))
        board.save("board.tiff")

    @classmethod
    def get_pixel_xy(cls,xy):
        x,y = xy
        return ((cls.UNIT_LENGTH-1)*x,
                (cls.UNIT_LENGTH-1)*y)

    def __init__(self):
        self.board = self.BOARD.copy()
        
    def draw(self,type,xy):
        """Draw a block on SELF at XY. TYPE is a string of the form "OS", where
        the "O" is the orientation ("H" for horizontal, "V" for vertical and "G"
        for goal block) and the "S" is the size, one of "2" or "3". So a
        possible format string is "H3" or "V2" or "G2" (but not "G3" as the goal
        block is always size 2)."""
        orientation, size = type[0].upper(),int(type[1])
        x,y = self.get_pixel_xy(xy)
        block_img = self.TYPE_TO_IMG[type]
        if orientation in "HG":
            frame_width = self.UNIT_LENGTH*size-size+1
            frame_height = self.UNIT_LENGTH
        else:
            frame_width = self.UNIT_LENGTH
            frame_height = self.UNIT_LENGTH*size-size+1
        x_offset = (frame_width-block_img.width)//2
        y_offset = (frame_height-block_img.height)//2
        x += x_offset
        y += y_offset
        self.board.paste(block_img,(x,y))

def create_board():
    square = Image.open("images/square.jpg")
    unit_length = square.width
    side_length = unit_length * 6 - 5
    board = Image.new("RGB", (side_length,side_length))
    for row,col in itertools.product(range(6),range(6)):
        x = (unit_length-1)*row
        y = (unit_length-1)*col
        board.paste(square, (x,y))
    board.save("images/board.tiff")

#════════════════════════════════════════
# misc

def get_board():
    PATH = "board.txt"
    text = open(PATH).read()
    return Board.from_str(text)

def get_board_and_actions():
    board = get_board()
    actions = board.search_bfs()
    print(f"### The solution has {len(actions)} actions")
    print(f"### It took {Board.ACTION_COUNT} actions to find this solution")

def apply_actions(board, actions):
    for action in actions:
        board = board.apply_action(action)
    return board

def main_char(cols):
    board, actions = get_board_and_actions()
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

def main_image():
    board, actions = get_board_and_actions()
    images = [board.image()]
    for action in actions:
        board = board.apply_action(action)
        images.append(board.image())
    name_length = 4+len(str(len(images)-1))
    OUTPUT_DIR = "image_output"
    try: shutil.rmtree(OUTPUT_DIR)
    except FileNotFoundError: pass
    os.mkdir(OUTPUT_DIR)
    for i,image in enumerate(images):
        file_name = f"{i}.jpg".rjust(name_length,"0")
        path = os.path.join(OUTPUT_DIR,file_name)
        image.save(path,quality=100)

#════════════════════════════════════════
# graphing the state space

def get_graph():
    board = get_board()
    return board.state_space()

def graph_dot(graph):
    lines = ["graph {"]
    for node,attrs in graph.nodes.items():
        state_id = attrs["id"]
        if attrs.get("initial"):
            color = "red"
        elif attrs.get("is_solution"):
            color = "blue"
        else:
            color = "white"
        lines.append(f'    {state_id} [label="", color={color}];')
    for (v1,v2),attrs in graph.edges.items():
        id1, id2 = graph.nodes[v1]["id"], graph.nodes[v2]["id"]
        lines.append(f'    {id1}--{id2};')
    lines.append["}"]

def test_dot():
    board = get_board()
    graph = board.state_space()
    with open("state_space.dot", "w") as f:
        f.write(graph_dot(graph))

def test_org():
    board = get_board()
    graph = board.state_space()
    with open("state_space.org", "w") as f:
        f.write(graph_org_mode(graph))

def graph_org_mode(graph):
    lines = ["#+TODO: INITIAL SOLUTION"]
    for node,attrs in graph.nodes.items():
        state_id = attrs["id"]
        line = None
        if attrs.get("initial"):
            line = f"* INITIAL {state_id}"
        elif attrs.get("is_solution"):
            line = f"* SOLUTION {state_id}"
        else:
            line = f"* {state_id}"
        lines.append(line)
        for neighbor in graph.neighbors(node):
            neigh_id = graph.nodes[neighbor]["id"]
            lines.append(f"** [[{neigh_id}]]")
    return "\n".join(lines)

def graph_plt(graph):
    import matplotlib.pyplot as plt
    networkx.draw(graph)
    plt.show()
    
def graph_dot(graph):
    lines = ["graph {"]
    for node,attrs in graph.nodes.items():
        state_id = attrs["id"]
        if attrs.get("initial"):
            color = "red"
        elif attrs.get("is_solution"):
            color = "blue"
        else:
            color = "white"
        lines.append(f'    {state_id} [label="", color={color}];')
    for (v1,v2),attrs in graph.edges.items():
        id1, id2 = graph.nodes[v1]["id"], graph.nodes[v2]["id"]
        lines.append(f'    {id1}--{id2};')
    lines.append("}"); lines.append("")
    return "\n".join(lines)

#════════════════════════════════════════
# scratch scripts
def scratch():
    bimg = BoardImage()
    bimg.draw("V2",(5,3))
    bimg.board.show()

def scratch():
    board = get_board()
    img = board.image()
    img.show()
