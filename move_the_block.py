import sys
import networkx as nx
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

class Board:
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

    @property
    def image(self):
        board_img = BoardImage()
        for (row,col),block in self.blocks:
            orient = "G" if block is self.goal_block else block.orient
            block_type = f"{orient}{block.size}"
            board_img.draw(block_type,(col,row))
        return board_img.board

    @property
    def state_space(self):
        try: return self._state_space
        except AttributeError: pass
        graph = nx.Graph()
        graph.add_node(self)
        graph.graph["initial"] = self
        board_id = 0
        frontier = deque([self])
        expanded = set()
        while frontier:
            board = frontier.popleft()
            if board in expanded:
                continue
            successors = list(board.successors)
            #════════════════════
            # expand
            attrs = graph.nodes[board]
            attrs["id"] = board_id; board_id += 1
            attrs["is_solution"] = board.is_solution
            for successor in successors:
                graph.add_edges_from([(board,successor)])
            expanded.add(board)
            #════════════════════
            # BFS
            for successor in successors:
                frontier.append(successor)
        self._state_space = graph
        return graph

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
        return [self.apply_action(action)
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

class Searcher:
    def __init__(self,board):
        self.board = board
        
    def bfs(self):
        """Returns a sequence of states beginning with SELF.BOARD and ending
        with a goal state, as well as the search tree as it was at the time the
        solution was found."""
        search_tree = nx.DiGraph()
        search_tree.add_node(self.board)
        frontier = deque([self.board])
        expanded = set()
        while frontier:
            board = frontier.popleft()
            if board in expanded:
                continue
            for successor in board.successors:
                if successor.is_solution:
                    search_tree.add_edge(board,successor)
                    path = [successor]
                    while board is not None:
                        path.append(board)
                        board = next(search_tree.predecessors(board),None)
                    path.reverse()
                    return path, search_tree
                elif (successor not in search_tree.nodes and
                      successor not in expanded):
                    search_tree.add_edge(board,successor)
                    frontier.append(successor)
            expanded.add(board)
        return None, search_tree

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

# misc
#════════════════════════════════════════

def get_board(name):
    path = os.path.join("boards",f"{name.lower()}.txt")
    text = open(path).read()
    return Board.from_str(text)

def find_solution(name):
    board = get_board(name)
    return Searcher(board).bfs()

def image_solution(name, output_dir):
    solution, search_tree = find_solution(name)
    images = [board.image for board in solution]
    name_length = 4+len(str(len(images)-1))
    print(f"### search_tree has {len(search_tree.nodes)} nodes")
    output_dir = output_dir or os.path.join("boards",f"{name}_solution")
    try: shutil.rmtree(output_dir)
    except FileNotFoundError: pass
    os.mkdir(output_dir)
    for i,image in enumerate(images):
        file_name = f"{i}.jpg".rjust(name_length,"0")
        path = os.path.join(output_dir,file_name)
        image.save(path,quality=100)

# graphing
#════════════════════════════════════════

def state_space_org_mode(board):
    graph = board.state_space
    lines = ["#+TODO: INITIAL SOLUTION"]
    for node,attrs in graph.nodes.items():
        board_id = attrs["id"]
        line = None
        if node is board:
            line = f"* INITIAL {board_id}"
        elif attrs.get("is_solution"):
            line = f"* SOLUTION {board_id}"
        else:
            line = f"* {board_id}"
        lines.append(line)
        for neighbor in graph.neighbors(node):
            neigh_id = graph.nodes[neighbor]["id"]
            lines.append(f"** [[{neigh_id}]]")
    return "\n".join(lines)

def state_space_plt(board):
    import matplotlib.pyplot as plt
    graph = board.state_space
    nx.draw(graph)
    plt.show()
    
def state_space_dot(board):
    graph = board.state_space
    lines = ["graph {"]
    for node,attrs in graph.nodes.items():
        board_id = attrs["id"]
        if node is board:
            color = "red"
        elif attrs.get("is_solution"):
            color = "blue"
        else:
            color = "white"
        lines.append(f'    {board_id} [label="", color={color}];')
    for (v1,v2),attrs in graph.edges.items():
        id1, id2 = graph.nodes[v1]["id"], graph.nodes[v2]["id"]
        lines.append(f'    {id1}--{id2};')
    lines.append("}"); lines.append("")
    return "\n".join(lines)

def search_tree_dot(board, path):
    _, search_tree = Searcher(board).bfs()
    lines = ["digraph {"]
    for (node,attrs),nid in zip(search_tree.nodes.items(),
                                itertools.count()):
        attrs["id"] = nid
        lines.append(f'    {nid} [label=""];')
    for n1, n2 in graph.edges:
        id1, id2 = graph.nodes[n1]["id"], graph.nodes[n2]["id"]
        lines.append(f'    {id1}->{id2};')
    lines.append("}"); lines.append("")
    with open(path,"w") as f:
        f.write("\n".join(lines))

def search_tree_plt(board):
    import matplotlib.pyplot as plt
    tree = Searcher(board).bfs()
    nx.draw(tree)
    plt.show()

# scratch scripts
#════════════════════════════════════════

def scratch():
    pass
