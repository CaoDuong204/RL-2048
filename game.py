import numpy as np
import matplotlib.pyplot as plt
import pickle

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

base_dir = '.'


# =============================================================================
# Standalone Afterstate Computation (pure numpy, no Game instance needed)
# =============================================================================
def _shift_row(row):
    """Shift non-zero elements to the left."""
    result = np.zeros_like(row)
    idx = 0
    for v in row:
        if v != 0:
            result[idx] = v
            idx += 1
    return result


def _merge_left(board):
    """Apply left-merge to a 4x4 board. Returns (merged_board, log2_reward)."""
    reward = 0.0
    result = np.empty_like(board)
    for i in range(board.shape[0]):
        shifted = _shift_row(board[i])
        for j in range(len(shifted) - 1):
            if shifted[j] != 0 and shifted[j] == shifted[j + 1]:
                shifted[j] *= 2
                shifted[j + 1] = 0
                reward += np.log2(shifted[j])
        result[i] = _shift_row(shifted)
    return result, reward


def compute_afterstate(board_2d, action):
    """
    Compute afterstate for a given 4x4 board and action.
    Pure numpy — no Game instance needed. Used during batch learning.
    
    Returns: (afterstate_2d, merge_reward, is_valid)
    """
    b = board_2d.copy()
    if action == ACTION_LEFT:
        b, reward = _merge_left(b)
    elif action == ACTION_RIGHT:
        b = np.flip(b, axis=1).copy()
        b, reward = _merge_left(b)
        b = np.flip(b, axis=1).copy()
    elif action == ACTION_UP:
        b = np.flip(np.transpose(b), axis=0).copy()
        b, reward = _merge_left(b)
        b = np.transpose(np.flip(b, axis=0)).copy()
    elif action == ACTION_DOWN:
        b = np.flip(np.transpose(b), axis=1).copy()
        b, reward = _merge_left(b)
        b = np.transpose(np.flip(b, axis=1)).copy()
    else:
        return board_2d.copy(), 0.0, False
    is_valid = not np.array_equal(board_2d, b)
    return b, reward, is_valid


def decode_onehot_to_board(encoded):
    """Decode (18, 4, 4) one-hot encoded state back to (4, 4) board."""
    indices = np.argmax(encoded, axis=0)  # (4, 4) log2 values
    board = np.power(2.0, indices)
    board[indices == 0] = 0  # index 0 = empty cell
    return board


# =============================================================================
# State Transformation (Moved from training_dqn.py to avoid circular import)
# =============================================================================
def transform_state_cnn(state):
    """Transform to CNN format: (18, 4, 4). Preserves 2D spatial structure."""
    board = np.reshape(state, (4, 4))
    safe_board = np.where(board == 0, 1, board)
    indices = np.where(board == 0, 0, np.log2(safe_board).astype(np.int32))
    out = np.zeros((18, 4, 4), dtype=np.float32)
    r, c = np.mgrid[0:4, 0:4]
    out[indices, r, c] = 1.0
    return out


def transform_state_flat(state, mode='one_hot'):
    """Transform to flat format for MLP."""
    state = np.reshape(state, -1).copy()
    state[state == 0] = 1
    if mode == 'log2':
        return (np.log2(state) / 17.0).astype(np.float32)
    else:
        state = np.log2(state).astype(int)
        return np.reshape(np.eye(18, dtype=np.float32)[state], -1)

class Game():
    """ 2048 game environment"""
    def __init__(self, size = 4, seed = 42, negative_reward = -10, reward_mode='log2', cell_move_penalty = 0.1):
        self.board_dim = size               # board dimension
        self.state_size = size * size       # total number of cells
        self.action_size = 4                # number of available actions
        np.random.seed(seed)
        self.best_game_history = []
        self.negative_reward = negative_reward
        self.reward_mode = reward_mode
        self.cell_move_penalty = cell_move_penalty

    def save_best_game_history(self):
        self.best_game_history = self.history.copy()
        with open(base_dir+'/best_game_hist.pkl', 'wb') as f:
            pickle.dump(self.best_game_history, f)
        
    def reset(self, init_fields = 2, step_penalty = 0, bootstrapping = False):
        """ Initializes the board
        
        Params
        ======
            init_fields (int): how many fields to fill initially
            step_penalty (int): the cost of an action
            bootstrapping (bool): whether to create a new (initial) board or simulate some intermediate game state
        """
        self.game_board = np.zeros((self.board_dim, self.board_dim))
        
        if not bootstrapping:
            for i in range(init_fields):
                self.fill_random_empty_cell()
        else:
            self.random_board()
            
        self.score = np.sum(self.game_board)
        self.reward = 0
        self.current_cell_move_penalty = 0
        self.done = False
        self.steps = 0
        self.rewards_list = []
        self.scores_list = []
        self.step_penalty = step_penalty
        self.history = []
        
        self.history.append({
            'action': -1,
            'new_board': self.game_board.copy(),  
            'old_board': None,
            'score': self.score,
            'reward': self.reward
        })
        
    def shift(self, board):
        """ Shifts all cells to the left and gathers penalties if needed """
        shifted_board = np.empty((board.shape[0], board.shape[1]))
        for i, row in enumerate(board):
            shifted = np.zeros(len(row))
            idx = 0
            for iv, v in enumerate(row):
                if v != 0:
                    shifted[idx] = v
                    if iv != idx:
                        self.current_cell_move_penalty += self.cell_move_penalty * v
                    idx += 1
            shifted_board[i] = shifted
        return shifted_board
        
    def calc_board(self, board):
        """ Calculate all cell mergers and return the new state of the board"""
        
        self.reward = 0
        self.current_cell_move_penalty = 0
        
        shifted_board = self.shift(board)
        
        merged_board = np.empty((shifted_board.shape[0], shifted_board.shape[1]))
        for idx, row in enumerate(shifted_board):
            for i in range(len(row)-1):
                if row[i] != 0 and row[i] == row[i+1]:
                    
                    row[i] = row[i] * 2
                    row[i+1] = 0
                    if self.reward_mode == 'log2':
                        self.reward += np.log2(row[i])
                    else:
                        self.reward += row[i]

            merged_board[idx] = row
        merged_board = self.shift(merged_board)
        
        return merged_board

    def current_state(self):
        """ Returns a flattened array of board cell values """
        return np.reshape(self.game_board.copy(), -1)
    
    def step(self, action):
        """ Applies the selected action to the board """
        old_board = self.game_board.copy()
        temp_board = self.game_board.copy()
        
        # Here we flip/transpose the board depending on the action in order to unify the calculation
        if action == ACTION_LEFT:
            temp_board = self.calc_board(temp_board)

        elif action == ACTION_RIGHT:
            temp_board = np.flip(self.calc_board(np.flip(temp_board, axis=1)), axis=1)

        elif action == ACTION_UP:
            temp_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=0)), axis=0))

        elif action == ACTION_DOWN:
            temp_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=1)), axis=1))
        else: # just in case it happens
            return (self.game_board, 0, self.done)
        
        if not np.array_equal(self.game_board, temp_board):
            # Fill an empty cell with a new value
            self.game_board = temp_board.copy()
            self.fill_random_empty_cell()

            # Reward is the sum of the merged cells minus step cost
            self.reward = self.reward - self.current_cell_move_penalty
            
            self.score = np.sum(self.game_board)
            self.done = self.check_is_done()
            self.moved = True
        else:
            self.reward = self.negative_reward
            self.moved = False
        self.steps += 1
        self.rewards_list.append(self.reward)
        
        # Save the new state
        self.history.append({
            'action': action,
            'old_board': old_board,
            'new_board': self.game_board.copy(),  
            'score': self.score,
            'reward': self.reward
        })

        return (self.game_board, self.reward, self.done)

    def virtual_step(self, action):
        """
        Simulate a step without modifying the internal state of the game.
        Returns: (new_game_board, reward, done)
        """
        saved_reward = self.reward
        saved_penalty = self.current_cell_move_penalty
        temp_board = self.game_board.copy()

        if action == ACTION_LEFT:
            new_game_board = self.calc_board(temp_board)
        elif action == ACTION_RIGHT:
            new_game_board = np.flip(self.calc_board(np.flip(temp_board, axis=1)), axis=1)
        elif action == ACTION_UP:
            new_game_board = np.transpose(
                np.flip(self.calc_board(np.flip(np.transpose(temp_board), axis=0)), axis=0))
        elif action == ACTION_DOWN:
            new_game_board = np.transpose(
                np.flip(self.calc_board(np.flip(np.transpose(temp_board), axis=1)), axis=1))
        else: # just in case it happens
            return (self.game_board.copy(), 0, self.done)
        
        step_reward = self.reward - self.step_penalty
        
        # Restore state mutated by calc_board
        self.reward = saved_reward
        self.current_cell_move_penalty = saved_penalty
        
        is_done = self.check_is_done(new_game_board)
        return (new_game_board, step_reward, is_done)

    def get_afterstate(self, action):
        """
        Compute afterstate WITHOUT modifying game state.
        Returns (afterstate_board, merge_reward, is_valid).
        
        afterstate = board AFTER merge, BEFORE random tile spawn.
        This is deterministic — no randomness involved.
        """
        saved_reward = self.reward
        saved_penalty = self.current_cell_move_penalty
        temp_board = self.game_board.copy()

        if action == ACTION_LEFT:
            result = self.calc_board(temp_board)
        elif action == ACTION_RIGHT:
            result = np.flip(self.calc_board(np.flip(temp_board, axis=1)), axis=1)
        elif action == ACTION_UP:
            result = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=0)), axis=0))
        elif action == ACTION_DOWN:
            result = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=1)), axis=1))
        else:
            return self.game_board.copy(), 0.0, False

        merge_reward = self.reward
        is_valid = not np.array_equal(self.game_board, result)

        # Restore game state (calc_board modifies self.reward/penalty)
        self.reward = saved_reward
        self.current_cell_move_penalty = saved_penalty

        return result, merge_reward, is_valid


    def check_is_done(self, board = None):
        """ Check if the game is over """
    
        if board is None:
            board = self.game_board
    
        # If there are at least one cell with 0, then the game is not over
        if not np.all(board):
            return False
        
        # If all cells are filled, we need to check if there are any possible moves
        else:
            # Check if there are any equal adjacent cells across horisontal and vertical axes
            for row in board:
                for cell in range(len(row) - 1):
                    if row[cell] == row[cell+1]:
                        return False
            
            for row in np.transpose(board):
                for cell in range(len(row) - 1):
                    if row[cell] == row[cell+1]:
                        return False
            
            # There are no equal adjacent cells, the game is over
            return True
    
    def print_board(self, transpose = False):
        """ Deprecated """
        if not transpose:
            print(self.game_board)
        else:
            print(np.transpose(self.game_board))
    
    def fill_random_empty_cell(self, playing=True):
        """ Finds an empty cell and fills it with 2 or 4 with 90/10% probability respectively (as per game rules on Wikipedia) """
        
        # If all cells are filled, there is no place to put a new value, just pass
        if np.all(self.game_board):
            return
        
        # Pick the cell
        x = np.random.randint(self.board_dim)
        y = np.random.randint(self.board_dim)
        
        # Check if it is empty, otherwise pick a new one
        while self.game_board[x, y] != 0:
            x = np.random.randint(self.board_dim)
            y = np.random.randint(self.board_dim)
        
        # If it is a regular game, only values 2 and 4 are allowed
        if playing:
            self.game_board[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])
        else:
            # Otherwise it is a boostrapping game, then any values are allowed with certain probability
            self.game_board[x, y] = np.random.choice([2**i for i in range(1, 17)], p=np.linspace(1, 0.001, 16)/np.sum(np.linspace(1, 0.001, 16)))
        
    def draw_board(self, board = None, title = 'Current game'):
        """ Draws a colored game board """
        cell_colors = {
            0: '#FFFFFF',
            2: '#EEE4DA',
            4: '#ECE0C8',
            8: '#ECB280',
            16:'#EC8D53',
            32:'#F57C5F',
            64:'#E95937',
            128:'#F3D96B',
            256:'#F2D04A',
            512:'#E5BF2E',
            1024:'#E2B814',
            2048:'#EBC502',
            4096:'#00A2D8',
            8192:'#9ED682',
            16384:'#9ED682',
            32768:'#9ED682',
            65536:'#9ED682',
            131072:'#9ED682',
        }

        if board is None:
            board = self.game_board
        
        ncols = self.board_dim
        nrows = self.board_dim

        # create the plots
        fig = plt.figure(figsize=(3,3))
        plt.suptitle(title)
        axes = [ fig.add_subplot(nrows, ncols, r * ncols + c) for r in range(0, nrows) for c in range(1, ncols+1) ]

        # add some data
        v = np.reshape(board, -1)
        for i, ax in enumerate(axes):
            ax.text(0.5, 0.5, str(int(v[i])), horizontalalignment='center', verticalalignment='center')
            ax.set_facecolor(cell_colors[int(v[i])])

        # remove the x and y ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()
        
    def random_board(self):
        """ Creates a randomly filled board for bootstrapping """
        
        # Define how many cells we want to fill
        num_filled_cells = np.random.randint(12) + 4
        
        # Fill these cells
        for i in range(num_filled_cells):
            self.fill_random_empty_cell(playing=False)