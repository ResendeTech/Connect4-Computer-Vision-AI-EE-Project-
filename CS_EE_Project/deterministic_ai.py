import numpy as np

class DeterministicAI:
    def __init__(self, strategy="left_to_right"):
        self.strategy = strategy
        self.move_history = []
    
    def get_move(self, board):
        valid_cols = []
        for col in range(7):
            if board[5][col] == 0:
                valid_cols.append(col)
        
        if not valid_cols:
            return None
            
        if self.strategy == "left_to_right":
            move = valid_cols[0]
        elif self.strategy == "center_out":
            preference_order = [3, 2, 4, 1, 5, 0, 6]
            for col in preference_order:
                if col in valid_cols:
                    move = col
                    break
            else:
                move = valid_cols[0]
        elif self.strategy == "alternating":
            if len(self.move_history) % 2 == 0:
                move = valid_cols[0]
            else:
                move = valid_cols[-1]
        else:
            move = valid_cols[0]
            
        self.move_history.append(move)
        return move
    
    def reset(self):
        self.move_history = []
