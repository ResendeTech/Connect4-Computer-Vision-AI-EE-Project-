import numpy as np

class DeterministicAI:
    """
    A deterministic AI that always plays the same moves given the same board state.
    This ensures reproducible results for data collection.
    """
    
    def __init__(self, strategy="left_to_right"):
        self.strategy = strategy
        self.move_history = []  # Track moves for reproducibility
    
    def get_move(self, board):
        """
        Get deterministic move based on strategy
        """
        valid_cols = []
        for col in range(7):
            if board[5][col] == 0:  # Top row is empty
                valid_cols.append(col)
        
        if not valid_cols:
            return None
            
        if self.strategy == "left_to_right":
            # Always pick leftmost available column
            move = valid_cols[0]
        elif self.strategy == "center_out":
            # Prefer center, then spiral outward
            preference_order = [3, 2, 4, 1, 5, 0, 6]
            for col in preference_order:
                if col in valid_cols:
                    move = col
                    break
            else:
                move = valid_cols[0]
        elif self.strategy == "alternating":
            # Alternate between left and right sides
            if len(self.move_history) % 2 == 0:
                move = valid_cols[0]  # Leftmost
            else:
                move = valid_cols[-1]  # Rightmost
        else:  # default to left_to_right
            move = valid_cols[0]
            
        self.move_history.append(move)
        return move
    
    def reset(self):
        """Reset move history for new game"""
        self.move_history = []
