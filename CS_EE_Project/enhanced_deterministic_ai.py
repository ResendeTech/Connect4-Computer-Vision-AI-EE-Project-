"""
Enhanced Deterministic AI for Connect 4 Experiments
===================================================

This module provides deterministic AI players that always make the same moves
given the same board state, ensuring reproducible experimental results.

Includes both simple rule-based and sophisticated scoring-based deterministic AIs.
"""

import numpy as np
from typing import List, Optional

class SimpleDeterministicAI:
    """Simple rule-based deterministic AI - weaker opponent"""
    
    def __init__(self, strategy="center_out"):
        self.strategy = strategy
        self.move_history = []
    
    def get_move(self, board: np.ndarray) -> Optional[int]:
        """Get deterministic move based on simple strategy"""
        valid_cols = self._get_valid_columns(board)
        
        if not valid_cols:
            return None
            
        if self.strategy == "left_to_right":
            move = valid_cols[0]
        elif self.strategy == "center_out":
            # Prefer center columns: 3, 2, 4, 1, 5, 0, 6
            preference_order = [3, 2, 4, 1, 5, 0, 6]
            for col in preference_order:
                if col in valid_cols:
                    move = col
                    break
            else:
                move = valid_cols[0]
        elif self.strategy == "alternating":
            if len(self.move_history) % 2 == 0:
                move = valid_cols[0]  # Leftmost
            else:
                move = valid_cols[-1]  # Rightmost
        else:
            move = valid_cols[0]  # Default to leftmost
            
        self.move_history.append(move)
        return move
    
    def _get_valid_columns(self, board: np.ndarray) -> List[int]:
        """Get list of columns that are not full"""
        valid_cols = []
        for col in range(7):
            if board[5][col] == 0:  # Top row is empty
                valid_cols.append(col)
        return valid_cols
    
    def reset(self):
        """Reset for new game"""
        self.move_history = []

class ScoringDeterministicAI:
    """
    Deterministic AI using your scoring function - stronger opponent
    Always picks the highest scoring move (deterministic)
    """
    
    def __init__(self, piece=1, depth=1):
        self.piece = piece  # Which piece this AI plays (1 or 2)
        self.depth = depth  # How many moves to look ahead
        self.move_history = []
    
    def get_move(self, board: np.ndarray) -> Optional[int]:
        """Get move using deterministic scoring evaluation"""
        valid_cols = self._get_valid_columns(board)
        
        if not valid_cols:
            return None
        
        # Evaluate each possible move
        move_scores = []
        for col in valid_cols:
            # Simulate the move
            temp_board = board.copy()
            row = self._get_next_open_row(temp_board, col)
            if row is not None:
                temp_board[row][col] = self.piece
                score = self._score_positions(temp_board, self.piece)
                move_scores.append((col, score))
        
        if not move_scores:
            return valid_cols[0]  # Fallback
        
        # Sort by score (descending) and pick the best
        # If multiple moves have same score, pick the leftmost (deterministic)
        move_scores.sort(key=lambda x: (-x[1], x[0]))
        best_move = move_scores[0][0]
        
        self.move_history.append(best_move)
        return best_move
    
    def _get_valid_columns(self, board: np.ndarray) -> List[int]:
        """Get list of columns that are not full"""
        valid_cols = []
        for col in range(7):
            if board[5][col] == 0:  # Top row is empty
                valid_cols.append(col)
        return valid_cols
    
    def _get_next_open_row(self, board: np.ndarray, col: int) -> Optional[int]:
        """Find the next open row in a column"""
        for row in range(6):
            if board[row][col] == 0:
                return row
        return None
    
    def _evaluate_window(self, window: List[int], piece: int) -> int:
        """
        Your scoring function - evaluates a 4-piece window
        """
        score = 0
        opp_piece = 1 if piece == 2 else 2
        
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 10
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 5

        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 80
        elif window.count(opp_piece) == 2 and window.count(0) == 2:
            score -= 7

        return score
    
    def _score_positions(self, board: np.ndarray, piece: int) -> int:
        """
        Your board scoring function - evaluates entire board position
        """
        score = 0
        rows, columns = 6, 7
        
        # Score center column
        center_array = [int(i) for i in list(board[:, columns//2])]
        center_count = center_array.count(piece)
        score += center_count * 3
        
        # Horizontal scoring
        for r in range(rows):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(columns - 3):
                window = row_array[c:c+4]
                score += self._evaluate_window(window, piece)

        # Vertical scoring
        for c in range(columns):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(rows - 3):
                window = col_array[r:r+4]
                score += self._evaluate_window(window, piece)

        # Positive diagonal scoring
        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self._evaluate_window(window, piece)

        # Negative diagonal scoring
        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self._evaluate_window(window, piece)

        return score
    
    def reset(self):
        """Reset for new game"""
        self.move_history = []

# Compatibility classes for different AI types
class WeakDeterministicAI(SimpleDeterministicAI):
    """Weak deterministic AI - for testing against strong distorted AI"""
    def __init__(self):
        super().__init__(strategy="left_to_right")

class ModerateDeterministicAI(SimpleDeterministicAI):
    """Moderate deterministic AI - balanced opponent"""
    def __init__(self):
        super().__init__(strategy="center_out")

class StrongDeterministicAI(ScoringDeterministicAI):
    """Strong deterministic AI - uses your scoring algorithm"""
    def __init__(self, piece=1):
        super().__init__(piece=piece, depth=1)

def create_deterministic_ai(strength: str = "moderate", piece: int = 1):
    """
    Factory function to create deterministic AI of specified strength
    
    Args:
        strength: "weak", "moderate", or "strong"  
        piece: 1 or 2 (which piece this AI plays)
    """
    if strength == "weak":
        return WeakDeterministicAI()
    elif strength == "moderate":
        return ModerateDeterministicAI()
    elif strength == "strong":
        return StrongDeterministicAI(piece=piece)
    else:
        return ModerateDeterministicAI()  # Default

# Test the deterministic properties
def test_determinism():
    """Test that the AIs are truly deterministic"""
    print("🧪 Testing AI Determinism...")
    
    # Create a test board
    test_board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [2, 2, 1, 2, 0, 0, 0]
    ])
    
    # Test each AI type
    ai_types = [
        ("Weak", WeakDeterministicAI()),
        ("Moderate", ModerateDeterministicAI()),
        ("Strong", StrongDeterministicAI(piece=1))
    ]
    
    for name, ai in ai_types:
        moves = []
        for trial in range(5):  # Test 5 times
            ai.reset()  # Reset AI state
            move = ai.get_move(test_board)
            moves.append(move)
        
        is_deterministic = all(m == moves[0] for m in moves)
        print(f"{name} AI: Moves = {moves}, Deterministic = {is_deterministic}")

if __name__ == "__main__":
    test_determinism()
