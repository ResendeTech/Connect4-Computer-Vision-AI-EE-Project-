import numpy as np
from typing import List, Optional

class ScoringDeterministicAI:
    
    def __init__(self, piece=1, depth=1):
        self.piece = piece
        self.depth = depth
        self.move_history = []
    
    def get_move(self, board: np.ndarray) -> Optional[int]:
        valid_cols = self._get_valid_columns(board)
        
        if not valid_cols:
            return None
        
        move_scores = []
        for col in valid_cols:
            temp_board = board.copy()
            row = self._get_next_open_row(temp_board, col)
            if row is not None:
                temp_board[row][col] = self.piece
                score = self._score_positions(temp_board, self.piece)
                move_scores.append((col, score))
        
        if not move_scores:
            return valid_cols[0]
        
        move_scores.sort(key=lambda x: (-x[1], x[0]))
        best_move = move_scores[0][0]
        
        self.move_history.append(best_move)
        return best_move
    
    def _get_valid_columns(self, board: np.ndarray) -> List[int]:
        valid_cols = []
        for col in range(7):
            if board[5][col] == 0:
                valid_cols.append(col)
        return valid_cols
    
    def _get_next_open_row(self, board: np.ndarray, col: int) -> Optional[int]:
        for row in range(6):
            if board[row][col] == 0:
                return row
        return None
    
    def _evaluate_window(self, window: List[int], piece: int) -> int:
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
        score = 0
        rows, columns = 6, 7
        
        center_array = [int(i) for i in list(board[:, columns//2])]
        center_count = center_array.count(piece)
        score += center_count * 3
        
        for r in range(rows):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(columns - 3):
                window = row_array[c:c+4]
                score += self._evaluate_window(window, piece)

        for c in range(columns):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(rows - 3):
                window = col_array[r:r+4]
                score += self._evaluate_window(window, piece)

        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self._evaluate_window(window, piece)

        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self._evaluate_window(window, piece)

        return score
    
    def reset(self):
        self.move_history = []

class StrongDeterministicAI(ScoringDeterministicAI):
    def __init__(self, piece=1):
        super().__init__(piece=piece, depth=1)

