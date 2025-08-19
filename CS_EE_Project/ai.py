import random
import math

def evaluate_window(window, piece):
    score = 0
    opp_piece = 1
    if piece == 1:
        opp_piece = 2
    
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

def score_positions(board, piece):
    import game
    score = 0
    
    #Score centre column
    centre_array = [int(i) for i in list(board[:, game.columns//2])]
    centre_count = centre_array.count(piece)
    score += centre_count * 3
    
    
    #Horizontal win
    for r in range(game.rows):
        rows_array = [int(i) for i in list(board[r,:])]
        for c in range(game.columns - 3):
            window = rows_array[c:c+4]
            score += evaluate_window(window, piece)

    # Vertical win
    for c in range(game.columns):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(game.rows - 3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    # positive diagonal slope
    for r in range(game.rows - 3):
        for c in range(game.columns - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    # negative diagonal slope
    for r in range(game.rows - 3):
        for c in range(game.columns - 3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    import game
    return game.win_checks(board, 1) or game.win_checks(board, 2) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximaxingplayer):
    import game
    
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    
    # Handle case where no valid moves exist
    if not valid_locations:
        return (None, 0)
        
    # Base cases - terminal states or depth limit reached
    if depth == 0 or is_terminal:
        if is_terminal:
            if game.win_checks(board, 2):  # AI wins
                return (None, 1000000000)
            elif game.win_checks(board, 1):  # Human wins
                return (None, -1000000000)
            else:  # Draw scenario, no plays left
                return (None, 0)
        else:  # Depth is zero but game continues
            return (None, score_positions(board, 2))
        
    if maximaxingplayer:  # AI maximizing player
        value = -math.inf
        best_col = valid_locations[0]  # Initialize with first valid column
        
        for col in valid_locations:
            row = game.for_next_open_row(board, col)
            if row is None:  # Column is full
                continue
            board_copy = board.copy()
            game.drop_piece(board_copy, row, col, 2)
            new_score = minimax(board_copy, depth-1, alpha, beta, False)[1]
            
            if new_score > value:
                value = new_score
                best_col = col
                
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return (best_col, value)
    
    else:  # Human minimizing player
        value = math.inf
        best_col = valid_locations[0]  # Initialize with first valid column
        
        for col in valid_locations:
            row = game.for_next_open_row(board, col)
            if row is None:  # Column is full
                continue
            board_copy = board.copy()
            game.drop_piece(board_copy, row, col, 1)
            new_score = minimax(board_copy, depth-1, alpha, beta, True)[1]
            
            if new_score < value:
                value = new_score
                best_col = col
                
            beta = min(beta, value)
            if alpha >= beta:
                break
        return (best_col, value)
                
def get_valid_locations(board):
    import game
    valid_locations = []
    for col in range(game.columns):
        if game.is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations
            
def pick_best_move(board, piece):
    import game
    valid_locations = get_valid_locations(board)
    
    if not valid_locations:
        return None
        
    # Try to win immediately
    for col in valid_locations:
        temp_board = board.copy()
        row = game.for_next_open_row(temp_board, col)
        if row is not None:
            game.drop_piece(temp_board, row, col, piece)
            if game.win_checks(temp_board, piece):
                return col
    
    # Block opponent from winning
    opp_piece = 1 if piece == 2 else 2
    for col in valid_locations:
        temp_board = board.copy()
        row = game.for_next_open_row(temp_board, col)
        if row is not None:
            game.drop_piece(temp_board, row, col, opp_piece)
            if game.win_checks(temp_board, opp_piece):
                return col
    
    # Use scoring heuristic for remaining moves
    best_score = -10000
    best_col = valid_locations[0]  # Initialize with first valid option
    
    for col in valid_locations:
        temp_board = board.copy()
        row = game.for_next_open_row(temp_board, col)
        if row is not None:
            game.drop_piece(temp_board, row, col, piece)
            score = score_positions(temp_board, piece)
            if score > best_score:
                best_score = score
                best_col = col
    
    return best_col
    