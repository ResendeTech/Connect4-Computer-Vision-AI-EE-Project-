import random

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
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = game.for_next_open_row(board, col)
        temp_board = board.copy()
        game.drop_piece(temp_board, row, col, piece)
        score = score_positions(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col
    