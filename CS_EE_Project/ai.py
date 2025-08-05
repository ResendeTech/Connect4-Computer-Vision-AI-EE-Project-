import random

def score_positions(board, piece):
    import game
    score = 0
    
    #Horizontal win
    for r in range(game.rows):
        rows_array = [int(i) for i in list(board[r,:])]
        for c in range(game.columns - 3):
            window = rows_array[c:c+4]
            if window.count(piece) == 4:
                score += 100
            elif window.count(piece) == 3 and window.count(0) == 1:
                score += 10
                
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
    best_score = -10
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
    