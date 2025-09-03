import numpy as np
import math


rows = 6
columns = 7

def create_board():
    board = np.zeros((rows, columns))
    return board


def is_valid_location(board, col):
    # Add bounds checking and debug info
    if col < 0 or col >= columns:
        return False
    
    # Debug: check board type and shape
    if not isinstance(board, np.ndarray):
        print(f"Error: board is not numpy array, it's {type(board)}")
        return False
    
    if board.shape != (rows, columns):
        print(f"Error: board shape is {board.shape}, expected ({rows}, {columns})")
        return False
        
    # Check if top cell in column is empty
    return board[rows-1][col] == 0

def for_next_open_row(board, col):
    for r in range(rows):
        if board[r][col] == 0:
            return r
    # Return None if column is full
    return None

def drop_piece(board, row, col, piece):
    board[row][col] = piece
    
def print_board(board):
    print(np.flip(board, 0))

def win_checks(board, piece):
    for c in range(columns-3):
        for r in range(rows):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True


    for c in range(columns):
        for r in range(rows - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True
            
    for c in range(columns - 3):
        for r in range(rows - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True
            
    for c in range(columns - 3):
        for r in range(3, rows):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True
    
    # Explicitly return False when no win is found
    return False
            
            
if __name__ == "__main__":
    import vision
    board = vision.get_board_from_image()

    if board is None:
        print("Could not get board from vision, creating empty board")
        board = create_board()
        turn = 0
    else:
        print("Board loaded from vision:")
        print(f"Board type: {type(board)}")
        print(f"Board shape: {board.shape if hasattr(board, 'shape') else 'No shape attribute'}")
        print(f"Board content: {board}")
        print_board(board)

        turn_result = vision.decide_turn(board)

        if turn_result == "invalid board":
            print("Invalid board detected, starting new game")
            board = create_board()
            turn = 0  # Start with player 1
        elif turn_result == 0:  # Red's turn
            turn = 0
        elif turn_result == 1:  # Yellow's turn  
            turn = 1
        else:
            print(f"Unexpected turn result: {turn_result}, starting with player 1")
            turn = 0

    game_over = False

    

    while not game_over:

        if turn == 0:
            try:
                col = int(input("Player 1:")) - 1
            except ValueError:
                print("Please enter a valid number!")
                continue
            
            if col is None or not is_valid_location(board, col):
                print("Invalid move! Try again.")
                continue
                
            row = for_next_open_row(board, col)
            if row is None:
                print("Column is full! Try again.")
                continue
                
            drop_piece(board, row, col, 1)
            
            if win_checks(board, 1) == True:
                print("player 1 wins") 
                game_over = True
                
            turn = 1
            print_board(board)
                
                
        elif turn == 1 and not game_over:
            import ai
            #col = ai.pick_best_move(board, 2)
            try:
                # col, minimax_score = ai.minimax(board, 6, -math.inf, math.inf, True)
                # print("the try ai part is working")
                result = ai.minimax(board, 6, -math.inf, math.inf, True)
                print(f"Minimax returned: {result}")  # Debug: see what minimax actually returns
                col, minimax_score = result
                print(f"Column: {col}, Score: {minimax_score}")
                print("the try ai part is working")
            except Exception as e:
                print(f"AI error: {e}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()  # This will show the full error trace
                col = None
            
            # If AI fails, try simpler approach or pick first valid move
            if col is None:
                print("AI minimax failed, trying simple approach...")
                valid_moves = ai.get_valid_locations(board)
                if valid_moves:
                    col = ai.pick_best_move(board, 2)  # Use simpler heuristic
                else:
                    print("No valid moves available - game should be over")
                    game_over = True
                    continue
                    
            # Double-check the move is valid
            if col is not None and not is_valid_location(board, col):
                print("AI move invalid, picking first available column...")
                valid_moves = ai.get_valid_locations(board)
                if valid_moves:
                    col = valid_moves[0]  # Pick first valid column
                else:
                    print("No valid moves available - game over")
                    game_over = True
                    continue
            
            if col is not None:
                print(f"AI chooses column {col + 1}")
                
                row = for_next_open_row(board, col)
                if row is None:
                    print("Column is full, picking another...")
                    valid_moves = ai.get_valid_locations(board)
                    if valid_moves:
                        col = valid_moves[0]
                        row = for_next_open_row(board, col)
                        print(f"AI switches to column {col + 1}")
                    else:
                        print("Board is full - game over")
                        game_over = True
                        continue
                
                # Make the move if we have both valid column and row
                if row is not None and col is not None:
                    drop_piece(board, row, col, 2)
                    
                    if win_checks(board, 2) == True:
                        print("player 2 wins") 
                        game_over = True
                        
                    turn = 0
                    print("----AI'S TURN----")
                    print_board(board)
                else:
                    print("Could not find valid move - game over")
                    game_over = True

