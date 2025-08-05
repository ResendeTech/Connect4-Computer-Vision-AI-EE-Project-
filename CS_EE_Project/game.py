import numpy as np


rows = 6
columns = 7

def create_board():
    board = np.zeros((rows, columns))
    return board


def is_valid_location(board, col):
    return board[rows-1][col] == 0

def for_next_open_row(board, col):
    for r in range(rows):
        if board[r][col] == 0:
            return r

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
if __name__ == "__main__":
    board = create_board()
    print_board(board)
    game_over = False
    turn = 0

    while not game_over:
        
        if turn == 0:
            col = int(input("Player 1:")) - 1
            
            if is_valid_location(board, col):
                row = for_next_open_row(board, col)
                drop_piece(board, row, col, 1)
                
                if win_checks(board, 1) == True:
                    print("player 1 wins") 
                    game_over = True
                    
                else:   
                    turn += 1
                    turn = turn % 2
                print_board(board)
                
                
        elif turn == 1 and not game_over:
            import ai
            col = ai.pick_best_move(board, 2)
            
            if is_valid_location(board, col):
                row = for_next_open_row(board, col)
                drop_piece(board, row, col, 2)
                
                if win_checks(board, 2) == True:
                    print("player 2 wins") 
                    game_over = True
                    
                else:
                    turn += 1
                    turn = turn % 2
                print_board(board)

