import numpy as np
import random
import csv
import time
import math
from datetime import datetime
import game
import ai
from deterministic_ai import DeterministicAI
from enhanced_deterministic_ai import StrongDeterministicAI

class QuickExperiment:
    def __init__(self):
        self.results = []
    
    def create_test_boards(self):
        boards = []
        boards.append(np.zeros((6, 7)))
        
        for i in range(5):
            board = np.zeros((6, 7))
            moves = random.randint(2, 6)
            for move in range(moves):
                col = random.randint(0, 6)
                for row in range(6):
                    if board[row][col] == 0:
                        board[row][col] = 1 if move % 2 == 0 else 2
                        break
            boards.append(board.copy())
        
        for i in range(10):
            board = np.zeros((6, 7))
            moves = random.randint(8, 18)
            for move in range(moves):
                valid_cols = [c for c in range(7) if board[5][c] == 0]
                if not valid_cols:
                    break
                col = random.choice(valid_cols)
                for row in range(6):
                    if board[row][col] == 0:
                        board[row][col] = 1 if move % 2 == 0 else 2
                        break
            boards.append(board.copy())
        
        near_win_patterns = [
            [(0,0,1), (0,1,1), (0,2,1)],
            [(0,3,1), (1,3,1), (2,3,1)],
            [(0,2,1), (0,3,1), (0,4,2), (1,2,2), (1,3,2)],
        ]
        
        for pattern in near_win_patterns:
            for i in range(3):
                board = np.zeros((6, 7))
                for row, col, piece in pattern:
                    if row < 6 and col < 7:
                        board[row][col] = piece
                for _ in range(random.randint(3, 8)):
                    valid_cols = [c for c in range(7) if board[5][c] == 0]
                    if valid_cols:
                        col = random.choice(valid_cols)
                        for row in range(6):
                            if board[row][col] == 0:
                                board[row][col] = random.choice([1, 2])
                                break
                boards.append(board.copy())
        
        while len(boards) < 30:
            board = self.create_random_board(random.randint(5, 20))
            boards.append(board)
        
        return boards[:30]
    
    def create_random_board(self, num_moves):
        board = np.zeros((6, 7))
        for move in range(num_moves):
            valid_cols = [c for c in range(7) if board[5][c] == 0]
            if not valid_cols:
                break
            col = random.choice(valid_cols)
            for row in range(6):
                if board[row][col] == 0:
                    board[row][col] = 1 if move % 2 == 0 else 2
                    break
        return board
    
    def apply_distortions(self, board, blur_rate, misclass_rate):
        distorted = board.copy()
        
        for r in range(6):
            for c in range(7):
                if board[r][c] != 0:
                    if random.random() < blur_rate:
                        distorted[r][c] = 0
                    elif random.random() < misclass_rate:
                        distorted[r][c] = 2 if board[r][c] == 1 else 1
        
        return distorted
    
    def play_single_game(self, perfect_board, distorted_board, starting_player=0):
        game_board = perfect_board.copy()
        distorted_view = distorted_board.copy()
        
        moves_made = 0
        illegal_moves = 0
        minimax_ai_wins = 0
        deterministic_ai_wins = 0
        move_times = []
        optimal_moves = 0
        total_minimax_moves = 0
        
        deterministic_ai = DeterministicAI(strategy="center_out")
        deterministic_ai.reset()
        
        oracle_ai = StrongDeterministicAI(piece=2)
        oracle_ai.reset()
        
        turn = starting_player
        max_moves = 42
        
        for move_num in range(max_moves):
            if turn == 0:
                try:
                    col = deterministic_ai.get_move(game_board)
                    if col is not None and game.is_valid_location(game_board, col):
                        row = game.for_next_open_row(game_board, col)
                        if row is not None:
                            game.drop_piece(game_board, row, col, 1)
                            distorted_row = game.for_next_open_row(distorted_view, col)
                            if distorted_row is not None:
                                game.drop_piece(distorted_view, distorted_row, col, 1)
                            moves_made += 1
                            
                            if game.win_checks(game_board, 1):
                                deterministic_ai_wins = 1
                                break
                        else:
                            illegal_moves += 1
                    else:
                        illegal_moves += 1
                except:
                    illegal_moves += 1
                
                turn = 1
                
            else:
                start_time = time.time()
                try:
                    result = ai.minimax(distorted_view, 6, -math.inf, math.inf, True)
                    col, score = result if result[0] is not None else (None, 0)
                    move_time = time.time() - start_time
                    move_times.append(move_time)
                    
                    total_minimax_moves += 1
                    optimal_col = oracle_ai.get_move(game_board)
                    if col == optimal_col:
                        optimal_moves += 1
                    
                    if col is not None and game.is_valid_location(game_board, col):
                        row = game.for_next_open_row(game_board, col)
                        if row is not None:
                            game.drop_piece(game_board, row, col, 2)
                            distorted_row = game.for_next_open_row(distorted_view, col) 
                            if distorted_row is not None:
                                game.drop_piece(distorted_view, distorted_row, col, 2)
                            moves_made += 1
                            
                            if game.win_checks(game_board, 2):
                                minimax_ai_wins = 1
                                break
                        else:
                            illegal_moves += 1
                    else:
                        illegal_moves += 1
                except:
                    illegal_moves += 1
                
                turn = 0
        
        move_optimality = (optimal_moves / total_minimax_moves * 100) if total_minimax_moves > 0 else 0
        
        return {
            'moves_made': moves_made,
            'illegal_moves': illegal_moves,
            'minimax_ai_wins': minimax_ai_wins,
            'deterministic_ai_wins': deterministic_ai_wins,
            'game_length': moves_made,
            'avg_move_time': np.mean(move_times) if move_times else 0,
            'optimal_moves': optimal_moves,
            'total_minimax_moves': total_minimax_moves,
            'move_optimality_percent': move_optimality
        }
    
    def run_experiment(self):
        print("Testing 5 accuracy levels:")
        print("   Trial 1: 100% accuracy (control)")
        print("   Trial 2: 90% accuracy")
        print("   Trial 3: 80% accuracy")
        print("   Trial 4: 70% accuracy")
        print("   Trial 5: 50% accuracy")
        
        boards = self.create_test_boards()
        distortion_levels = [(0.0, 0.0), (0.05, 0.05), (0.10, 0.10), (0.15, 0.15), (0.25, 0.25)]
        
        print(f"Total games to play: {len(boards) * len(distortion_levels)}")
        
        results = []
        
        for trial_num, (blur_rate, misclass_rate) in enumerate(distortion_levels, 1):
            accuracy = 100 - (blur_rate + misclass_rate) * 100
            print(f"\n--- Trial {trial_num}: {accuracy:.0f}% Accuracy (Blur={blur_rate:.1%}, Misclass={misclass_rate:.1%}) ---")
            
            level_results = []
            for board_idx, board in enumerate(boards):
                distorted_board = self.apply_distortions(board, blur_rate, misclass_rate)
                
                starting_player = board_idx % 2
                
                game_result = self.play_single_game(board, distorted_board, starting_player)
                
                accuracy = 100 - (blur_rate + misclass_rate) * 100
                game_result.update({
                    'trial_number': trial_num,
                    'accuracy_percent': accuracy,
                    'blur_rate': blur_rate,
                    'misclass_rate': misclass_rate,
                    'board_index': board_idx,
                    'starting_player': 'Deterministic' if starting_player == 0 else 'Minimax',
                    'distortion_level': f"{accuracy:.0f}%_accuracy"
                })
                
                level_results.append(game_result)
                results.append(game_result)
                
                if (board_idx + 1) % 10 == 0:
                    print(f"  Completed {board_idx + 1}/{len(boards)} games")
            
            minimax_wins = sum(r['minimax_ai_wins'] for r in level_results)
            det_wins = sum(r['deterministic_ai_wins'] for r in level_results)
            total_illegal = sum(r['illegal_moves'] for r in level_results)
            avg_length = np.mean([r['game_length'] for r in level_results])
            avg_optimality = np.mean([r['move_optimality_percent'] for r in level_results if r['move_optimality_percent'] > 0])
            
            print(f"  Minimax AI (distorted): {minimax_wins}/{len(boards)} wins ({minimax_wins/len(boards)*100:.1f}%)")
            print(f"  Deterministic AI (perfect): {det_wins}/{len(boards)} wins ({det_wins/len(boards)*100:.1f}%)")
            print(f"  Illegal moves: {total_illegal}")
            print(f"  Avg game length: {avg_length:.1f} moves")
            print(f"  Avg move optimality: {avg_optimality:.1f}%" if not np.isnan(avg_optimality) else "  Avg move optimality: N/A")
        
        self.results = results
        return results
    
    def save_results(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cv_distortion_experiment_{timestamp}.csv"
        
        fieldnames = [
            'trial_number', 'accuracy_percent', 'distortion_level', 
            'blur_rate', 'misclass_rate', 'board_index', 'starting_player',
            'moves_made', 'illegal_moves', 'minimax_ai_wins', 'deterministic_ai_wins',
            'game_length', 'avg_move_time', 'optimal_moves', 'total_minimax_moves', 'move_optimality_percent'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        return filename

def main():
    experiment = QuickExperiment()
    
    start_time = time.time()
    results = experiment.run_experiment()
    total_time = time.time() - start_time
    
    filename = experiment.save_results()
    
    print(f"\nResults saved to: {filename}")
    print(f"\nEXPERIMENT COMPLETE!")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Games played: {len(results)}")
    print(f"Data saved to: {filename}")

if __name__ == "__main__":
    main()
