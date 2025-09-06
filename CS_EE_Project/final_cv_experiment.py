import numpy as np
import random
import csv
import time
import math
from datetime import datetime
import game
import ai
from enhanced_deterministic_ai import StrongDeterministicAI

class QuickExperiment:
    def __init__(self):
        self.results = []
    
    def create_test_boards(self):
        boards = []
        
        for i in range(30):
            boards.append(np.zeros((6, 7)))
        
        return boards
    
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
                if board[r][c] != 0 and random.random() < blur_rate:
                    distorted[r][c] = 0
        
        for r in range(6):
            for c in range(7):
                if distorted[r][c] != 0 and random.random() < misclass_rate:
                    distorted[r][c] = 2 if distorted[r][c] == 1 else 1
        
        return distorted
    
    def play_single_game(self, perfect_board, distorted_board, starting_player=0, blur_rate=0.0, misclass_rate=0.0):
        game_board = perfect_board.copy()
        distorted_view = distorted_board.copy()
        
        moves_made = 0
        illegal_moves = 0
        minimax_ai_wins = 0
        deterministic_ai_wins = 0
        move_times = []
        
        deterministic_ai = StrongDeterministicAI(piece=2)
        deterministic_ai.reset()
        
        turn = starting_player
        max_moves = 42
        
        for move_num in range(max_moves):
            if turn == 0:
                col = deterministic_ai.get_move(game_board)
                if col is not None and game.is_valid_location(game_board, col):
                    row = game.for_next_open_row(game_board, col)
                    if row is not None:
                        game.drop_piece(game_board, row, col, 1)
                        distorted_view = self.apply_distortions(game_board, blur_rate, misclass_rate)
                        moves_made += 1
                        
                        if game.win_checks(game_board, 1):
                            deterministic_ai_wins = 1
                            break
                    else:
                        illegal_moves += 1
                else:
                    illegal_moves += 1
                
                turn = 1
                
            else:
                start_time = time.time()
                result = ai.minimax(distorted_view, 5, -math.inf, math.inf, True)
                col, score = result if result and result[0] is not None else (None, 0)
                move_time = time.time() - start_time
                move_times.append(move_time)
                
                if col is not None and game.is_valid_location(game_board, col):
                    row = game.for_next_open_row(game_board, col)
                    if row is not None:
                        game.drop_piece(game_board, row, col, 2)
                        distorted_view = self.apply_distortions(game_board, blur_rate, misclass_rate)
                        moves_made += 1
                        
                        if game.win_checks(game_board, 2):
                            minimax_ai_wins = 1
                            break
                    else:
                        illegal_moves += 1
                else:
                    illegal_moves += 1
                
                turn = 0
        
        return {
            'moves_made': moves_made,
            'illegal_moves': illegal_moves,
            'minimax_ai_wins': minimax_ai_wins,
            'deterministic_ai_wins': deterministic_ai_wins,
            'game_length': moves_made,
            'avg_move_time': np.mean(move_times) if move_times else 0
        }
    
    def run_experiment(self):
        print("STARTING CONNECT 4 CV DISTORTION EXPERIMENT")
        print("=" * 55)
        
        print("Creating 30 ground truth boards...")
        boards = self.create_test_boards()
        print(f"Created {len(boards)} boards")
        
        # Define distortion levels to test (5 trials as specified)
        distortion_levels = [
            (0.0, 0.0),
            (0.01, 0.01),
            (0.02, 0.02),
            (0.03, 0.03),
            (0.05, 0.05),
        ]
        
        print(f"Testing {len(distortion_levels)} accuracy levels:")
        print(f"   Trial 1: 100% accuracy (control)")
        print(f"   Trial 2: 98% accuracy") 
        print(f"   Trial 3: 96% accuracy")
        print(f"   Trial 4: 94% accuracy")
        print(f"   Trial 5: 90% accuracy")
        print(f"Total games to play: {len(boards) * len(distortion_levels)}")
        
        results = []
        
        for trial_num, (blur_rate, misclass_rate) in enumerate(distortion_levels, 1):
            accuracy = 100 - (blur_rate + misclass_rate) * 100
            print(f"\n--- Trial {trial_num}: {accuracy:.0f}% Accuracy (Blur={blur_rate:.1%}, Misclass={misclass_rate:.1%}) ---")
            
            level_results = []
            for board_idx, board in enumerate(boards):
                distorted_board = self.apply_distortions(board, blur_rate, misclass_rate)
                
                starting_player = board_idx % 2
                
                game_result = self.play_single_game(board, distorted_board, starting_player, blur_rate, misclass_rate)
                
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
            
            print(f"  Minimax AI (distorted): {minimax_wins}/{len(boards)} wins ({minimax_wins/len(boards)*100:.1f}%)")
            print(f"  Deterministic AI (perfect): {det_wins}/{len(boards)} wins ({det_wins/len(boards)*100:.1f}%)")
            print(f"  Illegal moves: {total_illegal}")
            print(f"  Avg game length: {avg_length:.1f} moves")
        
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
            'game_length', 'avg_move_time'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\nResults saved to: {filename}")
        return filename

def main():
    experiment = QuickExperiment()
    
    print("\nPress Enter to start, or Ctrl+C to cancel...")
    input()
    
    start_time = time.time()
    results = experiment.run_experiment()
    end_time = time.time()
    
    filename = experiment.save_results()
    
    print(f"\nEXPERIMENT COMPLETE!")
    print(f"Total time: {end_time - start_time:.1f} seconds")
    print(f"Games played: {len(results)}")
    print(f"Data saved to: {filename}")

if __name__ == "__main__":
    main()
    