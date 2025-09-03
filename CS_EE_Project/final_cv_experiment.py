"""
SIMPLIFIED CONNECT 4 CV DISTORTION EXPERIMENT
==============================================

This is the final experiment for your EE project.
Tests how computer vision distortions affect AI decision making.

EXPERIMENT DESIGN:
1. Minimax AI (sees distorted board) vs Deterministic AI (sees perfect board) 
2. Tests how CV distortions degrade the advanced Minimax AI's performance
3. Measure: Win rate, illegal moves, move quality as distortion increases
4. Results saved to CSV for analysis

Run this script for your final data collection!
"""

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
        """Create 30 representative board states"""
        boards = []
        
        # Empty board
        boards.append(np.zeros((6, 7)))
        
        # Early game (5 boards)
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
        
        # Mid game (10 boards)  
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
        
        # Near-win scenarios (10 boards)
        near_win_patterns = [
            # Horizontal threats
            [(0,0,1), (0,1,1), (0,2,1)],  # 3 in a row horizontally
            # Vertical threats  
            [(0,3,1), (1,3,1), (2,3,1)],  # 3 in a column
            # Some complex patterns
            [(0,2,1), (0,3,1), (0,4,2), (1,2,2), (1,3,2)],
        ]
        
        for pattern in near_win_patterns:
            for i in range(3):  # 3 variations per pattern
                board = np.zeros((6, 7))
                for row, col, piece in pattern:
                    if row < 6 and col < 7:
                        board[row][col] = piece
                # Add some random pieces
                for _ in range(random.randint(3, 8)):
                    valid_cols = [c for c in range(7) if board[5][c] == 0]
                    if valid_cols:
                        col = random.choice(valid_cols)
                        for row in range(6):
                            if board[row][col] == 0:
                                board[row][col] = random.choice([1, 2])
                                break
                boards.append(board.copy())
        
        # Fill to exactly 30 boards
        while len(boards) < 30:
            board = self.create_random_board(random.randint(5, 20))
            boards.append(board)
        
        return boards[:30]  # Exactly 30 boards
    
    def create_random_board(self, num_moves):
        """Create a random valid board with specified number of moves"""
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
        """Apply CV distortions to board"""
        distorted = board.copy()
        
        # Blur: randomly remove pieces
        for r in range(6):
            for c in range(7):
                if board[r][c] != 0 and random.random() < blur_rate:
                    distorted[r][c] = 0
        
        # Misclassification: swap piece colors
        for r in range(6):
            for c in range(7):
                if distorted[r][c] != 0 and random.random() < misclass_rate:
                    distorted[r][c] = 2 if distorted[r][c] == 1 else 1
        
        return distorted
    
    def play_single_game(self, perfect_board, distorted_board):
        """
        Play one game: Minimax AI (sees distorted board) vs Deterministic AI (sees perfect board)
        This tests how CV distortions affect the smart minimax AI performance
        """
        game_board = perfect_board.copy()
        distorted_view = distorted_board.copy()
        
        # Track metrics
        moves_made = 0
        illegal_moves = 0
        minimax_ai_wins = 0  # Minimax AI (affected by distortions)
        deterministic_ai_wins = 0  # Deterministic AI (baseline with perfect vision)
        move_times = []
        
        # Create deterministic opponent (baseline - sees perfect board)
        deterministic_ai = StrongDeterministicAI(piece=1)
        deterministic_ai.reset()
        
        turn = 0  # 0 = deterministic AI (perfect vision), 1 = minimax AI (distorted vision)
        max_moves = 42
        
        for move_num in range(max_moves):
            if turn == 0:  # Deterministic AI turn (sees PERFECT board - baseline)
                try:
                    col = deterministic_ai.get_move(game_board)  # Perfect board
                    if col is not None and game.is_valid_location(game_board, col):
                        row = game.for_next_open_row(game_board, col)
                        if row is not None:
                            # Make move on real board
                            game.drop_piece(game_board, row, col, 1)
                            # Update distorted view too
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
                
            else:  # Minimax AI turn (sees DISTORTED board - this is what we're testing)
                start_time = time.time()
                try:
                    result = ai.minimax(distorted_view, 6, -math.inf, math.inf, True)  # Depth = 6
                    col, score = result if result[0] is not None else (None, 0)
                    move_time = time.time() - start_time
                    move_times.append(move_time)
                    
                    if col is not None and game.is_valid_location(game_board, col):
                        row = game.for_next_open_row(game_board, col)
                        if row is not None:
                            game.drop_piece(game_board, row, col, 2)
                            # Also update distorted view
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
        
        return {
            'moves_made': moves_made,
            'illegal_moves': illegal_moves,
            'minimax_ai_wins': minimax_ai_wins,  # Smart AI affected by CV distortion
            'deterministic_ai_wins': deterministic_ai_wins,  # Baseline AI with perfect vision
            'game_length': moves_made,
            'avg_move_time': np.mean(move_times) if move_times else 0
        }
    
    def run_experiment(self):
        """Run the full experiment"""
        print("🚀 STARTING CONNECT 4 CV DISTORTION EXPERIMENT")
        print("=" * 55)
        
        # Create ground truth boards
        print("📋 Creating 30 ground truth boards...")
        boards = self.create_test_boards()
        print(f"✅ Created {len(boards)} boards")
        
        # Define distortion levels to test (5 trials as specified)
        distortion_levels = [
            # Trial 1: Control (100% accuracy - no distortion)
            (0.0, 0.0),    # 100% accuracy - perfect vision baseline
            
            # Trial 2: 90% accuracy (10% total distortion)
            (0.05, 0.05),  # 5% blur + 5% misclassification = 90% accuracy
            
            # Trial 3: 80% accuracy (20% total distortion) 
            (0.10, 0.10),  # 10% blur + 10% misclassification = 80% accuracy
            
            # Trial 4: 70% accuracy (30% total distortion)
            (0.15, 0.15),  # 15% blur + 15% misclassification = 70% accuracy
            
            # Trial 5: 50% accuracy (50% total distortion)
            (0.25, 0.25),  # 25% blur + 25% misclassification = 50% accuracy
        ]
        
        print(f"🔬 Testing {len(distortion_levels)} accuracy levels:")
        print(f"   Trial 1: 100% accuracy (control)")
        print(f"   Trial 2: 90% accuracy") 
        print(f"   Trial 3: 80% accuracy")
        print(f"   Trial 4: 70% accuracy")
        print(f"   Trial 5: 50% accuracy")
        print(f"📊 Total games to play: {len(boards) * len(distortion_levels)}")
        
        results = []
        
        for trial_num, (blur_rate, misclass_rate) in enumerate(distortion_levels, 1):
            accuracy = 100 - (blur_rate + misclass_rate) * 100
            print(f"\n--- Trial {trial_num}: {accuracy:.0f}% Accuracy (Blur={blur_rate:.1%}, Misclass={misclass_rate:.1%}) ---")
            
            level_results = []
            for board_idx, board in enumerate(boards):
                # Apply distortions
                distorted_board = self.apply_distortions(board, blur_rate, misclass_rate)
                
                # Play game
                game_result = self.play_single_game(board, distorted_board)
                
                # Add experiment info
                accuracy = 100 - (blur_rate + misclass_rate) * 100
                game_result.update({
                    'trial_number': trial_num,
                    'accuracy_percent': accuracy,
                    'blur_rate': blur_rate,
                    'misclass_rate': misclass_rate,
                    'board_index': board_idx,
                    'distortion_level': f"{accuracy:.0f}%_accuracy"
                })
                
                level_results.append(game_result)
                results.append(game_result)
                
                if (board_idx + 1) % 10 == 0:
                    print(f"  Completed {board_idx + 1}/{len(boards)} games")
            
            # Print level summary
            minimax_wins = sum(r['minimax_ai_wins'] for r in level_results)
            det_wins = sum(r['deterministic_ai_wins'] for r in level_results)
            total_illegal = sum(r['illegal_moves'] for r in level_results)
            avg_length = np.mean([r['game_length'] for r in level_results])
            
            print(f"  📈 Minimax AI (distorted): {minimax_wins}/{len(boards)} wins ({minimax_wins/len(boards)*100:.1f}%)")
            print(f"  🤖 Deterministic AI (perfect): {det_wins}/{len(boards)} wins ({det_wins/len(boards)*100:.1f}%)")
            print(f"  ❌ Illegal moves: {total_illegal}")
            print(f"  📏 Avg game length: {avg_length:.1f} moves")
        
        self.results = results
        return results
    
    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cv_distortion_experiment_{timestamp}.csv"
        
        fieldnames = [
            'trial_number', 'accuracy_percent', 'distortion_level', 
            'blur_rate', 'misclass_rate', 'board_index',
            'moves_made', 'illegal_moves', 'minimax_ai_wins', 'deterministic_ai_wins',
            'game_length', 'avg_move_time'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename

def main():
    """Run the complete experiment"""
    experiment = QuickExperiment()
    
    print("This experiment will test how computer vision distortions")
    print("affect AI decision-making in Connect 4.")
    print("\nEstimated time: 5-10 minutes")
    print("\nPress Enter to start, or Ctrl+C to cancel...")
    input()
    
    # Run the experiment
    start_time = time.time()
    results = experiment.run_experiment()
    end_time = time.time()
    
    # Save results
    filename = experiment.save_results()
    
    print(f"\n🎉 EXPERIMENT COMPLETE!")
    print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
    print(f"🎯 Games played: {len(results)}")
    print(f"📁 Data saved to: {filename}")
    print(f"\n📊 You can now analyze this data for your EE project!")
    print(f"The CSV contains all the metrics you need for your analysis.")

if __name__ == "__main__":
    main()
