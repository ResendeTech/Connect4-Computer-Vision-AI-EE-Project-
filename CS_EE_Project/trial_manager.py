import cv2
import numpy as np
import time
import os
from datetime import datetime
import random

# Import our modules
import game
import ai
from deterministic_ai import DeterministicAI
from data_logger import GameDataLogger
from vision_perturbation import VisionAccuracySimulator, create_accuracy_levels
import vision_complete_rewrite as vision  # Use the improved vision system
import math


class Connect4TrialManager:
    """
    Manages systematic trials of Connect 4 AI vs Deterministic AI
    with different vision accuracy levels
    """
    
    def __init__(self, base_save_path="trial_results"):
        self.base_save_path = base_save_path
        self.accuracy_levels = create_accuracy_levels()
        self.deterministic_ai = DeterministicAI()
        
        # Create save directory
        os.makedirs(base_save_path, exist_ok=True)
        
    def simulate_game_with_vision(self, accuracy_level, game_id, trial_name, 
                                image_path=None, use_camera=False):
        """
        Simulate a single game between minimax and deterministic AI
        with specified vision accuracy level and real physical board interaction
        """
        # Get accuracy simulator
        vision_sim = self.accuracy_levels[accuracy_level]
        
        # Initialize logger
        logger = GameDataLogger(trial_name, accuracy_level)
        logger.start_new_game(game_id)
        
        # Initialize game tracking (6x7 Connect 4 board)
        board = game.create_board()  # This tracks the actual state
        game_over = False
        turn = random.choice([0, 1])  # Randomize who starts
        
        print(f"\n🎮 Game {game_id}: Starting physical board game (Accuracy: {accuracy_level}%)")
        print(f"🎯 Turn order: {'Minimax AI' if turn == 0 else 'Deterministic AI'} starts")
        print(f"📊 Perturbations: {vision_sim.get_perturbation_summary()}")
        print("\n" + "="*60)
        
        move_count = 0
        max_moves = 42  # Maximum possible moves in Connect 4
        
        while not game_over and move_count < max_moves:
            move_count += 1
            current_player = 1 if turn == 0 else 2  # 1 = Minimax, 2 = Deterministic
            
            if turn == 0:  # Minimax AI turn (with vision)
                print(f"\n🤖 MINIMAX AI TURN (Move {move_count})")
                print("📸 Please ensure the physical board is ready for image capture")
                
                # Wait for human to trigger image capture
                input("Press ENTER when ready to capture image...")
                
                start_time = time.time()
                
                # Capture and process image with vision perturbations
                try:
                    if use_camera:
                        vision_board, vision_errors = vision_sim.get_perturbed_board_from_camera()
                    else:
                        print("⚠️  Camera mode not enabled, using simulated vision")
                        vision_board, vision_errors = vision_sim.perturb_board_detection(board)
                    
                    if vision_board is None:
                        print("❌ Vision failed completely! Using emergency fallback...")
                        vision_board = board.copy()
                        vision_errors = 5
                        
                except Exception as e:
                    print(f"❌ Vision processing error: {e}")
                    vision_board = board.copy()
                    vision_errors = 5
                
                # Get AI move based on vision
                col, minimax_score = ai.minimax(vision_board, 6, -math.inf, math.inf, True)
                
                end_time = time.time()
                evaluation_time = end_time - start_time
                
                # Display AI's decision
                print(f"🎯 Minimax AI chooses column: {col}")
                print(f"⚡ Evaluation time: {evaluation_time:.3f}s")
                print(f"🔍 Vision errors detected: {vision_errors}")
                print(f"📊 Minimax confidence score: {minimax_score}")
                
                # Check if move is legal
                is_legal = game.is_valid_location(board, col) if col is not None else False
                
                if is_legal:
                    print(f"✅ Legal move!")
                    print(f"🎲 Please place a MINIMAX piece in column {col} on the physical board")
                    
                    # Wait for human to place the piece
                    input("Press ENTER after placing the piece on the physical board...")
                    
                    # Update our tracking board
                    row = game.for_next_open_row(board, col)
                    game.drop_piece(board, row, col, 1)
                else:
                    print(f"❌ ILLEGAL MOVE! Column {col} is not valid")
                    # Get a valid move as fallback
                    valid_locations = ai.get_valid_locations(board)
                    if valid_locations:
                        col = random.choice(valid_locations)
                        print(f"🔄 Using fallback move: column {col}")
                        print(f"🎲 Please place a MINIMAX piece in column {col} on the physical board")
                        
                        input("Press ENTER after placing the piece on the physical board...")
                        
                        row = game.for_next_open_row(board, col)
                        game.drop_piece(board, row, col, 1)
                    else:
                        print("💥 No valid moves available! Game should end.")
                        game_over = True
                        break
                
                # Log move evaluation
                move_data = {
                    "chosen_column": col,
                    "is_legal": is_legal,
                    "evaluation_time": evaluation_time,
                    "minimax_score": minimax_score,
                    "vision_accuracy": vision_sim.target_accuracy,
                    "vision_errors": vision_errors
                }
                logger.log_move_evaluation(1, vision_board, move_data)
                
                # Check for win
                if game.win_checks(board, 1):
                    logger.finish_game(1, "minimax_win")
                    print(f"🏆 Game {game_id}: MINIMAX AI WINS!")
                    return logger
                
            else:  # Deterministic AI turn (perfect information)
                print(f"\n🎯 DETERMINISTIC AI TURN (Move {move_count})")
                
                start_time = time.time()
                col = self.deterministic_ai.get_move(board.copy())
                end_time = time.time()
                evaluation_time = end_time - start_time
                
                print(f"🎯 Deterministic AI chooses column: {col}")
                print(f"⚡ Evaluation time: {evaluation_time:.3f}s")
                
                # Check if move is legal
                is_legal = game.is_valid_location(board, col) if col is not None else False
                
                if is_legal:
                    print(f"✅ Legal move!")
                    print(f"🎲 Please place a DETERMINISTIC piece in column {col} on the physical board")
                    
                    # Wait for human to place the piece
                    input("Press ENTER after placing the piece on the physical board...")
                    
                    # Update our tracking board
                    row = game.for_next_open_row(board, col)
                    game.drop_piece(board, row, col, 2)
                else:
                    print(f"❌ ILLEGAL MOVE! This shouldn't happen with Deterministic AI")
                    # Emergency fallback
                    valid_locations = ai.get_valid_locations(board)
                    if valid_locations:
                        col = random.choice(valid_locations)
                        print(f"🔄 Using emergency fallback: column {col}")
                        print(f"🎲 Please place a DETERMINISTIC piece in column {col} on the physical board")
                        
                        input("Press ENTER after placing the piece on the physical board...")
                        
                        row = game.for_next_open_row(board, col)
                        game.drop_piece(board, row, col, 2)
                    else:
                        game_over = True
                        break
                
                # Log move evaluation
                move_data = {
                    "chosen_column": col,
                    "is_legal": is_legal,
                    "evaluation_time": evaluation_time,
                    "vision_accuracy": 1.0  # Deterministic AI has perfect info
                }
                logger.log_move_evaluation(2, board, move_data)
                
                # Check for win
                if game.win_checks(board, 2):
                    logger.finish_game(2, "deterministic_win")
                    print(f"🏆 Game {game_id}: DETERMINISTIC AI WINS!")
                    return logger
            
            # Check if board is full
            if len(ai.get_valid_locations(board)) == 0:
                logger.finish_game(0, "draw")
                print(f"🤝 Game {game_id}: DRAW!")
                game_over = True
            
            # Switch turns
            turn = 1 - turn
            
            # Show current board state
            print(f"\n📋 Current board state (internal tracking):")
            game.print_board(board)
        
        if move_count >= max_moves:
            logger.finish_game(0, "max_moves_reached")
            print(f"⏰ Game {game_id}: Maximum moves reached, declaring draw")
        
        return logger
    
    def run_accuracy_trial(self, accuracy_level, num_games=30, 
                          image_path=None, use_camera=False):
        """
        Run a complete trial for one accuracy level
        """
        trial_name = f"accuracy_{accuracy_level}_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n=== Starting Trial: {trial_name} ===")
        print(f"Accuracy Level: {accuracy_level}%")
        print(f"Number of Games: {num_games}")
        
        vision_sim = self.accuracy_levels[accuracy_level]
        print(f"Perturbations: {vision_sim.get_perturbation_summary()}")
        
        all_loggers = []
        
        for game_id in range(1, num_games + 1):
            logger = self.simulate_game_with_vision(
                accuracy_level, game_id, trial_name, 
                image_path, use_camera
            )
            all_loggers.append(logger)
            
            # Print progress
            if game_id % 5 == 0:
                print(f"Completed {game_id}/{num_games} games")
        
        # Combine all game data
        combined_logger = GameDataLogger(trial_name, accuracy_level)
        for logger in all_loggers:
            combined_logger.games_data.extend(logger.games_data)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.base_save_path, f"{trial_name}_{timestamp}")
        combined_logger.save_to_files(filename)
        
        # Print summary statistics
        stats = combined_logger.get_trial_statistics()
        print(f"\n=== Trial Results: {trial_name} ===")
        print(f"Total Games: {stats['total_games']}")
        print(f"Total Move Evaluations: {stats['total_move_evaluations']}")
        print(f"Minimax Win Rate: {stats['minimax_win_rate']:.1%}")
        print(f"Deterministic Win Rate: {stats['deterministic_win_rate']:.1%}")
        print(f"Draw Rate: {stats['draw_rate']:.1%}")
        print(f"Illegal Move Rate: {stats['illegal_move_rate']:.1%}")
        print(f"Avg Minimax Evaluation Time: {stats['avg_minimax_evaluation_time']:.3f}s")
        print(f"Vision Error Rate: {stats['vision_error_rate']:.1%}")
        
        return combined_logger, stats
    
    def run_complete_experiment(self, num_games_per_level=30, 
                               image_path=None, use_camera=False):
        """
        Run the complete experiment across all accuracy levels
        """
        print("=" * 60)
        print("STARTING COMPLETE CONNECT 4 AI VISION ACCURACY EXPERIMENT")
        print("=" * 60)
        
        experiment_results = {}
        experiment_start = time.time()
        
        for accuracy_level in sorted(self.accuracy_levels.keys(), reverse=True):
            trial_start = time.time()
            
            logger, stats = self.run_accuracy_trial(
                accuracy_level, num_games_per_level, 
                image_path, use_camera
            )
            
            experiment_results[accuracy_level] = {
                'logger': logger,
                'stats': stats,
                'trial_time': time.time() - trial_start
            }
            
            print(f"Trial completed in {time.time() - trial_start:.1f} seconds")
        
        # Save comprehensive experiment summary
        self.save_experiment_summary(experiment_results, experiment_start)
        
        return experiment_results
    
    def save_experiment_summary(self, experiment_results, start_time):
        """Save a comprehensive summary of the entire experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.base_save_path, f"experiment_summary_{timestamp}.txt")
        
        with open(summary_file, 'w') as f:
            f.write("CONNECT 4 AI VISION ACCURACY EXPERIMENT SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Experiment Start: {datetime.fromtimestamp(start_time)}\n")
            f.write(f"Total Experiment Time: {time.time() - start_time:.1f} seconds\n")
            f.write(f"Accuracy Levels Tested: {list(experiment_results.keys())}\n\n")
            
            for accuracy_level in sorted(experiment_results.keys(), reverse=True):
                stats = experiment_results[accuracy_level]['stats']
                trial_time = experiment_results[accuracy_level]['trial_time']
                
                f.write(f"ACCURACY LEVEL: {accuracy_level}%\n")
                f.write("-" * 30 + "\n")
                f.write(f"Trial Time: {trial_time:.1f} seconds\n")
                f.write(f"Games Played: {stats['total_games']}\n")
                f.write(f"Move Evaluations: {stats['total_move_evaluations']}\n")
                f.write(f"Minimax Win Rate: {stats['minimax_win_rate']:.1%}\n")
                f.write(f"Illegal Move Rate: {stats['illegal_move_rate']:.1%}\n")
                f.write(f"Avg Evaluation Time: {stats['avg_minimax_evaluation_time']:.3f}s\n")
                f.write(f"Vision Error Rate: {stats['vision_error_rate']:.1%}\n")
                f.write("\n")
        
        print(f"\nExperiment summary saved to: {summary_file}")


if __name__ == "__main__":
    # Example usage
    trial_manager = Connect4TrialManager("experiment_results")
    
    print("🎮 Connect 4 AI Vision Accuracy Trial System")
    print("=" * 60)
    print("This system conducts PHYSICAL board experiments where:")
    print("1. 📷 Camera captures real Connect 4 board")
    print("2. 🤖 Minimax AI analyzes perturbed vision")
    print("3. 🎲 You manually place pieces on physical board")
    print("4. 📊 Data is collected for analysis")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. 🎯 Run single accuracy level trial (recommended for testing)")
    print("2. 🚀 Run complete experiment (all accuracy levels)")
    print("3. 🔬 Test with specific image file")
    print("4. 📷 Test camera setup")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🎯 SINGLE ACCURACY LEVEL TRIAL")
        print("Available accuracy levels: 100%, 90%, 80%, 70%, 50%")
        accuracy = int(input("Enter accuracy level (e.g., 80): "))
        num_games = int(input("Enter number of games (default 5 for testing): ") or "5")
        
        use_camera = input("Use camera for real vision? (y/n, default=y): ").lower()
        use_camera = use_camera != 'n'
        
        print(f"\n🚀 Starting trial with {accuracy}% accuracy, {num_games} games")
        if use_camera:
            print("📷 Using REAL camera vision with perturbations")
        else:
            print("🎮 Using simulated vision (for testing)")
        
        print("\n⚠️  PREPARE YOUR PHYSICAL SETUP:")
        print("- 🎲 Connect 4 board ready")
        print("- 📷 Camera positioned and focused") 
        print("- 💡 Good lighting conditions")
        print("- ⌨️  Keyboard ready for manual triggers")
        
        input("\nPress ENTER when ready to start...")
        
        logger, stats = trial_manager.run_accuracy_trial(accuracy, num_games, use_camera=use_camera)
        
    elif choice == "2":
        print("\n🚀 COMPLETE EXPERIMENT")
        num_games = int(input("Enter games per accuracy level (default 10): ") or "10")
        
        use_camera = input("Use camera for real vision? (y/n, default=y): ").lower()
        use_camera = use_camera != 'n'
        
        print(f"\n⚠️  WARNING: This will run {num_games * 5} total games!")
        print("This is a FULL EXPERIMENT and will take several hours.")
        print("Each game requires manual piece placement.")
        
        confirm = input("Are you sure you want to proceed? (yes/no): ").lower()
        if confirm == "yes":
            print("\n🚀 Starting complete experiment...")
            results = trial_manager.run_complete_experiment(num_games, use_camera=use_camera)
        else:
            print("Experiment cancelled.")
            
    elif choice == "3":
        print("\n🔬 IMAGE FILE TEST")
        image_path = input("Enter path to Connect 4 board image: ")
        accuracy = int(input("Enter accuracy level (100, 90, 80, 70, 50): "))
        
        print(f"\n🧪 Testing with image: {image_path}")
        logger, stats = trial_manager.run_accuracy_trial(accuracy, 3, image_path=image_path)
        
    elif choice == "4":
        print("\n📷 CAMERA SETUP TEST")
        print("This will test your camera and vision pipeline...")
        
        try:
            import vision
            print("Testing camera capture...")
            img = vision.capture_from_camera()
            if img is not None:
                print("✅ Camera capture successful!")
                
                # Test vision pipeline
                board = vision.get_board_from_image_data(img)
                if board is not None:
                    print("✅ Vision processing successful!")
                    print("Detected board:")
                    print(board)
                else:
                    print("❌ Vision processing failed")
            else:
                print("❌ Camera capture failed")
                
        except Exception as e:
            print(f"❌ Camera test failed: {e}")
    
    else:
        print("❌ Invalid choice")
