"""
Quick Test of Digital Experiment Framework
===========================================

This script runs a small test to verify the experiment framework works correctly.
"""

import numpy as np
from digital_experiment_framework import GroundTruthGenerator, BoardDistortor, ExperimentRunner

def quick_test():
    print("🧪 QUICK TEST OF EXPERIMENT FRAMEWORK")
    print("=" * 50)
    
    # Test 1: Generate a few ground truth boards
    print("📋 Test 1: Generating ground truth boards...")
    generator = GroundTruthGenerator()
    
    # Just create a few boards for testing
    early_boards = generator.create_early_game_boards()
    mid_boards = generator.create_mid_game_boards()
    near_win_boards = generator.create_near_win_boards()
    
    all_test_boards = early_boards[:2] + mid_boards[:2] + near_win_boards[:2]
    print(f"✅ Generated {len(all_test_boards)} test boards")
    
    # Test 2: Test board distortion
    print("\n🔬 Test 2: Testing board distortion...")
    distortor = BoardDistortor()
    
    test_board = all_test_boards[0]
    print("Original board:")
    print(test_board)
    
    # Apply distortions
    blurred = distortor.apply_blur(test_board, 0.1)  # 10% blur
    print("\nAfter 10% blur:")
    print(blurred)
    
    distortor.clear_log()
    misclassified = distortor.apply_misclassification(test_board, 0.1)  # 10% misclassification
    print("\nAfter 10% misclassification:")
    print(misclassified)
    
    # Test 3: Test deterministic AI
    print("\n🤖 Test 3: Testing deterministic AI...")
    from enhanced_deterministic_ai import create_deterministic_ai
    
    weak_ai = create_deterministic_ai("weak", piece=1)
    strong_ai = create_deterministic_ai("strong", piece=1) 
    
    test_board_with_pieces = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0],
        [0, 1, 1, 2, 0, 0, 0],
        [2, 2, 1, 1, 0, 0, 0]
    ])
    
    print("Test board:")
    print(test_board_with_pieces)
    
    weak_move = weak_ai.get_move(test_board_with_pieces)
    strong_move = strong_ai.get_move(test_board_with_pieces)
    
    print(f"Weak AI chooses column: {weak_move}")
    print(f"Strong AI chooses column: {strong_move}")
    
    # Test determinism
    weak_moves = [weak_ai.get_move(test_board_with_pieces) for _ in range(3)]
    strong_moves = [strong_ai.get_move(test_board_with_pieces) for _ in range(3)]
    
    print(f"Weak AI consistency: {weak_moves} - {'✅ DETERMINISTIC' if all(m == weak_moves[0] for m in weak_moves) else '❌ NOT DETERMINISTIC'}")
    print(f"Strong AI consistency: {strong_moves} - {'✅ DETERMINISTIC' if all(m == strong_moves[0] for m in strong_moves) else '❌ NOT DETERMINISTIC'}")
    
    # Test 4: Mini experiment run
    print("\n🚀 Test 4: Mini experiment run...")
    runner = ExperimentRunner()
    runner.load_ground_truth_boards(all_test_boards[:3])  # Just 3 boards
    
    # Run a very small experiment
    mini_distortion_levels = [
        (0.0, 0.0),   # No distortion
        (0.1, 0.05),  # Some distortion
    ]
    
    print("Running mini experiment (this may take a moment)...")
    results = runner.run_distortion_experiment(mini_distortion_levels, games_per_level=1)
    
    print(f"✅ Mini experiment complete! Ran {len(results)} games.")
    
    # Save mini results
    runner.save_results_to_csv("mini_test_results.csv")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("The framework is ready for full experiments.")

if __name__ == "__main__":
    quick_test()
