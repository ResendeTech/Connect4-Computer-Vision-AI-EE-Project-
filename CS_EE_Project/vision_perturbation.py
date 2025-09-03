import cv2
import numpy as np
import random

class VisionAccuracySimulator:
    """
    Simulates different levels of vision accuracy by introducing
    controlled perturbations to images and board detection
    """
    
    def __init__(self, target_accuracy=1.0):
        """
        target_accuracy: float between 0.0 and 1.0
        1.0 = perfect vision, 0.5 = 50% accuracy
        """
        self.target_accuracy = target_accuracy
        self.error_rate = 1.0 - target_accuracy
        
    def perturb_image(self, image):
        """
        Apply image perturbations based on accuracy level
        Only uses Gaussian blur and lighting/color intensity changes
        """
        if self.target_accuracy >= 1.0:
            return image.copy()
        
        perturbed = image.copy().astype(np.float32)
        
        # Apply Gaussian blur (more blur for lower accuracy)
        if self.error_rate > 0:
            # Blur kernel size: 1 (no blur) to 15 (heavy blur)
            blur_kernel = int(1 + (self.error_rate * 14))
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            perturbed = cv2.GaussianBlur(perturbed, (blur_kernel, blur_kernel), 0)
        
        # Apply lighting/color intensity changes
        if self.error_rate > 0:
            # Brightness adjustment: -50 to +50
            brightness_change = (self.error_rate * 100 - 50) * 0.5
            perturbed = perturbed + brightness_change
            
            # Contrast adjustment: 0.5 to 1.5
            contrast_factor = 1.0 + (self.error_rate * 1.0 - 0.5)
            perturbed = perturbed * contrast_factor
            
            # Color saturation adjustment
            # Convert to HSV for saturation changes
            perturbed_uint8 = np.clip(perturbed, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(perturbed_uint8, cv2.COLOR_BGR2HSV)
            
            # Adjust saturation: 0.3 to 1.7
            saturation_factor = 0.3 + (1.0 - self.error_rate) * 1.4
            hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
            
            perturbed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # Ensure values are in valid range
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        
        return perturbed
    
    def perturb_board_detection(self, true_board):
        """
        Introduce errors in board state detection based on accuracy level
        Returns: (detected_board, vision_errors)
        """
        if self.target_accuracy >= 1.0:
            return np.copy(true_board), 0
        
        detected_board = np.copy(true_board)
        vision_errors = 0
        rows, cols = detected_board.shape
        
        # Calculate number of cells to potentially corrupt based on error rate
        total_cells = rows * cols
        # More conservative error rate - only affects small percentage of cells
        error_cells = int(total_cells * self.error_rate * 0.1)  # 10% of error rate affects cells
        
        for _ in range(error_cells):
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            
            # Only misclassify existing pieces (don't add/remove pieces)
            if detected_board[row][col] != 0:
                # Wrong color classification
                detected_board[row][col] = 1 if detected_board[row][col] == 2 else 2
                vision_errors += 1
        
        return detected_board, vision_errors
    
    def get_accuracy_level_name(self):
        """Get descriptive name for accuracy level"""
        if self.target_accuracy >= 1.0:
            return "Perfect"
        elif self.target_accuracy >= 0.9:
            return "Excellent"
        elif self.target_accuracy >= 0.8:
            return "Good"
        elif self.target_accuracy >= 0.7:
            return "Fair"
        elif self.target_accuracy >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def get_perturbation_summary(self):
        """Get summary of what perturbations are applied"""
        if self.target_accuracy >= 1.0:
            return "No perturbations applied"
        
        effects = []
        
        # Blur level
        blur_level = int(1 + (self.error_rate * 14))
        effects.append(f"Gaussian blur (kernel: {blur_level})")
        
        # Lighting changes
        brightness_change = abs((self.error_rate * 100 - 50) * 0.5)
        contrast_factor = 1.0 + (self.error_rate * 1.0 - 0.5)
        effects.append(f"Brightness: ±{brightness_change:.1f}, Contrast: {contrast_factor:.1f}x")
        
        # Saturation changes
        saturation_factor = 0.3 + (1.0 - self.error_rate) * 1.4
        effects.append(f"Saturation: {saturation_factor:.1f}x")
        
        # Board detection errors
        error_percent = self.error_rate * 10  # 10% of error rate
        effects.append(f"Board detection errors: ~{error_percent:.1f}% of pieces")
        
        return "; ".join(effects)
    
    def get_perturbed_board_from_camera(self):
        """
        Complete vision pipeline with perturbations using ML-enhanced system
        Returns: (board, vision_errors) or (None, error_count)
        """
        try:
            import vision_complete_rewrite as vision
            
            # Use the ML-enhanced vision system
            board = vision.detect_board_robust(debug=True)
            
            if board is not None:
                # Add board detection errors to simulate different accuracy levels
                final_board, detection_errors = self.perturb_board_detection(board)
                return final_board, detection_errors
            else:
                return None, 1  # Complete vision failure
                
        except Exception as e:
            print(f"Error in ML vision pipeline: {e}")
            return None, 1
    
    def get_perturbed_board_from_image(self, image_path):
        """
        Complete vision pipeline with perturbations from image file
        Returns: (board, vision_errors) or (None, error_count)
        """
        try:
            import vision_complete_rewrite as vision
            import cv2
            
            # Load original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                return None, 1
            
            # Apply image perturbations (blur + lighting)
            perturbed_image = self.perturb_image(original_image)
            
            # Run vision detection on perturbed image
            board = vision.get_board_from_image_data(perturbed_image)
            
            if board is not None:
                # Add board detection errors
                final_board, detection_errors = self.perturb_board_detection(board)
                return final_board, detection_errors
            else:
                return None, 1  # Complete vision failure due to perturbations
                
        except Exception as e:
            print(f"Error in perturbed vision pipeline: {e}")
            return None, 1


def create_accuracy_levels():
    """Create standard accuracy levels for testing"""
    return {
        100: VisionAccuracySimulator(1.0),
        90: VisionAccuracySimulator(0.9),
        80: VisionAccuracySimulator(0.8),
        70: VisionAccuracySimulator(0.7),
        50: VisionAccuracySimulator(0.5)
    }


def test_vision_perturbation():
    """Test the vision perturbation system with different accuracy levels"""
    print("Testing Vision Perturbation System")
    print("=" * 40)
    
    levels = [100, 90, 80, 70, 50]
    
    for accuracy in levels:
        simulator = VisionAccuracySimulator(accuracy / 100.0)
        print(f"\n{accuracy}% Accuracy ({simulator.get_accuracy_level_name()}):")
        print(f"Perturbations: {simulator.get_perturbation_summary()}")
        
        # Test with camera (if available)
        try:
            board, errors = simulator.get_perturbed_board_from_camera()
            if board is not None:
                print(f"✓ Camera test successful - {errors} vision errors detected")
            else:
                print("✗ Camera test failed - complete vision failure")
        except Exception as e:
            print(f"✗ Camera not available: {e}")


if __name__ == "__main__":
    test_vision_perturbation()
