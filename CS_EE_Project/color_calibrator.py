"""
HSV Color Calibration Tool for Connect 4 Vision System
This tool helps you find the optimal HSV ranges for your specific pieces and lighting
"""

import cv2
import numpy as np

class ColorCalibrator:
    def __init__(self):
        self.current_image = None
        self.selected_points = []
        
    def calibrate_colors(self):
        """Interactive color calibration"""
        print("🎨 HSV COLOR CALIBRATION TOOL")
        print("=" * 50)
        print("This will help you find the perfect HSV ranges for your pieces")
        
        # Initialize camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Could not open camera")
                return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("📸 Camera ready. Press 'c' to capture image for calibration, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.imshow("Capture for Calibration - Press 'c' to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                self.current_image = frame.copy()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.current_image is None:
            return None
        
        # Start interactive calibration
        self.interactive_hsv_calibration()
        
    def interactive_hsv_calibration(self):
        """Interactive HSV range finding"""
        print("\n🎯 INTERACTIVE HSV CALIBRATION")
        print("Click on pieces to analyze their colors:")
        print("- LEFT CLICK on YELLOW pieces")
        print("- RIGHT CLICK on RED/ORANGE pieces") 
        print("- MIDDLE CLICK when done")
        
        # Convert to HSV
        if isinstance(self.current_image, np.ndarray):
            if isinstance(self.current_image, np.ndarray):
                if isinstance(self.current_image, np.ndarray):
                    hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
                else:
                    print("❌ Error: current_image is not a valid numpy array.")
                    return
            else:
                print("❌ Error: current_image is not a valid numpy array.")
                return
        else:
            print("❌ Error: current_image is not a valid numpy array.")
            return
        
        # Set up mouse callback
        cv2.namedWindow("Color Calibration - Click on pieces")
        cv2.setMouseCallback("Color Calibration - Click on pieces", self.mouse_callback, 
                            {'bgr': self.current_image, 'hsv': hsv_image})
        
        display_image = self.current_image.copy()
        
        yellow_samples = []
        red_samples = []
        
        while True:
            cv2.imshow("Color Calibration - Click on pieces", display_image)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('d'):  # Done
                break
                
            # Process collected samples
            for point_data in self.selected_points:
                x, y, button = point_data['x'], point_data['y'], point_data['button']
                
                # Extract 5x5 region around click
                hsv_sample = self.extract_color_sample(hsv_image, x, y)
                bgr_sample = self.extract_color_sample(self.current_image, x, y)
                
                if button == 1:  # Left click - Yellow
                    yellow_samples.append(hsv_sample)
                    cv2.circle(display_image, (x, y), 5, (0, 255, 255), -1)  # Yellow dot
                    cv2.putText(display_image, "Y", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif button == 2:  # Right click - Red
                    red_samples.append(hsv_sample)
                    cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)  # Red dot
                    cv2.putText(display_image, "R", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                print(f"Sample at ({x},{y}): HSV=({hsv_sample[0]:.1f},{hsv_sample[1]:.1f},{hsv_sample[2]:.1f}) BGR=({bgr_sample[2]:.1f},{bgr_sample[1]:.1f},{bgr_sample[0]:.1f})")
        
        # Clear selected points for next round
        self.selected_points = []
        
        cv2.destroyAllWindows()
        
        # Calculate optimal ranges
        if yellow_samples and red_samples:
            self.calculate_optimal_ranges(yellow_samples, red_samples)
        else:
            print("❌ Need samples of both yellow and red pieces!")
    
    def mouse_callback(self, event, x, y, flags, params):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append({'x': x, 'y': y, 'button': 1})
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_points.append({'x': x, 'y': y, 'button': 2})
    
    def extract_color_sample(self, image, x, y, size=5):
        """Extract average color from region around point"""
        h, w = image.shape[:2]
        x1 = max(0, x - size)
        x2 = min(w, x + size)
        y1 = max(0, y - size)  
        y2 = min(h, y + size)
        
        region = image[y1:y2, x1:x2]
        return np.mean(region.reshape(-1, image.shape[2]), axis=0)
    
    def calculate_optimal_ranges(self, yellow_samples, red_samples):
        """Calculate optimal HSV ranges from samples"""
        print("\n📊 CALCULATING OPTIMAL HSV RANGES")
        print("=" * 40)
        
        # Convert to numpy arrays
        yellow_samples = np.array(yellow_samples)
        red_samples = np.array(red_samples)
        
        # Calculate statistics for yellow
        yellow_h_mean = np.mean(yellow_samples[:, 0])
        yellow_s_mean = np.mean(yellow_samples[:, 1]) 
        yellow_v_mean = np.mean(yellow_samples[:, 2])
        
        yellow_h_std = np.std(yellow_samples[:, 0])
        yellow_s_std = np.std(yellow_samples[:, 1])
        yellow_v_std = np.std(yellow_samples[:, 2])
        
        # Calculate statistics for red
        red_h_mean = np.mean(red_samples[:, 0])
        red_s_mean = np.mean(red_samples[:, 1])
        red_v_mean = np.mean(red_samples[:, 2])
        
        red_h_std = np.std(red_samples[:, 0])
        red_s_std = np.std(red_samples[:, 1])
        red_v_std = np.std(red_samples[:, 2])
        
        print(f"🟡 YELLOW SAMPLES ({len(yellow_samples)}):")
        print(f"   H: {yellow_h_mean:.1f} ± {yellow_h_std:.1f}")
        print(f"   S: {yellow_s_mean:.1f} ± {yellow_s_std:.1f}")
        print(f"   V: {yellow_v_mean:.1f} ± {yellow_v_std:.1f}")
        
        print(f"🔴 RED SAMPLES ({len(red_samples)}):")
        print(f"   H: {red_h_mean:.1f} ± {red_h_std:.1f}")
        print(f"   S: {red_s_mean:.1f} ± {red_s_std:.1f}")  
        print(f"   V: {red_v_mean:.1f} ± {red_v_std:.1f}")
        
        # Generate suggested ranges (mean ± 2*std, clamped to valid HSV ranges)
        margin = 2.0  # How many standard deviations to include
        
        # Yellow ranges
        yellow_h_min = max(0, int(round(yellow_h_mean - margin * yellow_h_std)))
        yellow_h_max = min(179, int(round(yellow_h_mean + margin * yellow_h_std)))
        yellow_s_min = max(0, int(round(yellow_s_mean - margin * yellow_s_std)))
        yellow_s_max = min(255, int(round(yellow_s_mean + margin * yellow_s_std)))
        yellow_v_min = max(0, int(round(yellow_v_mean - margin * yellow_v_std)))
        yellow_v_max = min(255, int(round(yellow_v_mean + margin * yellow_v_std)))
        
        # Red ranges (handle hue wraparound)
        red_h_min = max(0, int(round(red_h_mean - margin * red_h_std)))
        red_h_max = min(179, int(round(red_h_mean + margin * red_h_std)))
        red_s_min = max(0, int(round(red_s_mean - margin * red_s_std)))
        red_s_max = min(255, int(round(red_s_mean + margin * red_s_std)))
        red_v_min = max(0, int(round(red_v_mean - margin * red_v_std)))
        red_v_max = min(255, int(round(red_v_mean + margin * red_v_std)))
        
        print(f"\n🎯 SUGGESTED HSV RANGES:")
        print("=" * 30)
        print("Add these to your vision_simplified_hybrid.py:")
        print()
        print("# CALIBRATED Yellow ranges")
        print(f"lower_yellow1 = np.array([{yellow_h_min:.0f}, {yellow_s_min:.0f}, {yellow_v_min:.0f}])")
        print(f"upper_yellow1 = np.array([{yellow_h_max:.0f}, {yellow_s_max:.0f}, {yellow_v_max:.0f}])")
        print()
        print("# CALIBRATED Red ranges") 
        print(f"lower_red1 = np.array([{red_h_min:.0f}, {red_s_min:.0f}, {red_v_min:.0f}])")
        print(f"upper_red1 = np.array([{red_h_max:.0f}, {red_s_max:.0f}, {red_v_max:.0f}])")
        print()
        
        # Test the ranges
        self.test_ranges_visually(yellow_h_min, yellow_h_max, yellow_s_min, yellow_s_max, yellow_v_min, yellow_v_max,
                                 red_h_min, red_h_max, red_s_min, red_s_max, red_v_min, red_v_max)
    
    def test_ranges_visually(self, yh_min, yh_max, ys_min, ys_max, yv_min, yv_max,
                           rh_min, rh_max, rs_min, rs_max, rv_min, rv_max):
        """Visually test the calculated ranges"""
        print("\n🔬 TESTING RANGES VISUALLY")
        print("Press any key to see the masks with your calibrated ranges...")
        
        if self.current_image is None or not isinstance(self.current_image, np.ndarray):
            print("❌ Error: No valid image available for testing ranges.")
            return
            
        hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        
        # Create masks with calibrated ranges
        yellow_mask = cv2.inRange(hsv_image, 
                                 np.array([yh_min, ys_min, yv_min]), 
                                 np.array([yh_max, ys_max, yv_max]))
        
        red_mask = cv2.inRange(hsv_image,
                              np.array([rh_min, rs_min, rv_min]),
                              np.array([rh_max, rs_max, rv_max]))
        
        # Show results
        if self.current_image is not None:
            cv2.imshow("Original", self.current_image)
        else:
            print("❌ Error: No image available to display.")
        cv2.imshow("Calibrated Yellow Mask", yellow_mask)
        cv2.imshow("Calibrated Red Mask", red_mask)
        
        # Combined result
        if self.current_image is not None:
            result = self.current_image.copy()
            result[yellow_mask > 0] = [0, 255, 255]  # Highlight yellow areas
            result[red_mask > 0] = [0, 0, 255]      # Highlight red areas
            cv2.imshow("Combined Detection", result)
        else:
            print("❌ Error: No image available for combined detection.")
        
        print("Check the masks - do they correctly identify your pieces?")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Run the color calibration tool"""
    calibrator = ColorCalibrator()
    calibrator.calibrate_colors()

if __name__ == "__main__":
    main()
