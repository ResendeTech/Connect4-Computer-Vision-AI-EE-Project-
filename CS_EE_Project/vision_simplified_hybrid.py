"""
Simplified Connect 4 Vision System
Hybrid approach combining Matt Jennings' elegant algorithm with your camera integration
Much simpler and more reliable than the current complex system
"""

import cv2
import numpy as np
import time

class SimplifiedConnect4Vision:
    def __init__(self):
        self.cap = None
        self.is_camera_initialized = False
        
    def initialize_camera(self):
        """Initialize camera with optimized settings"""
        if self.is_camera_initialized and self.cap is not None and self.cap.isOpened():
            return True
            
        print("🔧 Initializing camera...")
        
        # Try DirectShow backend first (faster on Windows)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("DirectShow failed, trying default backend...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Could not open camera")
                return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm up camera
        for _ in range(3):
            self.cap.read()
        
        self.is_camera_initialized = True
        print("✅ Camera ready!")
        return True
    
    def capture_frame(self):
        """Capture frame from camera"""
        if not self.is_camera_initialized:
            if not self.initialize_camera():
                return None
        
        print("📸 Camera ready. Press 'c' to capture, 'q' to quit")
        
        while True:
            if self.cap is None:
                print("Camera is not initialized")
                break
            ret, frame = self.cap.read()
            if not ret:
                break
                
            cv2.imshow("Connect 4 - Press 'c' to capture, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                print("✅ Image captured!")
                cv2.destroyAllWindows()
                return frame
            elif key == ord('q'):
                print("❌ Cancelled by user")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return None
    
    def detect_board_from_camera(self, debug=False):
        """Main function: capture from camera and detect board"""
        frame = self.capture_frame()
        if frame is None:
            return None
            
        return self.detect_board_from_image(frame, debug)
    
    def detect_board_from_image(self, img, debug=False):
        """
        Simplified board detection using Matt Jennings' approach
        Much more reliable than the complex HoughCircles method
        """
        start_time = time.time()
        
        if debug:
            print("🔍 Starting simplified board detection...")
        
        # Step 1: Preprocessing (Matt's approach)
        img_resized = self.preprocess_image(img, debug)
        
        # Step 2: Find circles using contour detection (more reliable than HoughCircles)
        circles = self.find_circles_via_contours(img_resized, debug)
        
        if circles is None or len(circles) < 10:
            print("❌ Not enough circles detected")
            return None
        
        # Step 3: Create perfect 6x7 grid using interpolation (Matt's key insight)
        grid_positions = self.interpolate_perfect_grid(circles, debug)
        
        # Step 4: Classify colors using HSV masking (much more reliable)
        board = self.classify_grid_colors(img_resized, grid_positions, debug)
        
        # Step 5: Apply Connect 4 physics validation
        board = self.apply_gravity_validation(board, debug)
        
        detection_time = time.time() - start_time
        
        if debug:
            print(f"✅ Board detection completed in {detection_time:.3f}s")
            self.print_board_summary(board)
            self.show_detection_visualization(img_resized, grid_positions, board)
        
        return board
    
    def preprocess_image(self, img, debug=False):
        """Preprocess image - resize and filter"""
        # Resize to manageable size
        new_width = 640
        img_h, img_w = img.shape[:2]
        scale = new_width / img_w
        img_w = int(img_w * scale)
        img_h = int(img_h * scale)
        img_resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        
        if debug:
            print(f"📏 Resized to {img_w}x{img_h}")
        
        return img_resized
    
    def find_circles_via_contours(self, img, debug=False):
        """
        Find circles using contour detection (Matt's approach)
        More reliable than HoughCircles for Connect 4 boards
        """
        # Bilateral filter to smooth while preserving edges
        filtered = cv2.bilateralFilter(img, 15, 190, 190)
        
        # Edge detection
        edges = cv2.Canny(filtered, 75, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        img_h, img_w = img.shape[:2]
        
        for contour in contours:
            # Approximate polygon
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            
            # Bounding rectangle
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            x_center = x + w/2
            y_center = y + h/2
            area_rect = w * h
            
            # Matt's circle conditions (adapted)
            is_circular = (len(approx) > 8 and len(approx) < 23)  # Roughly circular
            good_size = (area > 200 and area_rect < (img_w * img_h) / 8)  # Reasonable size
            square_like = abs(w - h) < 15  # Roughly square bounding box
            
            if is_circular and good_size and square_like:
                circles.append((int(x_center), int(y_center), int((w + h) / 4)))
                if debug:
                    print(f"  Circle: center=({x_center:.0f}, {y_center:.0f}), size={w}x{h}, area={area:.0f}")
        
        if debug:
            print(f"🔍 Found {len(circles)} potential circles")
        
        return circles if circles else None
    
    def interpolate_perfect_grid(self, circles, debug=False):
        """
        Matt's key insight: Create perfect 6x7 grid using interpolation
        Even if some circles are missing, we can still get perfect positions
        """
        if len(circles) < 6:
            return None
        
        # Extract positions
        positions = [(x, y) for x, y, r in circles]
        
        # Find grid boundaries
        positions.sort(key=lambda p: p[0])  # Sort by x
        min_x, max_x = positions[0][0], positions[-1][0]
        
        positions.sort(key=lambda p: p[1])  # Sort by y
        min_y, max_y = positions[0][1], positions[-1][1]
        
        # Calculate spacing
        grid_width = max_x - min_x
        grid_height = max_y - min_y
        col_spacing = grid_width / 6 if grid_width > 0 else 80  # 6 intervals for 7 columns
        row_spacing = grid_height / 5 if grid_height > 0 else 70  # 5 intervals for 6 rows
        
        # Calculate average circle radius
        avg_radius = sum([r for _, _, r in circles]) / len(circles)
        
        if debug:
            print(f"📐 Grid boundaries: x({min_x:.0f}-{max_x:.0f}), y({min_y:.0f}-{max_y:.0f})")
            print(f"📏 Spacing: col={col_spacing:.0f}, row={row_spacing:.0f}, radius={avg_radius:.0f}")
        
        # Generate perfect 6x7 grid
        grid_positions = []
        for row in range(6):
            row_positions = []
            for col in range(7):
                x = int(min_x + col * col_spacing)
                y = int(min_y + row * row_spacing)
                r = int(avg_radius)
                row_positions.append((x, y, r))
            grid_positions.append(row_positions)
        
        if debug:
            print(f"✅ Generated perfect 6x7 grid ({len(grid_positions)} rows)")
        
        return grid_positions
    
    def classify_grid_colors(self, img, grid_positions, debug=False):
        """
        Enhanced color classification with better HSV ranges and debugging
        """
        # Convert to HSV for better color detection
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # IMPROVED HSV RANGES for better yellow detection
        # Yellow pieces - broader and more sensitive range
        lower_yellow1 = np.array([29, 26, 243])   # Lower yellow range
        upper_yellow1 = np.array([37, 87, 255])   # Upper yellow range
        mask_yellow = cv2.inRange(img_hsv, lower_yellow1, upper_yellow1)

        
        
        # Red/Orange pieces - refined range
        lower_red1 = np.array([0, 89, 253])      # Red-orange range 1
        upper_red1 = np.array([179, 134, 255])    
        mask_red = cv2.inRange(img_hsv, lower_red1, upper_red1)
        
        
        if debug:
            print("\n🎨 COLOR DETECTION DEBUG:")
            cv2.imshow("Original", img)
            cv2.imshow("HSV", img_hsv)
            cv2.imshow("Yellow Mask", mask_yellow)
            cv2.imshow("Red Mask", mask_red)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Initialize board
        board = np.zeros((6, 7), dtype=int)
        
        # Check each grid position with enhanced analysis
        for row_idx in range(6):
            for col_idx in range(7):
                x, y, r = grid_positions[row_idx][col_idx]
                
                # Create circular mask for this position (smaller radius for better accuracy)
                img_h, img_w = img.shape[:2]
                circle_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), int(r * 0.7), 255, thickness=-1)  # Use 70% of radius
                
                # Extract the actual cell region for additional analysis
                cell_region = self.extract_cell_region(img, img_hsv, x, y, r)
                
                # Check for yellow pieces (multiple methods)
                yellow_result = cv2.bitwise_and(circle_mask, mask_yellow)
                yellow_pixels = np.count_nonzero(yellow_result)
                
                # Check for red pieces  
                red_result = cv2.bitwise_and(circle_mask, mask_red)
                red_pixels = np.count_nonzero(red_result)
                
                # Additional color analysis using cell region
                yellow_confidence, red_confidence = self.analyze_cell_colors(cell_region, debug and row_idx < 2)
                
                # Enhanced classification logic
                min_pixels = max(10, r)  # Dynamic threshold based on circle size
                confidence_threshold = 0.3
                
                # Prioritize yellow detection (since it's having issues)
                is_yellow = (yellow_pixels > min_pixels or yellow_confidence > confidence_threshold)
                is_red = (red_pixels > min_pixels or red_confidence > confidence_threshold)
                
                if is_yellow and not is_red:
                    board[row_idx][col_idx] = 1  # Yellow piece
                    if debug:
                        print(f"  ({row_idx},{col_idx}): YELLOW (pixels:{yellow_pixels}, conf:{yellow_confidence:.2f})")
                elif is_red and not is_yellow:
                    board[row_idx][col_idx] = 2  # Red piece
                    if debug:
                        print(f"  ({row_idx},{col_idx}): RED (pixels:{red_pixels}, conf:{red_confidence:.2f})")
                elif is_yellow and is_red:
                    # Both detected - use confidence and pixel count
                    if yellow_confidence > red_confidence and yellow_pixels >= red_pixels * 0.8:
                        board[row_idx][col_idx] = 1  # Yellow
                        if debug:
                            print(f"  ({row_idx},{col_idx}): YELLOW (conflict resolved: yc={yellow_confidence:.2f} > rc={red_confidence:.2f})")
                    else:
                        board[row_idx][col_idx] = 2  # Red
                        if debug:
                            print(f"  ({row_idx},{col_idx}): RED (conflict resolved: rc={red_confidence:.2f} > yc={yellow_confidence:.2f})")
                else:
                    board[row_idx][col_idx] = 0  # Empty
                    if debug and (red_pixels > 5 or yellow_pixels > 5):
                        print(f"  ({row_idx},{col_idx}): EMPTY (y:{yellow_pixels}/{yellow_confidence:.2f}, r:{red_pixels}/{red_confidence:.2f})")
        
        return board
    
    def extract_cell_region(self, img, img_hsv, x, y, r):
        """Extract the cell region for detailed analysis"""
        # Extract square region around the circle
        size = int(r * 1.2)
        x1, x2 = max(0, x - size), min(img.shape[1], x + size)
        y1, y2 = max(0, y - size), min(img.shape[0], y + size)
        
        cell_bgr = img[y1:y2, x1:x2]
        cell_hsv = img_hsv[y1:y2, x1:x2]
        
        return {'bgr': cell_bgr, 'hsv': cell_hsv}
    
    def analyze_cell_colors(self, cell_region, debug=False):
        """
        Additional color analysis using multiple methods
        Returns confidence scores for yellow and red
        """
        if cell_region['bgr'].size == 0:
            return 0.0, 0.0
        
        # Method 1: HSV statistics
        hsv = cell_region['hsv']
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Method 2: BGR statistics
        bgr = cell_region['bgr']
        b_mean, g_mean, r_mean = np.mean(bgr.reshape(-1, 3), axis=0)
        
        # Yellow detection confidence
        yellow_conf = 0.0
        # Hue in yellow range (15-40)
        if 15 <= h_mean <= 40:
            yellow_conf += 0.4
        # High saturation and value
        if s_mean > 100 and v_mean > 100:
            yellow_conf += 0.3
        # BGR pattern for yellow (high G, high R, low B)
        if g_mean > 150 and r_mean > 120 and g_mean > b_mean and r_mean > b_mean:
            yellow_conf += 0.3
        
        # Red detection confidence  
        red_conf = 0.0
        # Hue in red range (0-10 or 160-180)
        if h_mean <= 10 or h_mean >= 160:
            red_conf += 0.4
        # High saturation
        if s_mean > 120:
            red_conf += 0.3
        # BGR pattern for red/orange (high R, moderate G, low B)
        if r_mean > 120 and r_mean > g_mean and r_mean > b_mean:
            red_conf += 0.3
        
        if debug:
            print(f"    HSV: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
            print(f"    BGR: B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f}")
            print(f"    Confidence: Yellow={yellow_conf:.2f}, Red={red_conf:.2f}")
        
        return yellow_conf, red_conf
    
    def apply_gravity_validation(self, board, debug=False):
        """Apply Connect 4 physics - pieces can't float"""
        if debug:
            print("\n🔍 Applying gravity validation...")
        
        corrected_board = board.copy()
        corrections = 0
        
        for col in range(7):
            # Find pieces in this column from bottom up
            column_pieces = []
            for row in range(5, -1, -1):  # Bottom to top
                if board[row][col] != 0:
                    column_pieces.append((row, board[row][col]))
            
            if not column_pieces:
                continue
            
            # Rebuild column with proper stacking
            corrected_column = [0] * 6
            for i, (_, piece_type) in enumerate(column_pieces):
                corrected_column[5 - i] = piece_type  # Place from bottom
            
            # Apply corrections
            for row in range(6):
                if corrected_board[row][col] != corrected_column[row]:
                    if corrected_column[row] == 0 and corrected_board[row][col] != 0:
                        corrections += 1
                        if debug:
                            print(f"  Removed floating piece at ({row},{col})")
                    corrected_board[row][col] = corrected_column[row]
        
        if debug:
            print(f"  🔧 Made {corrections} gravity corrections")
        
        return corrected_board
    
    def print_board_summary(self, board):
        """Print board summary"""
        empty = np.sum(board == 0)
        yellow = np.sum(board == 1)
        orange = np.sum(board == 2)
        
        print(f"\n🎯 BOARD SUMMARY:")
        print(f"  Empty: {empty}, Yellow: {yellow}, Orange: {orange}")
        print("  Board state:")
        for i, row in enumerate(board):
            row_str = " ".join(['🟡' if cell == 1 else '🔴' if cell == 2 else '⚪' for cell in row])
            print(f"    Row {i}: {row_str}")
    
    def show_detection_visualization(self, img, grid_positions, board):
        """Show detection visualization"""
        vis_img = img.copy()
        
        colors = {0: (128, 128, 128),    # Gray for empty
                 1: (0, 255, 255),      # Yellow  
                 2: (0, 0, 255)}        # Red
        
        for row_idx in range(6):
            for col_idx in range(7):
                x, y, r = grid_positions[row_idx][col_idx]
                piece_type = board[row_idx][col_idx]
                color = colors[piece_type]
                
                cv2.circle(vis_img, (x, y), r, color, 2)
                cv2.circle(vis_img, (x, y), 2, color, -1)
                
                # Add label
                label = ["E", "Y", "O"][piece_type]
                cv2.putText(vis_img, label, (x-5, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow("Detection Result", vis_img)
        print("🖼️  Visualization shown. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_camera_initialized = False
        print("📷 Camera released")

# Compatibility functions for your existing code
def get_board_from_image():
    """Compatibility function for trial_manager"""
    vision = SimplifiedConnect4Vision()
    return vision.detect_board_from_camera(debug=False)

def get_board_from_image_data(img):
    """Compatibility function for trial_manager"""
    vision = SimplifiedConnect4Vision()
    return vision.detect_board_from_image(img, debug=False)

def main():
    """Test the simplified vision system"""
    print("🚀 SIMPLIFIED CONNECT 4 VISION SYSTEM")
    print("=" * 50)
    print("Based on Matt Jennings' elegant approach")
    print("Much simpler and more reliable than complex ML systems")
    
    vision = SimplifiedConnect4Vision()
    
    try:
        board = vision.detect_board_from_camera(debug=True)
        
        if board is not None:
            print("\n✅ SUCCESS! Board detected successfully")
            print("This approach is much simpler and more reliable!")
        else:
            print("\n❌ Detection failed - check camera and lighting")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        vision.release_camera()

if __name__ == "__main__":
    main()
