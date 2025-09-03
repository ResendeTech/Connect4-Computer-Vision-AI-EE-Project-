import cv2
import numpy as np
from PIL import Image, ImageOps

# img_pil = Image.open("Connect4.png")
# img_pil = ImageOps.exif_transpose(img_pil)
def capture_from_camera():
    """Capture image from laptop camera"""
    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Set higher resolution for larger image display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Set width to 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720
    
    print("Camera opened. Press 'c' to capture, 'q' to quit")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame at full size
        cv2.imshow('Camera Feed - Press C to capture, Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):  # Capture image when 'c' is pressed
            print("✅ Image captured!")
            cap.release()
            cv2.destroyAllWindows()
            return frame
            
        elif key == ord('q'):  # Quit when 'q' is pressed
            print("🔄 Camera session ended")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def preprocess(img):
    y = 1.5
    x = 1.5
    resizeimg = cv2.resize(img, None, fx=x, fy=y, interpolation=cv2.INTER_LINEAR)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(grayimg, (11, 11) ,0)
    detect_circles(grayimg, rgbimg)
    return blur, rgbimg

def detect_circles(preprocessed_img, rgbimg):
    # Revert to more restrictive parameters to get ~42 circles, not 66
    circles = cv2.HoughCircles(preprocessed_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=15, minRadius=15, maxRadius=25)

    if circles is not None:
        for circle in circles[0, :]:
            cv2.circle(rgbimg, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 0, 255), 10)
        
        coordinate_point = []
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            coordinate = (x, y, r)
            coordinate_point.append(f"the coords are: {coordinate}")
        print(coordinate_point)
        print(circles)

        cv2.imshow("rgb", rgbimg)
        sort_into_grid(circles, rgbimg)

        return circles, coordinate_point
    else:
        print("No circles detected by HoughCircles")
        return None, None

def sort_into_grid(circles, processed_img):
    sorted_circles = sorted(circles, key=lambda c: c[1])
    circles_list = circles.tolist()
    # total = 0
    # count = 0
    # for circle in circles:
    #     total += circle[2]
    #     count += 1
    # r_avg = total/count
    # tolerance = r_avg * 0.7

    r_avg = np.mean(circles[:, 2])
    tolerance = r_avg * 0.7

    rows = []
    while len(circles_list) > 0:
        current_row = []
        first_circle = circles_list[0]

        for c in circles_list:
            if abs(c[1] - first_circle[1]) <= tolerance:
                current_row.append(c)

        for c in current_row:
            circles_list.remove(c)

        # for c in circles_to_remove:
        #     circles = np.delete(circles, np.where((circles == c).all(axis=1))[0], axis=0)

        sorted_row = sorted(current_row, key=lambda c: c[0])
        rows.append(sorted_row)

    rows = sorted(rows, key=lambda row: row[0][1])

    print(f"Detected {len(rows)} rows:")
    for i, row in enumerate(rows):
        print(f"  Row {i}: {len(row)} circles")

    return rows

def classify_cell(rows, img, empty_thresh=130):
    grid = []
    target_rows = 6
    target_cols = 7

    # Ensure we have exactly 6 rows
    while len(rows) < target_rows:
        rows.append([])  # Add empty rows
    
    if len(rows) > target_rows:
        rows = rows[:target_rows]

    for row_idx, row in enumerate(rows):
        row_values = []
        for circle in row:
            x, y, r = circle

            x1 = max(0, int(x - r))
            x2 = min(img.shape[1], int(x + r))
            y1 = max(0, int(y - r))
            y2 = min(img.shape[0], int(y + r))

            if x2 <= x1 or y2 <= y1:
                row_values.append(0)  # Default to empty
                continue

            cell_img = img[y1:y2, x1:x2]

            if cell_img.size == 0:
                row_values.append(0)
                continue

            hsv_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
            avg_hsv = np.mean(hsv_cell.reshape(-1, 3), axis=0)
            h, s, v = avg_hsv
            
            print(f"Circle at ({x},{y}): HSV=({h:.1f},{s:.1f},{v:.1f})")
            

            # Alternative approach: Use intensity and color variance instead of HSV
            # Extract the cell region for more detailed analysis
            cell_bgr = cv2.cvtColor(cell_img, cv2.COLOR_HSV2BGR)
            
            # Calculate color statistics
            mean_bgr = np.mean(cell_bgr.reshape(-1, 3), axis=0)
            std_bgr = np.std(cell_bgr.reshape(-1, 3), axis=0)
            b, g, r = mean_bgr
            
            # Calculate color intensity and ratios
            intensity = np.mean(mean_bgr)
            color_variance = np.sum(std_bgr)
            
            # Check for circular object detection using edge density
            gray_cell = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_cell, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            print(f"Circle at ({x},{y}): BGR=({b:.1f},{g:.1f},{r:.1f}) intensity={intensity:.1f} variance={color_variance:.1f} edges={edge_density:.3f}")
            
            # Classification based on actual data analysis
            # From your output, I can see distinct patterns:
            # Empty holes: BGR around (155-170, 160-183, 209-233) with high intensity ~183-191
            # Potential pieces: BGR like (172.1,213.9,191.1) or (171.5,206.8,196.6) - notice the high green!
            
            # Check for pieces based on observed patterns
            piece_detected = False
            
            # Orange pieces detection: Look for high green channel (orange reflects green strongly)
            if g > 205 and edge_density > 0.10 and intensity > 190:
                row_values.append(2)  # Orange 
                print(f"  -> ORANGE (high green {g:.1f})")
                piece_detected = True
                
            # Yellow pieces detection: Look for high overall intensity with balanced colors
            elif intensity > 200 and color_variance < 210 and edge_density > 0.10:
                # Check if it's more yellow-like (high red+green)
                if r > 190 and g > 190:
                    row_values.append(1)  # Yellow
                    print(f"  -> YELLOW (bright balanced)")
                    piece_detected = True
                    
            # Alternative detection: Look for unusual color distributions
            elif color_variance < 200 and edge_density > 0.10:
                # This might be a piece with different surface properties
                if g > r * 1.1 and g > b * 1.1:  # Green dominant
                    row_values.append(2)  # Orange
                    print(f"  -> ORANGE (green dominant)")
                    piece_detected = True
                elif r > 200 and g > 190:  # High red+green
                    row_values.append(1)  # Yellow  
                    print(f"  -> YELLOW (high R+G)")
                    piece_detected = True
                    
            if not piece_detected:
                row_values.append(0)  # Empty
                print(f"  -> EMPTY (no piece detected)")


            # if np.mean(cell_img) > empty_thresh:
            #     row_values.append(0)
            # else:
            #     gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            #     if np.mean(gray_cell) > empty_thresh:  # Light areas = empty
            #         row_values.append(0)
            #     else:
            #         # Color analysis in BGR (OpenCV format)
            #         avg_colour = np.mean(cell_img.reshape(-1, 3), axis=0)
            #         b, g, r_value = avg_colour

            #         print(f"Circle at ({x},{y}): BGR=({b:.1f},{g:.1f},{r_value:.1f})")
                    
            #         # Red detection: High Red, Low Green
            #         if r_value > 120 and g > 60 and g < 150 and b < 80 and r_value > g:
            #             row_values.append(1)  # Red
            #             #print(f"  -> RED detected")
            #         # Yellow detection: High Red and Green, moderate Blue
            #         elif r_value > 130 and g > 80 and b < 100 and abs(r_value - g) < 40:
            #             row_values.append(2)  # Yellow
            #             #print(f"  -> YELLOW detected")
            #         else:
            #             row_values.append(0)  # Empty/unclear
            #             #print(f"  -> EMPTY (default)")

                
        while len(row_values) < target_cols:
            row_values.append(0)  # Pad with empty cells
        
        if len(row_values) > target_cols:
            row_values = row_values[:target_cols]  # Trim extra columns
            
        print(f"Row {row_idx}: {row_values} ({len(row_values)} cells)")
        grid.append(row_values)
    
    print(f"Final grid ({len(grid)} rows):")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    return grid

def convert_to_board(grid):
    board = np.array(grid)
    
    # Ensure we have the right dimensions (6 rows x 7 columns)
    if len(grid) == 0:
        print("Warning: Empty grid, creating zero board")
        board = np.zeros((6, 7))
    elif board.shape != (6, 7):
        print(f"Warning: Board shape {board.shape} is not (6, 7)")
        # Create a proper 6x7 board
        proper_board = np.zeros((6, 7))
        rows_to_copy = min(6, board.shape[0])
        cols_to_copy = min(7, board.shape[1] if len(board.shape) > 1 else len(grid[0]) if grid else 0)
        
        for i in range(rows_to_copy):
            for j in range(cols_to_copy):
                if j < len(grid[i]):
                    proper_board[i][j] = grid[i][j]
        board = proper_board
    
    board = board.astype(int)  # Ensure integer type
    
    # Flip the board vertically so pieces appear at the bottom (correct Connect 4 orientation)
    board = np.flip(board, 0)
    
    print(f"Final board shape: {board.shape}")
    print("Board (flipped - pieces at bottom):")
    print(board)
    return board

def decide_turn(board):
    red_count = np.sum(board == 1)
    yellow_count = np.sum(board == 2)

    if red_count == yellow_count:
        return 0
    elif yellow_count == red_count + 1:
        return 1
    else:
        return "invalid board"
    


def get_board_from_image():
    """Get the board state from vision processing"""
    img = capture_from_camera()
    
    if img is None:
        print("Error: Image not found or path is incorrect.")
        return None

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    circles, coordinate_point = detect_circles(grayimg, rgbimg)

    
    if circles is not None:
        rows = sort_into_grid(circles, rgbimg)
        grid = classify_cell(rows, rgbimg)
        board = convert_to_board(grid)
        return board
    else:
        return None


def get_board_from_image_data(img):
    """Get board state from image data (numpy array) instead of camera capture"""
    if img is None:
        print("Error: No image data provided.")
        return None

    try:
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        circles, coordinate_point = detect_circles(grayimg, rgbimg)

        if circles is not None:
            print(f"Total circles detected: {len(circles)}")
            rows = sort_into_grid(circles, rgbimg)
            grid = classify_cell(rows, rgbimg)
            board = convert_to_board(grid)
            return board
        else:
            print("No circles detected in perturbed image")
            return None
            
    except Exception as e:
        print(f"Error processing image data: {e}")
        return None
    
if __name__ == "__main__":
    img = capture_from_camera()
    preprocess(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
