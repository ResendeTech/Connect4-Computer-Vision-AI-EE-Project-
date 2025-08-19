import cv2
import numpy as np
from PIL import Image, ImageOps

# img_pil = Image.open("Connect4.png")
# img_pil = ImageOps.exif_transpose(img_pil)


def preprocess(img):
    y = 0.5
    x = 0.5
    resizeimg = cv2.resize(img, None, fx=x, fy=y, interpolation=cv2.INTER_LINEAR)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(grayimg, (11, 11) ,0)
    detect_circles(grayimg, rgbimg)
    return blur, rgbimg

def detect_circles(preprocessed_img, rgbimg):
    circles = cv2.HoughCircles(preprocessed_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=15, minRadius=23, maxRadius=25)

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
    print(f"The rows gotten {rows}")
    return rows

def classify_cell(rows, img, empty_thresh=130):
    grid = []

    for row in rows:
        row_values = []
        for circle in row:
            x, y, r = circle

            x1 = int(x - r)
            x2 = int(x + r)
            y1 = int(y - r)
            y2 = int(y + r)

            cell_img = img[y1:y2, x1:x2]

            if np.mean(cell_img) > empty_thresh:
                row_values.append(0)
            else:
                gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                if np.mean(gray_cell) > empty_thresh:  # Light areas = empty
                    row_values.append(0)
                else:
                    # Color analysis in BGR (OpenCV format)
                    avg_colour = np.mean(cell_img.reshape(-1, 3), axis=0)
                    b, g, r_value = avg_colour

                    print(f"Circle at ({x},{y}): BGR=({b:.1f},{g:.1f},{r_value:.1f})")
                    
                    # Red detection: High Red, Low Green
                    if r_value > 130 and g < 20 and r_value > b + 10:
                        row_values.append(1)  # Red
                        #print(f"  -> RED detected")
                    # Yellow detection: High Red and Green, moderate Blue
                    elif r_value > 130 and g > 80 and g < r_value + 20:
                        row_values.append(2)  # Yellow
                        #print(f"  -> YELLOW detected")
                    else:
                        row_values.append(0)  # Empty/unclear
                        #print(f"  -> EMPTY (default)")
                        
        grid.append(row_values)
    print(grid)
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
    img = cv2.imread("C:/Users/victor.andraderesend/OneDrive - Tampereen seudun toisen asteen koulutus/Documents/GitHub/Connect4-Computer-Vision-AI-EE-Project-/CS_EE_Project/Connect_4_board.webp")
    
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
    
if __name__ == "__main__":
    img = cv2.imread("C:/Users/victor.andraderesend/OneDrive - Tampereen seudun toisen asteen koulutus/Documents/GitHub/Connect4-Computer-Vision-AI-EE-Project-/CS_EE_Project/C4B.png")
    if img is not None:
        preprocess(img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not load image")