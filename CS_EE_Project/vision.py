import cv2
import numpy as np
from PIL import Image, ImageOps

# img_pil = Image.open("Connect4.png")
# img_pil = ImageOps.exif_transpose(img_pil)
img = cv2.imread("C:/Users/victor.andraderesend/OneDrive - Tampereen seudun toisen asteen koulutus/Documents/GitHub/Connect4-Computer-Vision-AI-EE-Project-/CS_EE_Project/Connect_4_board.webp")
# img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def preprocess_read(img, location = True):
    # y = 0.5
    # x = 0.5
    
    image = img.copy()
    #resizeimg = cv2.resize(img, None, fx=x, fy=y, interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("Resizeimg", resizeimg)

    lower_red = np.array([90, 130, 80])
    upper_red = np.array([115, 255, 255])

     
    HSVimage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 208, 94], dtype="uint8")
    upper = np.array([179, 255, 232], dtype="uint8")
    mask = cv2.inRange(HSVimage, lower, upper)

    # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
    #                         param1=50,param2=30,minRadius=0,maxRadius=0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = contours[0] if len(contours) == 2 else contours[1]
    #cv2.imshow("contour", HSVimage)

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        if len(approx) > 5:
            cv2.drawContours(image, [c], -1, (36, 255, 12), -1)
    
    cv2.imshow('mask', mask)
    cv2.imshow('original', image)


def find_board(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayimg, (11, 11) ,0)
    canny = cv2.Canny(blur, 50, 300)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.imshow("canny", canny)

    cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
    board_contour = None
    min_area = 3000

    for contour in cnt:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        

        if len(approx) >= 4:
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Connect 4 board is typically wider than tall (7x6 grid)
            if 0.8 < aspect_ratio < 1.8:

                rect_area = w * h
                extent = area / rect_area
                
                if extent > 0.6:
                    board_contour = contour
                    break
    
        return board_contour
    
    # Visualize the result
    result_img = img.copy()
    if board_contour is not None:
        cv2.drawContours(result_img, [board_contour], -1, (0, 255, 0), 3)
        print(f"Board found! Area: {cv2.contourArea(board_contour)}")
    else:
        print("No suitable board contour found")
    
    cv2.imshow("board_detection", result_img)
    split_into_cells(board_contour, img)

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
    classify_cell(rows, processed_img)
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
                hsv_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
                gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                if np.mean(gray_cell) > empty_thresh:  # Light areas = empty
                    row_values.append(0)
                else:
                    # Color analysis in RGB
                    avg_colour = np.mean(cell_img.reshape(-1, 3), axis=0)
                    b, g, r_value = avg_colour

                    print(f"Circle at ({x},{y}): RGB=({r_value:.1f},{g:.1f},{b:.1f})")
                    
                    # Red detection
                    if r_value > g + 30 and r_value > b + 30 and r_value > 100:
                        row_values.append(1)  # Red
                    # Yellow detection  
                    elif r_value > 120 and g > 120 and b < 80:
                        row_values.append(2)  # Yellow
                    else:
                        row_values.append(0)  # Empty/unclear
                        
        grid.append(row_values)
    print(grid)
    return grid

    
def split_into_cells(board, og_img):
    x, y, w, h = cv2.boundingRect(board)
    height = h//6
    width = w//7

    columns = 7
    rows = 6
    grid = []

    board_roi = og_img[y:y+h, x:x+w]
    for row in range(0, 6):
        row_cells = []
        for col in range(0, 7):
            x_start = col * width
            y_start = row * height
            
            cell_image = board[y_start : y_start + height, x_start : x_start + width]
            row_cells.append(cell_image)
            print(len(cell_image))
            #cv2.imshow(f"Cell_{row}_{col}", cell_image)
        grid.append(row_cells)
        print(len(row_cells))
    return grid
#preprocess_read(img)
preprocess(img)
cv2.waitKey(0)
cv2.destroyAllWindows()