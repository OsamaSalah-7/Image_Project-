import cv2
import numpy as np

def bgr_to_hsi(img_bgr):

    img_float = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    i = (r + g + b) / 3.0
    s = 1.0 - np.minimum(np.minimum(r, g), b) / i
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    den = np.maximum(den, 1e-10)
    h = np.arccos(num / den)
    h[b > g] = 2 * np.pi - h[b > g]
    h = h / (2 * np.pi)

    h = (h * 255).astype(np.uint8)
    s = (s * 255).astype(np.uint8)
    i = (i * 255).astype(np.uint8)

    img_hsi = cv2.merge((h, s, i))

    return img_hsi


cap = cv2.VideoCapture(0)

frame_width = 640
frame_height = 480

zoom = 1
predis=0
prev_centroids = []
distance=0

while True:

    ret, frame = cap.read()


    hsi_frame = bgr_to_hsi(frame)

    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])

    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsi_frame, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsi_frame, red_lower2, red_upper2)

    mask = cv2.bitwise_or(mask1, mask2)


    num_red_pixels = cv2.countNonZero(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursblue, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print("Centroid coordinates:", cx, cy)

            prev_centroids.append((cx, cy))

            prev_centroids = prev_centroids[-500:]

            for c in prev_centroids:
                cv2.circle(frame, c, 5, (0, 0, 255), -1)

            if num_red_pixels > 0:
                cv2.imshow('mask', mask)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_lower = np.array([90, 50, 50])
        blue_upper = np.array([130, 255, 255])

        mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        threshold_area = 500
        contours = [c for c in contours if cv2.contourArea(c) > threshold_area]

        largest_areas = sorted(contours, key=cv2.contourArea)[-2:]
        centers = []
        for contour in largest_areas:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 255, 255), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))

        if len(centers) == 2:
            distance = np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2)
            cv2.putText(frame, f"Distance: {distance:.2f} pixels", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
        elif len(centers) == 1:
            # Wait for 10 frames to see if the blue object splits into two
            for i in range(10):
                ret, frame = cap.read()
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = [c for c in contours if cv2.contourArea(c) > threshold_area]
                if len(contours) == 2:
                    # Blue object has split into two, exit loop and continue processing
                    break
                elif len(contours) == 1:
                    #continue
                    pass
                else:

                    break
            else:

                cv2.imwrite("blue_object_merged.png", frame)



        hsv_frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])

        mask1 = cv2.inRange(hsv_frame1, green_lower, green_upper)

        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_areas = sorted(contours1, key=cv2.contourArea)[-2:]
        centers = []
        for contour in largest_areas:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))

        if len(centers) == 2:
            distance = np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2)

            if distance < predis:
                zoom += 0.07
            elif distance > predis:
                zoom -= 0.07

            zoom = max(1, min(zoom, 3))

            new_width = int(frame_width * zoom)
            new_height = int(frame_height * zoom)
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            x = int((new_width - frame_width) / 2)
            y = int((new_height - frame_height) / 2)
            frame_cropped = frame_resized[y:y + frame_height, x:x + frame_width]

            cv2.imshow('frame', frame_cropped)

        else:

            cv2.imshow('frame', frame)
        predis = distance





    if cv2.waitKey(1) & 0xFF == ord('z'):
        break


cap.release()
cv2.destroyAllWindows()




