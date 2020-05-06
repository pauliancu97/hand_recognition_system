import cv2 as cv
import numpy as np
import easygui
from math import sqrt, cos, sin, pi, atan2, acos, degrees, radians
from functools import partial
from random import randint
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from time import time
import moderngl
import moderngl_window as mglw
from threading import Thread
from queue import Queue
from time import sleep


ORIGINAL_IMAGE_WINDOW = 'Original Image'
SEGMENTED_HAND_WINDOW = 'Segmented Hand'
FINAL_SEGMENTED_HAND_WINDOW = 'Final Segmented Hand'


class HSVValues:

    def __init__(self):
        self.lower_hue = 0
        self.upper_hue = 180
        self.lower_saturation = 0
        self.upper_saturation = 255
        self.lower_value = 0
        self.upper_value = 255


class TrackBar:

    def __init__(self, obj, attribute_name, lower_value, upper_value, track_bar_name, window_name, event_handler=None):
        self.obj = obj
        self.attribute_name = attribute_name
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.track_bar_name = track_bar_name
        self.window_name = window_name
        self.event_handler = event_handler
        cv.createTrackbar(track_bar_name, window_name, lower_value, upper_value, self.on_change)

    def on_change(self, value):
        setattr(self.obj, self.attribute_name, value)
        if self.event_handler is not None:
            self.event_handler()


class DistanceTransformVisualization:

    def __init__(self, img):
        self.img = img

    def get_image(self):
        distance_transform = cv.distanceTransform(self.img, cv.DIST_L2, 3)
        distance_transform_normalized = np.zeros(distance_transform.shape, dtype=np.uint8)
        cv.normalize(distance_transform, distance_transform_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        return distance_transform_normalized


class PalmPointVisualization:

    def __init__(self, img, point):
        self.img = img
        self.point = point

    def get_image(self):
        distance_transform = cv.distanceTransform(self.img, cv.DIST_L2, 3)
        distance_transform_normalized = np.zeros(distance_transform.shape, dtype=np.uint8)
        cv.normalize(distance_transform, distance_transform_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result = cv.cvtColor(distance_transform_normalized, cv.COLOR_GRAY2BGR)
        cv.circle(result, self.point, 10, (0, 255, 0), -1)
        return result


class PalmRadiusVisualisation:

    def __init__(self, img, point, radius):
        self.img = img
        self.point = point
        self.radius = radius

    def get_image(self):
        result = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        cv.circle(result, self.point, 10, (0, 255, 0), -1)
        cv.circle(result, self.point, int(self.radius), (0, 0, 255), 2)
        return result


class PalmMaskPointsVisualisation:

    def __init__(self, img, center, radius, sampled_points, palm_mask_points):
        self.img = img
        self.center = center
        self.radius = radius
        self.sampled_points = sampled_points
        self.palm_mask_points = palm_mask_points

    def get_image(self):
        result = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        cv.circle(result, self.center, 10, (0, 255, 0), -1)
        cv.circle(result, self.center, int(self.radius), (0, 0, 255), 2)
        for sample_point, mask_point in zip(self.sampled_points, self.palm_mask_points):
            cv.line(result, sample_point, mask_point, (255, 0, 0), 1)
        return result


class PalmWristPointsVisualisation:

    def __init__(self, img, center, radius, sampled_points, palm_mask_points, first_wrist_point, second_wrist_point):
        self.img = img
        self.center = center
        self.radius = radius
        self.sampled_points = sampled_points
        self.palm_mask_points = palm_mask_points
        self.first_wrist_point = first_wrist_point
        self.second_wrist_point = second_wrist_point

    def get_image(self):
        vis = PalmMaskPointsVisualisation(self.img, self.center, self.radius, self.sampled_points,
                                          self.palm_mask_points)
        result = vis.get_image()
        cv.circle(result, self.first_wrist_point, 10, (255, 255, 0), -1)
        cv.circle(result, self.second_wrist_point, 10, (255, 255, 0), -1)
        cv.line(result, self.first_wrist_point, self.second_wrist_point, (0, 255, 255), 3)
        return result


class FingersBoundingBoxesVisualisation:

    def __init__(self, fingers_image):
        self.fingers_image = fingers_image

    def get_image(self):
        result = cv.cvtColor(self.fingers_image, cv.COLOR_GRAY2BGR)
        contours, _ = cv.findContours(self.fingers_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        rectangles = [cv.minAreaRect(contour) for contour in contours]
        for rectangle in rectangles:
            points = cv.boxPoints(rectangle)
            for first_index in range(0, 4):
                second_index = first_index + 1
                second_index = 0 if second_index == 4 else second_index
                first_point = points[first_index]
                first_point = (int(first_point[0]), int(first_point[1]))
                second_point = points[second_index]
                second_point = (int(second_point[0]), int(second_point[1]))
                cv.line(result, first_point, second_point, (0, 0, 255), 2)
        return result


class FingerLinesVisualisation:

    def __init__(self, fingers_image, finger_lines):
        self.fingers_image = fingers_image
        self.finger_lines = finger_lines

    def get_image(self):
        result = cv.cvtColor(self.fingers_image, cv.COLOR_GRAY2BGR)
        for center, bottom in self.finger_lines:
            center = (int(center[0]), int(center[1]))
            bottom = (int(bottom[0]), int(bottom[1]))
            cv.circle(result, center, 5, (0, 255, 0), 2)
            cv.circle(result, bottom, 5, (0, 255, 0), 2)
            cv.line(result, center, bottom, (0, 0, 255), 2)
        return result


class FinalVisualization:

    def __init__(self, palm_point, palm_line, wrist_line, finger_lines, finger_contours, finger_status, thumb_index):
        self.palm_point = palm_point
        self.palm_line = palm_line
        self.wrist_line = wrist_line
        self.finger_lines = finger_lines
        self.finger_contours = finger_contours
        self.finger_status = finger_status
        self.thumb_index = thumb_index

    def get_image(self, size):
        fingers_descriptors = ('Thumb Finger', 'Index Finger', 'Middle Finger', 'Ring Finger', 'Little Finger')
        fingers_texts = []
        for index, status in enumerate(self.finger_status):
            status_str = 'Up' if status else 'Down'
            finger_text = '{}: {}'.format(fingers_descriptors[index], status_str)
            fingers_texts.append(finger_text)
        fingers_display_info = []
        for finger_text in fingers_texts:
            (width, height), baseline = cv.getTextSize(finger_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            height += baseline
            fingers_display_info.append((finger_text, width, height))
        max_width = max(fingers_display_info, key=lambda x: x[1])[1]
        whole_image = np.zeros((size[0], size[1] + max_width, 3), dtype=np.uint8)
        current_height_offset = 0
        for finger_display_info in fingers_display_info:
            cv.putText(whole_image, finger_display_info[0], (0, current_height_offset + finger_display_info[2]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            current_height_offset += finger_display_info[2]
        image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cv.circle(image, self.palm_point, 5, (0, 255, 0), 2)
        cv.circle(image, (self.wrist_line[0][0], self.wrist_line[0][1]), 3, (255, 0, 0), -1)
        cv.circle(image, (self.wrist_line[1][0], self.wrist_line[1][1]), 3, (255, 0, 0), -1)
        cv.line(image, (self.wrist_line[0][0], self.wrist_line[0][1]), (self.wrist_line[1][0], self.wrist_line[1][1]),
                (0, 255, 255), 2)
        for contour, _ in self.finger_contours:
            cv.drawContours(image, contour, -1, (0, 0, 255))
        if self.thumb_index is not None:
            thumb_contour, thumb_center = self.finger_contours[self.thumb_index]
            thumb_rect = cv.minAreaRect(thumb_contour)
            thumb_bottom = get_rectangle_bottom(thumb_rect)
            self.finger_lines.append((thumb_center, thumb_bottom))
        self.finger_lines = [((int(xc), int(yc)), (int(xb), int(yb))) for (xc, yc), (xb, yb) in self.finger_lines]
        for center, bottom in self.finger_lines:
            cv.circle(image, center, 3, (0, 255, 0), -1)
            cv.circle(image, bottom, 3, (0, 255, 0), -1)
            cv.line(image, center, bottom, (0, 255, 0))
            cv.line(image, bottom, self.palm_point, (0, 255, 0))
        if self.thumb_index is not None:
            fifth_length = (self.palm_line[1][0] - self.palm_line[0][0]) // 5
            first_palm_point = (self.palm_line[0][0] + fifth_length, self.palm_line[0][1])
            self.palm_line = (first_palm_point, self.palm_line[1])
        cv.line(image, self.palm_line[0], self.palm_line[1], (0, 255, 255))
        quarter_length = (self.palm_line[1][0] - self.palm_line[0][0]) // 4
        for index in range(1, 4):
            first_palm_point = self.palm_line[0]
            first_limit_point = (first_palm_point[0] + index * quarter_length, first_palm_point[1] + 5)
            second_limit_point = (first_palm_point[0] + index * quarter_length, first_palm_point[1] - 5)
            cv.line(image, first_limit_point, second_limit_point, (0, 0, 255))
        whole_image[:, max_width:] = image
        return whole_image


def on_change(img, hsv_values):
    img_equalized = equalize_histogram(img)
    img_hsv = cv.cvtColor(img_equalized, cv.COLOR_BGR2HSV)
    img_bin = cv.inRange(img_hsv, (hsv_values.lower_hue, hsv_values.lower_saturation, hsv_values.lower_value),
                         (hsv_values.upper_hue, hsv_values.upper_saturation, hsv_values.upper_value))
    cv.imshow(SEGMENTED_HAND_WINDOW, img_bin)


def equalize_histogram(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_channel = hsv_img[:, :, -1]
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    value_channel = clahe.apply(value_channel)
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            hsv_img[row, col, 2] = value_channel[row, col]
    result = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
    return result


def segment_hand(img, hsv_values):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_bin = cv.inRange(img_hsv, (hsv_values.lower_hue, hsv_values.lower_saturation, hsv_values.lower_value),
                         (hsv_values.upper_hue, hsv_values.upper_saturation, hsv_values.upper_value))
    kernel = np.ones(shape=(7, 7), dtype=np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img_bin)
    hand_label = max([(label, stats[label - 1, cv.CC_STAT_AREA]) for label in range(1, num_labels + 1)],
                     key=lambda x: x[1])[0]
    top = stats[hand_label - 1, cv.CC_STAT_TOP]
    left = stats[hand_label - 1, cv.CC_STAT_LEFT]
    rows = stats[hand_label - 1, cv.CC_STAT_HEIGHT]
    cols = stats[hand_label - 1, cv.CC_STAT_WIDTH]
    segmented_hand = np.zeros((rows, cols), dtype=np.uint8)
    for row in range(0, labels.shape[0]):
        for col in range(0, labels.shape[1]):
            if labels[row, col] == hand_label:
                segmented_hand[row - top, col - left] = 255
    return segmented_hand


def segment_hand_ycrcb(img):
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    img_bin= cv.inRange(img_ycrcb, (54, 131, 110), (163, 157, 135))
    kernel = np.ones(shape=(7, 7), dtype=np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img_bin)
    hand_label = max([(label, stats[label - 1, cv.CC_STAT_AREA]) for label in range(1, num_labels + 1)],
                     key=lambda x: x[1])[0]
    top = stats[hand_label - 1, cv.CC_STAT_TOP]
    left = stats[hand_label - 1, cv.CC_STAT_LEFT]
    rows = stats[hand_label - 1, cv.CC_STAT_HEIGHT]
    cols = stats[hand_label - 1, cv.CC_STAT_WIDTH]
    segmented_hand = np.zeros((rows, cols), dtype=np.uint8)
    for row in range(0, labels.shape[0]):
        for col in range(0, labels.shape[1]):
            if labels[row, col] == hand_label:
                segmented_hand[row - top, col - left] = 255
    return segmented_hand


def get_palm_point(segmented_hand):
    distance_transform = cv.distanceTransform(segmented_hand, cv.DIST_L2, 3)
    distance_transform_normalized = np.zeros(distance_transform.shape, dtype=np.uint8)
    cv.normalize(distance_transform, distance_transform_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    point = np.unravel_index(np.argmax(distance_transform, axis=None), distance_transform.shape)
    return point[1], point[0]


def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def get_maximum_radius(palm_point, contour):
    distances = [distance(palm_point, point) for point in contour]
    return min(distances)


def get_wrist_points(palm_mask_points):
    max_dist = 0
    max_index = 0
    for index in range(0, len(palm_mask_points) - 1):
        p1 = palm_mask_points[index]
        p2 = palm_mask_points[index + 1]
        if distance(p1, p2) > max_dist:
            max_dist = distance(p1, p2)
            max_index = index
    return palm_mask_points[max_index], palm_mask_points[max_index + 1]


def get_sampled_points(palm_point, radius):
    sampled_points = []
    for angle in range(0, 360):
        x = radius * cos(angle * pi / 180.0) + palm_point[0]
        y = radius * sin(angle * pi / 180.0) + palm_point[1]
        sampled_points.append((int(x), int(y)))
    return sampled_points


def get_palm_mask_points(sampled_points, contour):
    palm_mask_points = []
    np_contour = np.asarray(contour)
    for sampled_point in sampled_points:
        deltas = np_contour - np.array(sampled_point)
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        min_index = np.argmin(dist_2)
        palm_mask_points.append((contour[min_index][0], contour[min_index][1]))
    return palm_mask_points


def get_palm_mask_points_alg(sampled_points, segmented_hand):
    result = []
    offsets = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if not (x == 0 and y == 0)]
    for point in sampled_points:
        angle = 0
        rad = 1
        boundary_point = None
        while boundary_point is None:
            x = int(cos(radians(angle)) * rad + point[0])
            y = int(point[1] - sin(radians(angle)) * rad)
            if segmented_hand[x, y] == 0:
                has_black_pixels = False
                has_white_pixels = False
                for offset_x, offset_y in offsets:
                    if 0 <= offset_x + x < segmented_hand.shape[0] and 0 <= offset_y + y < segmented_hand.shape[1]:
                        offset = (x + offset_x, y + offset_y)
                        if segmented_hand[offset] == 0:
                            has_black_pixels = True
                        else:
                            has_white_pixels = True
                if has_black_pixels and (not has_white_pixels):
                    boundary_point = (x, y)
            if boundary_point is None:
                angle += 1
                rad += 1
        result.append(boundary_point)
    return result


def get_rotation_angle(palm_point, middle_wrist_point):
    x = palm_point[0] - middle_wrist_point[0]
    y = palm_point[1] - middle_wrist_point[1]
    hand_angle = atan2(y, x)
    rotation_angle = (hand_angle + pi / 2) * 180.0 / pi
    return rotation_angle


def get_rotation_matrix(palm_point, middle_wrist_point):
    rotation_angle = get_rotation_angle(palm_point, middle_wrist_point)
    rotation_matrix = cv.getRotationMatrix2D(palm_point, rotation_angle, 1)
    return rotation_matrix


def get_rotation_matrix_2d_point(angle, center):
    angle = 2 * pi - angle * pi / 180.0
    translation_matrix = np.array([[1.0, 0.0, -center[0]],
                                   [0.0, 1.0, -center[1]],
                                   [0.0, 0.0, 1.0]], dtype=np.float32)
    rotation_matrix = np.array([[cos(angle), -sin(angle), 0.0],
                                [sin(angle), cos(angle), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
    inverse_translation_matrix = np.array([[1.0, 0.0, center[0]],
                                   [0.0, 1.0, center[1]],
                                   [0.0, 0.0, 1.0]], dtype=np.float32)
    T = inverse_translation_matrix.dot(rotation_matrix.dot(translation_matrix))
    return T


def transform_point(point, angle, center):
    angle = 2 * pi - angle * pi / 180.0
    point = np.array([point[0], point[1], 1.0], dtype=np.float32)
    translation_matrix = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [-center[0], -center[1], 1.0]], dtype=np.float32)
    rotation_matrix = np.array([[cos(angle), sin(angle), 0.0],
                                [-sin(angle), cos(angle), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
    inverse_translation_matrix = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [center[0], center[1], 1.0]], dtype=np.float32)
    point = point.dot(translation_matrix)
    point = point.dot(rotation_matrix)
    point = point.dot(inverse_translation_matrix)
    return np.array((point[0], point[1]), dtype=np.float32)


def get_rect_area(rect):
    (_, _), (width, height), _ = rect
    return width * height


def get_fingers(segmented_hand):
    contours, _ = cv.findContours(segmented_hand, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    fingers_rectangles = [cv.minAreaRect(contour) for contour in contours]
    fingers_areas = [get_rect_area(finger_rectangle) for finger_rectangle in fingers_rectangles]
    avg_area = sum(fingers_areas) / len(fingers_areas)
    fingers_states = [finger_area >= avg_area / 2.5 for finger_area in fingers_areas]
    fingers_rectangles = [finger_rectangle for index, finger_rectangle in enumerate(fingers_rectangles)
                          if fingers_states[index]]
    contours = [contour for index, contour in enumerate(contours) if fingers_states[index]]
    centers = [(int(x), int(y)) for (x, y), (_, _), _ in fingers_rectangles]
    return list(zip(contours, centers))


def get_thumb_index(fingers, palm_point, first_wrist_point, second_wrist_point):
    wrist_vector = (second_wrist_point[0] - first_wrist_point[0], second_wrist_point[1] - first_wrist_point[1])
    wrist_vector = np.array(wrist_vector, dtype=np.float32)
    for index, (finger_contour, finger_center) in enumerate(fingers):
        finger_vector = (finger_center[0] - palm_point[0], finger_center[1] - palm_point[1])
        finger_vector = np.array(finger_vector, dtype=np.float32)
        cos_angle = wrist_vector.dot(finger_vector) / (np.linalg.norm(wrist_vector) * np.linalg.norm(finger_vector))
        angle = degrees(acos(cos_angle))
        if angle < 35:
            return index
    return None


def get_palm_line(segmented_hand, fingers, thumb_index):
    if thumb_index is not None:
        thumb_contour, _ = fingers[thumb_index]
        cv.drawContours(segmented_hand, [thumb_contour], -1, 0, -1)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    for palm_line in range(segmented_hand.shape[0] - 1, -1, -1):
        copy_segmented_hand = np.copy(segmented_hand)
        for row in range(palm_line, segmented_hand.shape[0]):
            for col in range(0, segmented_hand.shape[1]):
                copy_segmented_hand[row, col] = 0
        num_labels, _, _, _ = cv.connectedComponentsWithStats(copy_segmented_hand)
        if num_labels >= 3:
            x_min = None
            x_max = None
            for col in range(0, segmented_hand.shape[1]):
                if segmented_hand[palm_line, col] != 0:
                    if x_min is None:
                        x_min = col
                    x_max = col
            return palm_line - 1


def get_finger_line(segmented_hand, fingers, thumb_index):
    for contour, _ in fingers:
        cv.drawContours(segmented_hand, [contour], -1, 0, -1)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    global_x_min = None
    global_x_max = None
    palm_line = None
    stop = segmented_hand.shape[0]
    if thumb_index is not None:
        thumb_contour, center = fingers[thumb_index]
        rect = cv.minAreaRect(thumb_contour)
        box = cv.boxPoints(rect)
        box = sorted(box, key=lambda t: t[1])
        box = [np.array(point, dtype=np.int32) for point in box]
        stop = box[2][1]
    for row in range(0, stop):
        local_x_min = None
        local_x_max = None
        for col in range(0, segmented_hand.shape[1]):
            if segmented_hand[row, col] != 0:
                if local_x_min is None:
                    local_x_min = col
                local_x_max = col
        if global_x_max is None:
            global_x_max = local_x_max
            global_x_min = local_x_min
            palm_line = row
        else:
            if local_x_max is not None:
                if (local_x_max - local_x_min) > (global_x_max - global_x_min):
                    global_x_max = local_x_max
                    global_x_min = local_x_min
                    palm_line = row
    return (global_x_min, palm_line), (global_x_max, palm_line)


def get_horizontal_line_intersection(first_point, second_point, line):
    if second_point[0] - first_point[0] == 0.0:
        return second_point[0], line
    m = (second_point[1] - first_point[1]) / (second_point[0] - first_point[0])
    return int((line - first_point[1] + m * first_point[0]) / m), line


def get_fingers_status(fingers, thumb_index, first_finger_point, second_finger_point):
    fifth_length = (second_finger_point[0] - first_finger_point[0]) // 5
    if thumb_index is not None:
        quarter_length = ((second_finger_point[0] - first_finger_point[0]) - fifth_length) // 4
    else:
        quarter_length = (second_finger_point[0] - first_finger_point[0]) // 4
    fingers_status = [False, False, False, False, False]
    fingers_status[0] = thumb_index is not None
    finger_lines = get_fingers_lines(fingers, thumb_index, first_finger_point, second_finger_point)
    for center, bottom_center in finger_lines:
        if thumb_index is not None:
            length = int(bottom_center[0]) - first_finger_point[0] - fifth_length
        else:
            length = int(bottom_center[0]) - first_finger_point[0]
        finger_index = length // quarter_length
        finger_index = finger_index if finger_index >= 0 else 0
        if finger_index + 1 < 5:
            fingers_status[finger_index + 1] = True
        else:
            fingers_status[4] = True
    return fingers_status


def draw_fingers_line(first_finger_point, second_finger_point, thumb_index, color_image):
    fifth_length = (second_finger_point[0] - first_finger_point[0]) // 5
    if thumb_index is not None:
        quarter_length = ((second_finger_point[0] - first_finger_point[0]) - fifth_length) // 4
    else:
        quarter_length = (second_finger_point[0] - first_finger_point[0]) // 4
    if thumb_index is not None:
        first_point = (first_finger_point[0] + fifth_length, first_finger_point[1])
    else:
        first_point = (first_finger_point[0], first_finger_point[1])
    second_point = (first_point[0] + quarter_length, first_finger_point[1])
    third_point = (second_point[0] + quarter_length, first_finger_point[1])
    fourth_point = (third_point[0] + quarter_length, first_finger_point[1])
    fifth_point = (fourth_point[0] + quarter_length, first_finger_point[1])
    points = [first_point, second_point, third_point, fourth_point, fifth_point]
    for index in range(0, len(points) - 1):
        first_point = points[index]
        second_point = points[index + 1]
        red = randint(0, 255)
        green = randint(0, 255)
        blue = randint(0, 255)
        color = (blue, green, red)
        cv.line(color_image, first_point, second_point, color, 2)


def get_fingers_lines(fingers, thumb_index, first_palm_point, second_palm_point):
    lines = []
    palm_length = (second_palm_point[0] - first_palm_point[0]) / 5.0
    palm_length = ((second_palm_point[0] - first_palm_point[0]) - palm_length) / 4.0
    for index, finger in enumerate(fingers):
        if index != thumb_index:
            contour, center = finger
            (x, y), (width, height), angle = cv.minAreaRect(contour)
            original_rect = ((x, y), (width, height), angle)
            if width > height:
                temp = width
                width = height
                height = temp
            if round(float(width) / float(palm_length)) > 1:
                rect = original_rect
                num_of_rectangles = round(float(width) / float(palm_length))
                rectangles = divide_rect(rect, num_of_rectangles)
                for center, bottom in rectangles:
                    lines.append((center, bottom))
            else:
                rect = original_rect
                bottom_center = get_rectangle_bottom(rect)
                lines.append(((x, y), bottom_center))
    return lines


def get_fingers_distance_transform(segmented_hand, fingers, thumb_index):
    image = np.zeros(segmented_hand.shape, dtype=np.uint8)
    for index, finger in enumerate(fingers):
        if index != thumb_index:
            contour, _ = finger
            cv.fillPoly(image, [contour], 255)
    distance_transform = cv.distanceTransform(image, cv.DIST_L2, 3)
    return distance_transform


def get_finger_tips(segmented_hand, fingers, thumb_index, palm_center):
    distance_transform = get_fingers_distance_transform(segmented_hand, fingers, thumb_index)
    histogram = np.zeros((360,), dtype=np.float32)
    for row in range(0, distance_transform.shape[0]):
        for col in range(0, distance_transform.shape[1]):
            angle = degrees(atan2(row - palm_center[1], col - palm_center[0]))
            if angle < 0:
                angle += 360
            histogram[int(angle)] = histogram[int(angle)] + distance_transform[row, col]
    histogram = gaussian_filter1d(histogram, 3)
    peaks, _ = find_peaks(histogram)


def get_hand_attributes_with_visualisations(segmented_hand):
    contours, _ = cv.findContours(segmented_hand, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=lambda cnt: cv.arcLength(cnt, True))
    contour = [point[0] for point in contour]
    poly_image = np.zeros((segmented_hand.shape[0], segmented_hand.shape[1], 3), dtype=np.uint8)
    cv.drawContours(poly_image, np.array([contour]), -1, (255, 0, 0), 1)
    hand_contour_image = np.zeros(segmented_hand.shape, dtype=np.uint8)
    cv.fillPoly(hand_contour_image, np.array([contour]), 255)
    palm_point = get_palm_point(hand_contour_image)
    maximum_radius = get_maximum_radius(palm_point, contour)
    radius = 1.2 * maximum_radius
    sampled_points = get_sampled_points(palm_point, radius)
    palm_mask_points = get_palm_mask_points(sampled_points, contour)
    palm_mask_points_visualisation = PalmMaskPointsVisualisation(hand_contour_image, palm_point, radius, sampled_points,
                                                                 palm_mask_points)
    first_wrist_point, second_wrist_point = get_wrist_points(palm_mask_points)
    palm_wrist_points_visualisation = PalmWristPointsVisualisation(hand_contour_image, palm_point, radius,
                                                                   sampled_points, palm_mask_points, first_wrist_point,
                                                                   second_wrist_point)
    middle_wrist_point = ((first_wrist_point[0] + second_wrist_point[0]) / 2.0,
                          (first_wrist_point[1] + second_wrist_point[1]) / 2.0)
    rotation_angle = get_rotation_angle(palm_point, middle_wrist_point)
    rotation_matrix = get_rotation_matrix(palm_point, middle_wrist_point)
    segmented_hand = cv.warpAffine(segmented_hand, rotation_matrix, (segmented_hand.shape[0], segmented_hand.shape[1]))
    rotated_segmented_hand = np.copy(segmented_hand)
    transform_matrix = get_rotation_matrix_2d_point(rotation_angle, palm_point)
    first_wrist_point = np.array((first_wrist_point[0], first_wrist_point[1], 1.0), dtype=np.float32)
    second_wrist_point = np.array((second_wrist_point[0], second_wrist_point[1], 1.0), dtype=np.float32)
    palm_mask_points = np.array([(palm_mask_point[0], palm_mask_point[1], 1.0) for palm_mask_point in palm_mask_points],
                                dtype=np.float32)
    first_wrist_point = transform_matrix.dot(first_wrist_point)
    second_wrist_point = transform_matrix.dot(second_wrist_point)
    palm_mask_points = np.einsum('tj,ij->ti', palm_mask_points, transform_matrix)
    palm_mask_points = np.array(palm_mask_points, dtype=np.int32)
    palm_mask_points = palm_mask_points[:, :2]
    minimum_row = segmented_hand.shape[0]
    if first_wrist_point[1] < minimum_row:
        minimum_row = first_wrist_point[1]
    if second_wrist_point[1] < minimum_row:
        minimum_row = second_wrist_point[1]
    for row in range(int(minimum_row), segmented_hand.shape[0]):
        for col in range(0, segmented_hand.shape[1]):
            segmented_hand[row, col] = 0
    segmented_hand_no_arm = np.copy(segmented_hand)
    cv.fillPoly(segmented_hand, np.int32([palm_mask_points]), 0)
    mask_image = np.zeros(segmented_hand.shape, dtype=np.uint8)
    cv.fillPoly(mask_image, np.int32([palm_mask_points]), 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    fingers = get_fingers(segmented_hand)
    thumb_index = get_thumb_index(fingers, palm_point, first_wrist_point, second_wrist_point)
    first_finger_point, second_finger_point = get_finger_line(segmented_hand_no_arm, fingers, thumb_index)
    finger_lines = get_fingers_lines(fingers, thumb_index, first_finger_point, second_finger_point)
    fingers_status = get_fingers_status(fingers, thumb_index, first_finger_point, second_finger_point)
    visualization = FinalVisualization(palm_point, (first_finger_point, second_finger_point),
                                       (first_wrist_point, second_wrist_point), finger_lines, fingers, fingers_status,
                                       thumb_index)
    visualization_image = visualization.get_image(segmented_hand.shape)
    distance_transform_visualization = DistanceTransformVisualization(hand_contour_image)
    distance_transform_image = distance_transform_visualization.get_image()
    palm_point_visualisation = PalmPointVisualization(hand_contour_image, palm_point)
    palm_point_image = palm_point_visualisation.get_image()
    palm_radius_visualisation = PalmRadiusVisualisation(hand_contour_image, palm_point, radius)
    palm_radius_image = palm_radius_visualisation.get_image()
    palm_mask_points_image = palm_mask_points_visualisation.get_image()
    wrist_points_image = palm_wrist_points_visualisation.get_image()
    fingers_bounding_boxes_visualisation = FingersBoundingBoxesVisualisation(segmented_hand)
    fingers_bounding_boxes_image = fingers_bounding_boxes_visualisation.get_image()
    finger_lines_visualisation = FingerLinesVisualisation(segmented_hand, finger_lines)
    finger_lines_image = finger_lines_visualisation.get_image()
    cv.imshow('Final', visualization_image)
    cv.imshow('Distance Transform', distance_transform_image)
    cv.imshow('Palm Point', palm_point_image)
    cv.imshow('Palm Radius', palm_radius_image)
    cv.imshow('Palm Mask Points', palm_mask_points_image)
    cv.imshow('Wrist Points', wrist_points_image)
    cv.imshow('Rotated segmented hand', rotated_segmented_hand)
    cv.imshow('Palm Mask Image', mask_image)
    cv.imshow('Fingers', segmented_hand)
    cv.imshow('Fingers Bounding Boxes', fingers_bounding_boxes_image)
    cv.imshow('Finger Lines', finger_lines_image)

    cv.imwrite('Final.jpg', visualization_image)
    cv.imwrite('Distance Transform.jpg', distance_transform_image)
    cv.imwrite('Palm Point.jpg', palm_point_image)
    cv.imwrite('Palm Radius.jpg', palm_radius_image)
    cv.imwrite('Palm Mask Points.jpg', palm_mask_points_image)
    cv.imwrite('Wrist Points.jpg', wrist_points_image)
    cv.imwrite('Rotated segmented hand.jpg', rotated_segmented_hand)
    cv.imwrite('Palm Mask Image.jpg', mask_image)
    cv.imwrite('Fingers.jpg', segmented_hand)
    cv.imwrite('Fingers Bounding Boxes.jpg', fingers_bounding_boxes_image)
    cv.imwrite('Finger Lines.jpg', finger_lines_image)
    return fingers_status


def get_hand_attributes(segmented_hand):
    contours, _ = cv.findContours(segmented_hand, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=lambda cnt: cv.arcLength(cnt, True))
    contour = [point[0] for point in contour]
    poly_image = np.zeros((segmented_hand.shape[0], segmented_hand.shape[1], 3), dtype=np.uint8)
    cv.drawContours(poly_image, np.array([contour]), -1, (255, 0, 0), 1)
    hand_contour_image = np.zeros(segmented_hand.shape, dtype=np.uint8)
    cv.fillPoly(hand_contour_image, np.array([contour]), 255)
    palm_point = get_palm_point(hand_contour_image)
    maximum_radius = get_maximum_radius(palm_point, contour)
    radius = 1.2 * maximum_radius
    sampled_points = get_sampled_points(palm_point, radius)
    palm_mask_points = get_palm_mask_points(sampled_points, contour)
    first_wrist_point, second_wrist_point = get_wrist_points(palm_mask_points)
    middle_wrist_point = ((first_wrist_point[0] + second_wrist_point[0]) / 2.0,
                          (first_wrist_point[1] + second_wrist_point[1]) / 2.0)
    rotation_angle = get_rotation_angle(palm_point, middle_wrist_point)
    rotation_matrix = get_rotation_matrix(palm_point, middle_wrist_point)
    segmented_hand = cv.warpAffine(segmented_hand, rotation_matrix, (segmented_hand.shape[0], segmented_hand.shape[1]))
    transform_matrix = get_rotation_matrix_2d_point(rotation_angle, palm_point)
    first_wrist_point = np.array((first_wrist_point[0], first_wrist_point[1], 1.0), dtype=np.float32)
    second_wrist_point = np.array((second_wrist_point[0], second_wrist_point[1], 1.0), dtype=np.float32)
    palm_mask_points = np.array([(palm_mask_point[0], palm_mask_point[1], 1.0) for palm_mask_point in palm_mask_points],
                                dtype=np.float32)
    first_wrist_point = transform_matrix.dot(first_wrist_point)
    second_wrist_point = transform_matrix.dot(second_wrist_point)
    palm_mask_points = np.einsum('tj,ij->ti', palm_mask_points, transform_matrix)
    palm_mask_points = np.array(palm_mask_points, dtype=np.int32)
    palm_mask_points = palm_mask_points[:, :2]
    minimum_row = segmented_hand.shape[0]
    if first_wrist_point[1] < minimum_row:
        minimum_row = first_wrist_point[1]
    if second_wrist_point[1] < minimum_row:
        minimum_row = second_wrist_point[1]
    for row in range(int(minimum_row), segmented_hand.shape[0]):
        for col in range(0, segmented_hand.shape[1]):
            segmented_hand[row, col] = 0
    segmented_hand_no_arm = np.copy(segmented_hand)
    cv.fillPoly(segmented_hand, np.int32([palm_mask_points]), 0)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    fingers = get_fingers(segmented_hand)
    thumb_index = get_thumb_index(fingers, palm_point, first_wrist_point, second_wrist_point)
    first_finger_point, second_finger_point = get_finger_line(segmented_hand_no_arm, fingers, thumb_index)
    fingers_status = get_fingers_status(fingers, thumb_index, first_finger_point, second_finger_point)
    return fingers_status


def show_finger_status(fingers_status, frame):
    fingers_status = ['Up' if finger_status else 'Down' for finger_status in fingers_status]
    fingers_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    fingers_texts = ['{}: {}'.format(finger_name, finger_status)
                     for finger_name, finger_status in zip(fingers_names, fingers_status)]
    fingers_text_info = []
    for finger_text in fingers_texts:
        (width, height), baseline = cv.getTextSize(finger_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        height += baseline
        fingers_text_info.append((width, height, finger_text))
    current_height = 0
    for width, height, text in fingers_text_info:
        cv.putText(frame, text, (frame.shape[1] - width, current_height + height), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 0, 255), 1)
        current_height += height
    return frame


class Mouse:

    def __init__(self):
        self.selected = False
        self.finished = False
        self.first_corner = None
        self.second_corner = None
        self.first_point = None
        self.second_point = None

    def callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if not self.selected:
                self.selected = True
                self.first_corner = (x, y)
        elif event == cv.EVENT_LBUTTONUP:
            if self.selected and (not self.finished):
                self.finished = True
                self.selected = False
        elif event == cv.EVENT_MOUSEMOVE:
            if self.selected:
                self.second_corner = (x, y)
                self.first_point = [0, 0]
                self.second_point = [0, 0]
                self.first_point[0] = self.first_corner[0] if self.first_corner[0] < self.second_corner[0] else \
                    self.second_corner[0]
                self.first_point[1] = self.first_corner[1] if self.first_corner[1] < self.second_corner[1] else \
                    self.second_corner[1]
                self.second_point[0] = self.first_corner[0] if self.first_corner[0] > self.second_corner[0] else \
                    self.second_corner[0]
                self.second_point[1] = self.first_corner[1] if self.first_corner[1] > self.second_corner[1] else \
                    self.second_corner[1]
                self.first_point = (int(self.first_point[0]), int(self.first_point[1]))
                self.second_point = (int(self.second_point[0]), int(self.second_point[1]))

    def draw(self, image):
        if self.selected:
            cv.rectangle(image, self.first_point, self.second_point, (255, 0, 0))


def extract_foreground(image, background):
    foreground = np.zeros(image.shape, dtype=image.dtype)
    for row in range(0, foreground.shape[0]):
        for col in range(0, foreground.shape[1]):
            if np.logical_not(np.equal(image[row, col], background[row, col])).any():
                foreground[row, col] = image[row, col]
    return foreground


def third_main():
    camera = cv.VideoCapture(0)
    mouse = Mouse()
    cv.namedWindow('Camera')
    cv.setMouseCallback('Camera', mouse.callback)
    selected_roi = False
    first_point = None
    second_point = None
    hsv_values = None
    finger_status = [False] * 5
    hand_clear = False
    key_code = None
    tick_mark = cv.getTickCount()
    background_roi = None
    background_subtractor = cv.createBackgroundSubtractorMOG2()
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            if not selected_roi:
                mouse.draw(frame)
                if mouse.finished:
                    selected_roi = True
                    first_point = mouse.first_point
                    second_point = mouse.second_point
            else:
                cv.namedWindow('Panel')
                cv.rectangle(frame, first_point, second_point, (255, 0, 0))
                roi = frame[first_point[1]: second_point[1], first_point[0]: second_point[0]]
                if hsv_values is None:
                    background_roi = roi
                    background_roi = cv.cvtColor(background_roi, cv.COLOR_BGR2GRAY)
                    background_roi = cv.GaussianBlur(background_roi, (5, 5), 0)
                    hsv_values = HSVValues()
                    lower_hue_bar = TrackBar(hsv_values, 'lower_hue', 0, 180, 'Lower Hue', 'Panel')
                    upper_hue_bar = TrackBar(hsv_values, 'upper_hue', 0, 180, 'Upper Hue', 'Panel')
                    lower_saturation_bar = TrackBar(hsv_values, 'lower_saturation', 0, 255, 'Lower Saturation', 'Panel')
                    upper_saturation_bar = TrackBar(hsv_values, 'upper_saturation', 0, 255, 'Upper Saturation', 'Panel')
                    lower_value_bar = TrackBar(hsv_values, 'lower_value', 0, 255, 'Lower Value', 'Panel')
                    upper_value_bar = TrackBar(hsv_values, 'upper_value', 0, 255, 'Upper Value', 'Panel')
                else:
                    if not hand_clear:
                        '''roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                        roi = cv.GaussianBlur(roi, (21, 21), 0)
                        difference = cv.absdiff(roi, background_roi)
                        _, foreground = cv.threshold(difference, 20, 255, cv.THRESH_BINARY)
                        foreground = cv.dilate(foreground, kernel)'''
                        roi = equalize_histogram(roi)
                        roi = cv.GaussianBlur(roi, (7, 7), 0)
                        roi = np.flip(roi, axis=1)
                        segmented_hand = segment_hand(roi, hsv_values)
                        cv.imshow('Foreground', segmented_hand)
                        #equalized_histogram = equalize_histogram(roi)
                        #segmented_hand = segment_hand(equalized_histogram, hsv_values)
                        #cv.imshow('Segmented', segmented_hand)
                        if key_code == ord('S') or key_code == ord('s'):
                            hand_clear = True
                    else:
                        roi = equalize_histogram(roi)
                        roi = cv.GaussianBlur(roi, (9, 9), 0)
                        roi = np.flip(roi, axis=1)
                        #roi = cv.resize(roi, (200, 200), interpolation=cv.INTER_AREA)
                        segmented_hand = segment_hand(roi, hsv_values)
                        cv.imshow('Segmented', segmented_hand)
                        try:
                            finger_status = get_hand_attributes(segmented_hand)
                            frame = show_finger_status(finger_status, frame)
                        except Exception:
                            pass
            fps = cv.getTickFrequency() / (cv.getTickCount() - tick_mark)
            tick_mark = cv.getTickCount()
            print('FPS: {}'.format(fps))
            print(finger_status)
            cv.imshow('Camera', frame)
            key_code = cv.waitKey(5) & 0xFF
            if key_code == ord('q') or key_code == ord('Q'):
                break
    camera.release()


def get_masked_image(img, mask):
    result = np.zeros(img.shape, dtype=img.dtype)
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            if mask[row, col] == 255:
                result[row, col] = img[row, col]
    return result


def main():
    image_name = easygui.fileopenbox()
    img = cv.imread(image_name)
    hsv_values = HSVValues()
    cv.namedWindow(ORIGINAL_IMAGE_WINDOW)
    cv.namedWindow('Control Panel')
    cv.imshow(ORIGINAL_IMAGE_WINDOW, img)
    lower_hue_bar = TrackBar(hsv_values, 'lower_hue', 0, 180, 'Lower Hue', 'Control Panel',
                             partial(on_change, img, hsv_values))
    upper_hue_bar = TrackBar(hsv_values, 'upper_hue', 0, 180, 'Upper Hue', 'Control Panel',
                             partial(on_change, img, hsv_values))
    lower_saturation_bar = TrackBar(hsv_values, 'lower_saturation', 0, 255, 'Lower Saturation', 'Control Panel',
                                    partial(on_change, img, hsv_values))
    upper_saturation_bar = TrackBar(hsv_values, 'upper_saturation', 0, 255, 'Upper Saturation', 'Control Panel',
                                    partial(on_change, img, hsv_values))
    lower_value_bar = TrackBar(hsv_values, 'lower_value', 0, 255, 'Lower Value', 'Control Panel',
                               partial(on_change, img, hsv_values))
    upper_value_bar = TrackBar(hsv_values, 'upper_value', 0, 255, 'Upper Value', 'Control Panel',
                               partial(on_change, img, hsv_values))
    while True:
        if cv.waitKey(1) & 0xFF == ord('s'):
            break
    cv.destroyAllWindows()
    equalized_histogram = equalize_histogram(img)
    cv.imwrite('Original.jpg', img)
    cv.imwrite('Equalized histogram.jpg', equalized_histogram)
    segmented_hand = segment_hand(equalized_histogram, hsv_values)
    cv.imwrite('Mask.jpg', segmented_hand)
    finger_status = get_hand_attributes_with_visualisations(segmented_hand)
    print(finger_status)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


class Rectangle:

    def __init__(self, x, y, width, height, angle):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle

    def rect(self):
        return (self.x, self.y), (self.width, self.height), self.angle


def on_change_angle(rectangle, value):
    rectangle.angle = value
    rect = rectangle.rect()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    points = cv.boxPoints(rect)
    points = np.int0(points)
    cv.drawContours(image, [points], -1, (0, 0, 255), 2)
    cv.imshow('Rectangle', image)
    rectangles = divide_rect(rect, 3)
    draw_divided_rectangles(rectangles, image)


def get_rectangle_attributes(rect):
    (x, y), _, _ = rect
    points = cv.boxPoints(rect)
    dist_first_second = distance(points[0], points[1])
    dist_second_third = distance(points[1], points[2])
    first_point = points[0]
    second_point = points[1]
    width = dist_second_third
    height = dist_first_second
    if dist_second_third > dist_first_second:
        first_point = points[1]
        second_point = points[2]
        width = dist_first_second
        height = dist_second_third
    angle = degrees(atan2(second_point[1] - first_point[1], second_point[0] - first_point[0]))
    return (x, y), (width, height), angle


def divide_rect(rect, num_of_rectangles):
    rectangles = []
    (x, y), (width, height), angle = get_rectangle_attributes(rect)
    opposite_angle = 90.0 - angle
    x_offset = width * cos(radians(opposite_angle))
    y_offset = -width * sin(radians(opposite_angle))
    bottom_x_offset = width * cos(radians(-angle))
    bottom_y_offset = -width * sin(radians(-angle))
    bottom_offset = np.array((bottom_x_offset, bottom_y_offset), dtype=np.float32)
    first_margin = (x - x_offset / 2, y - y_offset / 2)
    for index in range(0, num_of_rectangles):
        first_margin_np = np.array(first_margin)
        offset = np.array((x_offset, y_offset), dtype=np.float32)
        point = first_margin_np + float(index) / float(num_of_rectangles) * offset \
                + offset / float(num_of_rectangles) / 2.0
        bottom = point - bottom_offset
        rectangles.append(((int(point[0]), int(point[1])), (int(bottom[0]), int(bottom[1]))))
    return rectangles


def draw_divided_rectangles(rectangles, image):
    for rectangle in rectangles:
        center, bottom = rectangle
        cv.circle(image, center, 3, (0, 0, 255), 2)
        cv.circle(image, bottom, 3, (255, 0, 0), 2)
    cv.imshow('Rectangle', image)


def get_rectangle_bottom(rectangle):
    points = cv.boxPoints(rectangle)
    points = sorted(points, key=lambda t: t[1])
    points = [np.array(point) for point in points]
    bottom = (points[2] + points[3]) / 2
    return bottom[0], bottom[1]


def second_main():
    rectangle = Rectangle(250, 250, 100, 200, 0)
    cv.namedWindow('Rectangle')
    rect = rectangle.rect()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    points = cv.boxPoints(rect)
    points = np.int0(points)
    cv.drawContours(image, [points], -1, (0, 0, 255), 2)
    cv.imshow('Rectangle', image)
    rectangles = divide_rect(rect, 3)
    draw_divided_rectangles(rectangles, image)
    cv.createTrackbar('Angle', 'Rectangle', 0, 360, partial(on_change_angle, rectangle))
    while True:
        if cv.waitKey(0) & 0xFF == ord('q') or cv.waitKey(0) & 0xFF == ord('Q'):
            break
    cv.destroyAllWindows()


def read_file(path):
    file_str = ''
    with open(path) as file:
        file_str = ''.join(file.readlines())
    return file_str


class Window(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernGL Example"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vertex_shader_str = read_file('vertex_shader.glsl')
        geometry_shader_str = read_file('geometry_shader.glsl')
        fragment_shader_str = read_file('fragment_shader.glsl')
        self.program = self.ctx.program(vertex_shader=vertex_shader_str, geometry_shader=geometry_shader_str,
                                        fragment_shader=fragment_shader_str)
        vertices = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, -0.5, 0.5], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f', 'pos')])

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.vao.render(mode=moderngl.LINES)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)


def fourth_main():
    Window.run()


class Yolo:

    def __init__(self, config_path, weights_path, confidence, threshold):
        self.config_path = config_path
        self.weights_path = weights_path
        self.net = cv.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.confidence = confidence
        self.threshold = threshold
        self.output_layers_names = self.net.getUnconnectedOutLayersNames()

    def match(self, img):
        height, width = img.shape[0], img.shape[1]
        blob = cv.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        results = self.net.forward(self.output_layers_names)
        boxes = []
        class_ids = []
        confidences = []
        for output in results:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    (x_match, y_match, width_match, height_match) = detection[:4]
                    x_match *= width
                    y_match *= height
                    width_match *= width
                    height_match *= height
                    x_match -= width_match / 2
                    y_match -= height_match / 2
                    x_match = int(x_match)
                    y_match = int(y_match)
                    width_match = int(width_match)
                    height_match = int(height_match)
                    boxes.append((x_match, y_match, width_match, height_match))
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        if len(indexes) == 0:
            return []
        indexes = indexes.flatten()
        boxes = [boxes[index] for index in indexes]
        return boxes


def test_hand_detection():
    yolo = Yolo('cross-hands.cfg', 'cross-hands.weights', 0.5, 0.3)
    tracker = cv.cv2.TrackerCSRT_create()
    camera = cv.VideoCapture(0)
    hand_bounding_box = None
    roi = None
    finger_status = None
    face_detector = cv.CascadeClassifier('faces.xml')
    eye_detector = cv.CascadeClassifier('eyes.xml')
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            if hand_bounding_box is None:
                boxes = yolo.match(frame)
                if cv.waitKey(1) & 0xFF == ord('s'):
                    hand_bounding_box = boxes[0]
                    tracker.init(frame, hand_bounding_box)
                for x, y, width, height in boxes:
                    cv.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
            else:
                success, hand_bounding_box = tracker.update(frame)
                if success:
                    x, y, width, height = hand_bounding_box
                    center_x = x + width / 2.0
                    center_y = y + height / 2.0
                    width *= 1.7
                    height *= 1.7
                    x = center_x - width / 2
                    y = center_y - height / 2
                    split_x = int(x) if x >= 0 else 0
                    split_y = int(y) if y >= 0 else 0
                    roi = frame[split_y: split_y + int(height), split_x: split_x + int(width)]
                    hand_roi = np.copy(roi)
                    roi = np.flip(roi, axis=1)
                    roi = segment_hand_ycrcb(roi)
                    try:
                        finger_status = get_hand_attributes(roi)
                    except Exception:
                        pass
                    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    face_detections = face_detector.detectMultiScale(frame_gray)
                    if len(face_detections) > 0:
                        face_x, face_y, face_width, face_height = face_detections[0]
                        face_center_x = face_x + face_width / 2
                        face_center_y = face_y + face_height / 2
                        face_width *= 0.75
                        face_x = int(face_center_x - face_width / 2)
                        face_y = int(face_center_y - face_height / 2)
                        face_width = int(face_width)
                        face_roi = np.copy(frame[face_y: face_y + face_height, face_x: face_x + face_width])
                        face_roi_gray = frame_gray[face_y: face_y + face_height, face_x: face_x + face_width]
                        eye_detections = eye_detector.detectMultiScale(face_roi_gray)
                        face_area = face_width * face_height
                        for eye_x, eye_y, eye_width, eye_height in eye_detections:
                            first_point = (eye_x, eye_y)
                            second_point = (eye_x + eye_width, eye_y + eye_height)
                            cv.rectangle(face_roi, first_point, second_point, (0, 0, 0), -1)
                            face_area -= (eye_width * eye_height)
                        face_mean_color = np.array(np.sum(face_roi, axis=0).sum(axis=0), dtype=np.float64)
                        face_mean_color *= (1.0 / face_area)
                        face_mean_color = np.array(face_mean_color, dtype=np.uint8)
                        mean_image = np.zeros(hand_roi.shape, dtype=np.uint8)
                        for row in range(0, mean_image.shape[0]):
                            for col in range(0, mean_image.shape[1]):
                                mean_image[row, col] = face_mean_color
                        sub = cv.subtract(hand_roi, mean_image)
                        cv.imshow('Sub', sub)
                        print(face_mean_color)
                    cv.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (255, 0, 0), 2)
            cv.imshow('Frame', frame)
            if roi is not None:
                cv.imshow('Roi', roi)
            if finger_status is not None:
                #print(finger_status)
                pass
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


def test_rgb_to_ycb():
    camera = cv.VideoCapture(0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            frame_cvt = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
            skin_mask = cv.inRange(frame_cvt, (54, 131, 110), (163, 157, 135))
            skin_mask = cv.dilate(skin_mask, kernel)
            skin_mask = cv.cvtColor(skin_mask, cv.COLOR_GRAY2BGR)
            frame_skin = cv.bitwise_and(frame, skin_mask)
            cv.imshow('Frame', frame_skin)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def test_web_camera_thread():
    queue = Queue()
    web_camera_thread = WebCameraThread()
    visualisation_thread = VisualisationThread(queue)
    web_camera_thread.init()
    web_camera_thread.start()
    visualisation_thread.start()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    finger_status = [False] * 5
    while True:
        frame = web_camera_thread.frame()
        if not visualisation_thread.selected():
            sleep(0.01)
        if visualisation_thread.selected():
            roi = visualisation_thread.rectangle()
            frame = frame[roi]
            frame = segment_hand_ycrcb(frame)
            frame = np.flip(frame, axis=1)
            try:
                finger_status = get_hand_attributes(frame)
                print(finger_status)
            except Exception:
                pass
        queue.put(frame)


class WebCameraThread(Thread):

    def __init__(self):
        super().__init__()
        self.daemon = True
        self.current_frame = None
        self.camera = None

    def init(self):
        self.camera = cv.VideoCapture(0)
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame

    def frame(self):
        return self.current_frame


class VisualisationThread(Thread):

    def __init__(self, queue):
        super().__init__()
        self.daemon = True
        self.queue = queue
        self.mouse = Mouse()
        self.is_callback_set = False

    def run(self):
        while True:
            frame = self.queue.get()
            self.mouse.draw(frame)
            cv.imshow('Frame', frame)
            if not self.is_callback_set:
                cv.setMouseCallback('Frame', self.mouse.callback)
                self.is_callback_set = True
            cv.waitKey(1)

    def selected(self):
        return self.mouse.finished

    def rectangle(self):
        first_slice = slice(self.mouse.first_point[1], self.mouse.second_point[1])
        second_slice = slice(self.mouse.first_point[0], self.mouse.second_point[0])
        return first_slice, second_slice


if __name__ == '__main__':
    #main()
    #third_main()
    #fourth_main()
    test_hand_detection()
    #test_rgb_to_ycb()
    #test_web_camera_thread()