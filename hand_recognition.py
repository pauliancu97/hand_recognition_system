import cv2 as cv
import numpy as np
import easygui
from math import sqrt, cos, sin, pi, atan2, acos, degrees, radians
from functools import partial
from random import randint
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from threading import Thread
from queue import Queue
from time import sleep
from numba import jit
from json import load
import random
import os
import re
import shutil


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
            cv.line(result, (sample_point[0], sample_point[1]), mask_point, (255, 0, 0), 1)
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
    return get_largest_connected_component(img_bin)


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


'''def get_maximum_radius(palm_point, contour):
    distances = [distance(palm_point, point) for point in contour]
    res = min(distances)
    return min(distances)'''


def get_maximum_radius(palm_point, contour):
    distances = contour - palm_point
    distances_prime = np.einsum('ij,ij->i', distances, distances)
    return np.sqrt(np.min(distances_prime))


'''def get_wrist_points(palm_mask_points):
    max_dist = 0
    max_index = 0
    for index in range(0, len(palm_mask_points) - 1):
        p1 = palm_mask_points[index]
        p2 = palm_mask_points[index + 1]
        if distance(p1, p2) > max_dist:
            max_dist = distance(p1, p2)
            max_index = index
    return palm_mask_points[max_index], palm_mask_points[max_index + 1]'''


def get_wrist_points(palm_mask_points):
    points = np.array(palm_mask_points, dtype=np.int32)
    first_sub_points = points[:-1]
    second_sub_points = points[1:]
    distances = first_sub_points - second_sub_points
    distances = np.einsum('ij,ij->i', distances, distances)
    index = np.argmax(distances)
    return palm_mask_points[index], palm_mask_points[index + 1]


'''def get_sampled_points(palm_point, radius):
    sampled_points = []
    for angle in range(0, 360):
        x = radius * cos(angle * pi / 180.0) + palm_point[0]
        y = radius * sin(angle * pi / 180.0) + palm_point[1]
        sampled_points.append((int(x), int(y)))
    return sampled_points'''


@jit(nopython=True)
def get_sampled_points(palm_point, radius):
    point = np.array(palm_point, dtype=np.int32)
    sampled_points = np.zeros((360, 2), dtype=np.int32)
    for angle in range(0, 360):
        sampled_points[angle, 0] = int(np.cos(np.radians(angle)) * radius)
        sampled_points[angle, 1] = int(np.sin(np.radians(angle)) * radius)
    '''sampled_points = np.fromfunction(lambda angle:
        np.array((radius * np.cos(angle * np.pi / 180.0), radius * np.sin(angle* np.pi / 180.0)), dtype=np.float32), (360,))'''
    sampled_points += point
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


@jit(nopython=True)
def get_finger_line(segmented_hand, stop):
    global_x_min = None
    global_x_max = None
    palm_line = None
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
    fifth_length = int((second_finger_point[0] - first_finger_point[0]) / 5 * 0.7)
    '''if thumb_index is not None:
        quarter_length = ((second_finger_point[0] - first_finger_point[0]) - fifth_length) // 4
    else:
        quarter_length = (second_finger_point[0] - first_finger_point[0]) // 4'''
    quarter_length = (second_finger_point[0] - first_finger_point[0]) // 4
    fingers_status = [False, False, False, False, False]
    fingers_status[0] = thumb_index is not None
    finger_lines = get_fingers_lines(fingers, thumb_index, first_finger_point, second_finger_point)
    for center, bottom_center in finger_lines:
        if thumb_index is not None:
            length = int(bottom_center[0]) - first_finger_point[0] - fifth_length
        else:
            length = int(bottom_center[0]) - first_finger_point[0]
        length = int(bottom_center[0]) - first_finger_point[0] - fifth_length
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
    maximum_radius = get_maximum_radius(np.array(palm_point, dtype=np.int32), np.array(contour, dtype=np.int32))
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
    cv.rectangle(segmented_hand, (0, int(minimum_row)), (segmented_hand.shape[1], segmented_hand.shape[0]), 0, -1)
    segmented_hand_no_arm = np.copy(segmented_hand)
    cv.fillPoly(segmented_hand, np.int32([palm_mask_points]), 0)
    mask_image = np.zeros(segmented_hand.shape, dtype=np.uint8)
    cv.fillPoly(mask_image, np.int32([palm_mask_points]), 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    fingers = get_fingers(segmented_hand)
    thumb_index = get_thumb_index(fingers, palm_point, first_wrist_point, second_wrist_point)
    '''first_finger_point, second_finger_point = get_finger_line(segmented_hand_no_arm, fingers, thumb_index)
    finger_lines = get_fingers_lines(fingers, thumb_index, first_finger_point, second_finger_point)
    fingers_status = get_fingers_status(fingers, thumb_index, first_finger_point, second_finger_point)'''
    for contour, _ in fingers:
        cv.drawContours(segmented_hand_no_arm, [contour], -1, 0, -1)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand_no_arm = cv.morphologyEx(segmented_hand_no_arm, cv.MORPH_OPEN, kernel)
    stop = segmented_hand_no_arm.shape[0]
    if thumb_index is not None:
        thumb_contour, center = fingers[thumb_index]
        rect = cv.minAreaRect(thumb_contour)
        box = cv.boxPoints(rect)
        box = sorted(box, key=lambda t: t[1])
        box = [np.array(point, dtype=np.int32) for point in box]
        stop = int(box[2][1] * 0.9)
    first_finger_point, second_finger_point = get_finger_line(segmented_hand_no_arm, stop)
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
    maximum_radius = get_maximum_radius(np.array(palm_point, dtype=np.int32), np.array(contour, dtype=np.int32))
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
    cv.rectangle(segmented_hand, (0, int(minimum_row)), (segmented_hand.shape[1], segmented_hand.shape[0]), 0, -1)
    segmented_hand_no_arm = np.copy(segmented_hand)
    cv.fillPoly(segmented_hand, np.int32([palm_mask_points]), 0)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    fingers = get_fingers(segmented_hand)
    thumb_index = get_thumb_index(fingers, palm_point, first_wrist_point, second_wrist_point)
    for contour, _ in fingers:
        cv.drawContours(segmented_hand_no_arm, [contour], -1, 0, -1)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand_no_arm = cv.morphologyEx(segmented_hand_no_arm, cv.MORPH_OPEN, kernel)
    stop = segmented_hand_no_arm.shape[0]
    if thumb_index is not None:
        thumb_contour, center = fingers[thumb_index]
        rect = cv.minAreaRect(thumb_contour)
        box = cv.boxPoints(rect)
        box = sorted(box, key=lambda t: t[1])
        box = [np.array(point, dtype=np.int32) for point in box]
        stop = int(box[2][1])
    first_finger_point, second_finger_point = get_finger_line(segmented_hand_no_arm, stop)
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


class BackgroundSubtractor:

    def __init__(self, width, height, num_background_frames=10):
        self.width = width
        self.height = height
        self.num_background_frames = num_background_frames
        self.current_num_background_frames = 0
        self.background_accumulator = np.zeros((height, width, 3), dtype=np.int64)
        self.background = np.zeros((height, width), dtype=np.uint8)
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

    def is_initialized(self):
        return self.current_num_background_frames >= self.num_background_frames

    def update(self, frame):
        frame_int64 = np.array(frame, dtype=np.int64)
        self.background_accumulator += frame_int64
        self.current_num_background_frames += 1
        if self.is_initialized():
            self.background = self.background_accumulator // self.num_background_frames
            self.background = np.array(self.background, dtype=np.uint8)
            self.background = cv.cvtColor(self.background, cv.COLOR_BGR2GRAY)
            self.background = cv.GaussianBlur(self.background, (7, 7), 0)

    def foreground_mask(self, frame):
        foreground = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        foreground = cv.GaussianBlur(foreground, (7, 7), 0)
        foreground = cv.absdiff(foreground, self.background)
        _, foreground = cv.threshold(foreground, 25, 255, cv.THRESH_BINARY)
        foreground = cv.morphologyEx(foreground, cv.MORPH_CLOSE, self.kernel)
        return foreground


class HandDetector:

    def __init__(self, config_path='cross-hands-tiny-prn.cfg', weights_path='cross-hands-tiny-prn.weights',
                 confidence=0.5, threshold=0.3):
        self.yolo = Yolo(config_path, weights_path, confidence, threshold)
        self.h_bins = 8
        self.s_bins = 12
        self.h_range = [0, 180]
        self.s_range = [0, 256]
        self.histogram = None
        self.kernel_7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        self.kernel_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        self.tracker_box = None
        self.tracker = cv.cv2.TrackerCSRT_create()
        self.last_frame = None

    def detect(self, frame, foreground_mask):
        self.last_frame = frame
        boxes = self.yolo.match(frame)
        if len(boxes) == 0:
            return None
        hand_box = boxes[0]
        self.tracker_box = hand_box
        hand_box = resize_bounding_box(hand_box, 1.7)
        hand_frame_roi = get_box_roi(frame, hand_box)
        hand_foreground_roi = get_box_roi(foreground_mask, hand_box)
        hand_frame_roi = cv.cvtColor(hand_frame_roi, cv.COLOR_BGR2HSV)
        self.histogram = cv.calcHist([hand_frame_roi], [0, 1], hand_foreground_roi, [self.h_bins, self.s_bins],
                                     [self.h_range[0], self.h_range[1], self.s_range[0], self.s_range[1]])
        cv.normalize(self.histogram, self.histogram, 0, 255, cv.NORM_MINMAX)
        back_projection = cv.calcBackProject([hand_frame_roi], [0, 1], self.histogram,
                                             [self.h_range[0], self.h_range[1], self.s_range[0], self.s_range[1]], 1)
        back_projection = cv.filter2D(back_projection, -1, self.kernel_3)
        _, back_projection = cv.threshold(back_projection, 250, 255, cv.THRESH_BINARY)
        back_projection = cv.morphologyEx(back_projection, cv.MORPH_CLOSE, self.kernel_7)
        back_projection = get_largest_connected_component(back_projection)
        return hand_box, back_projection

    def capture(self):
        self.tracker.init(self.last_frame, self.tracker_box)

    def track(self, frame):
        success, hand_box = self.tracker.update(frame)
        if not success:
            return None
        hand_box = resize_bounding_box(hand_box, 1.7)
        hand_frame_roi = get_box_roi(frame, hand_box)
        hand_frame_roi = cv.cvtColor(hand_frame_roi, cv.COLOR_BGR2HSV)
        back_projection = cv.calcBackProject([hand_frame_roi], [0, 1], self.histogram,
                                             [self.h_range[0], self.h_range[1], self.s_range[0], self.s_range[1]], 1)
        back_projection = cv.filter2D(back_projection, -1, self.kernel_3)
        _, back_projection = cv.threshold(back_projection, 250, 255, cv.THRESH_BINARY)
        back_projection = cv.morphologyEx(back_projection, cv.MORPH_CLOSE, self.kernel_7)
        back_projection = get_largest_connected_component(back_projection)
        return hand_box, back_projection


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
                        roi = np.flip(roi, axis=1)
                        segmented_hand = segment_hand(roi, hsv_values)
                        cv.imshow('Foreground', segmented_hand)
                        #equalized_histogram = equalize_histogram(roi)
                        #segmented_hand = segment_hand(equalized_histogram, hsv_values)
                        #cv.imshow('Segmented', segmented_hand)
                        if key_code == ord('S') or key_code == ord('s'):
                            hand_clear = True
                    else:
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


def main(is_gray=False):
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
    if is_gray:
        segmented_hand = get_largest_connected_component(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    else:
        segmented_hand = segment_hand(equalized_histogram, hsv_values)
    cv.imwrite('Mask.jpg', segmented_hand)
    finger_status = get_hand_attributes_with_visualisations(segmented_hand)
    print(finger_status)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def main_gray():
    image_name = easygui.fileopenbox()
    img = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
    _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.namedWindow(ORIGINAL_IMAGE_WINDOW)
    cv.imshow(ORIGINAL_IMAGE_WINDOW, img)
    segmented_hand = get_largest_connected_component(img)
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


class Yolo:

    def __init__(self, config_path='cross-hands-tiny-prn.cfg', weights_path='cross-hands-tiny-prn.weights',
                 confidence=0.5, threshold=0.3):
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
    yolo = Yolo('cross-hands-tiny-prn.cfg', 'cross-hands-tiny-prn.weights', 0.5, 0.3)
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
                    cv.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (255, 0, 0), 2)
            cv.imshow('Frame', frame)
            if roi is not None:
                cv.imshow('Roi', roi)
            if finger_status is not None:
                print(finger_status)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


def resize_bounding_box(box, factor):
    x, y, width, height = box
    center_x = x + width / 2
    center_y = y + height / 2
    width *= factor
    height *= factor
    x = center_x - width / 2
    y = center_y - height / 2
    return int(x), int(y), int(width), int(height)


def get_box_roi(img, box):
    rows, cols = img.shape[:2]
    x, y, width, height = box
    min_x = 0 if x < 0 else (cols if x > cols else x)
    max_x = 0 if (x + width) < 0 else (cols if (x + width) > cols else (x + width))
    min_y = 0 if y < 0 else (rows if y > rows else y)
    max_y = 0 if (y + height) < 0 else (rows if (y + height) > rows else (y + height))
    return img[min_y: max_y, min_x: max_x]


def get_largest_connected_component(img):
    num_of_labels, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8, ltype=cv.CV_32S)
    if num_of_labels == 1:
        return labels
    largest_label = max(range(1, num_of_labels), key=lambda label: stats[label, cv.CC_STAT_AREA])
    result = np.where(labels == largest_label, 255, 0)
    result = np.array(result, dtype=np.uint8)
    return result


def fill_holes(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, 255, -1)
    return img


def test_camera_subtraction():
    NUM_BACKGROUND_FRAMES = 10
    yolo = Yolo('cross-hands-tiny-prn.cfg', 'cross-hands-tiny-prn.weights', 0.5, 0.3)
    hand_box = None
    current_background_frames = 0
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        return
    ret, frame = camera.read()
    if not ret:
        return
    rows, cols = (frame.shape[0], frame.shape[1])
    background_acc = np.zeros((rows, cols, 3), dtype=np.int64)
    background = None
    background_color = None
    foreground = None
    hist = None
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    h_bins = 8
    s_bins = 12
    h_range = [0, 180]
    s_range = [0, 256]
    while camera.isOpened():
        key_code = cv.waitKey(1) & 0xFF
        ret, frame = camera.read()
        if ret:
            boxes = yolo.match(frame)
            if len(boxes) > 0:
                hand_box = boxes[0]
            else:
                hand_box = None
            display_frame = np.copy(frame)
            if hand_box is not None:
                x, y, w, h = hand_box
                cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255))
                if hist is not None:
                    resized_hand_box = resize_bounding_box(hand_box, 1.7)
                    hand_roi = get_box_roi(frame, resized_hand_box)
                    hand_roi = cv.cvtColor(hand_roi, cv.COLOR_BGR2HSV)
                    back_projection = cv.calcBackProject([hand_roi], [0, 1], hist,
                                                         [h_range[0], h_range[1], s_range[0], s_range[1]], scale=1)
                    back_projection_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                    cv.filter2D(back_projection, -1, back_projection_kernel, back_projection)
                    _, back_projection = cv.threshold(back_projection, 250, 255, cv.THRESH_BINARY)
                    back_projection = cv.morphologyEx(back_projection, cv.MORPH_CLOSE, kernel)
                    hand = get_largest_connected_component(back_projection)
                    cv.imshow('Back Projection', hand)
            cv.imshow('Frame', display_frame)
            if background is None:
                if current_background_frames < NUM_BACKGROUND_FRAMES:
                    current_background_frames += 1
                    frame_int64 = np.array(frame, dtype=np.int64)
                    background_acc += frame_int64
                else:
                    background = background_acc / NUM_BACKGROUND_FRAMES
                    background_color = np.array(background, dtype=np.uint8)
                    background = cv.cvtColor(background_color, cv.COLOR_BGR2GRAY)
                    background = cv.GaussianBlur(background, (7, 7), 0)
            else:
                foreground = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                foreground = cv.GaussianBlur(foreground, (7, 7), 0)
                foreground = cv.absdiff(foreground, background)
                _, foreground = cv.threshold(foreground, 25, 255, cv.THRESH_BINARY)
                foreground = cv.morphologyEx(foreground, cv.MORPH_CLOSE, kernel)
                cv.imshow('Foreground', foreground)
                cv.imshow('Bacground', background_color)
            if key_code == ord('s'):
                if (hist is None) and (foreground is not None) and (hand_box is not None):
                    hand_box = resize_bounding_box(hand_box, 1.7)
                    frame_roi = get_box_roi(frame, hand_box)
                    foreground_roi = get_box_roi(foreground, hand_box)
                    frame_roi = cv.cvtColor(frame_roi, cv.COLOR_BGR2HSV)
                    hist = cv.calcHist([frame_roi], [0, 1], foreground_roi, [h_bins, s_bins],
                                       [h_range[0], h_range[1], s_range[0], s_range[1]])
                    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        if key_code == ord('q'):
            break


class Buffer:

    def __init__(self, size):
        self.size = size
        self.num_of_elements = 0
        self.elements = []

    def is_full(self):
        return self.num_of_elements >= self.size

    def add(self, element):
        if not self.is_full():
            self.elements.append(element)
            self.num_of_elements += 1
        else:
            self.elements = self.elements[1:]
            self.elements.append(element)

    def get_majority_element(self):
        if not self.is_full():
            return None
        else:
            frequencies = dict()
            for element in self.elements:
                if element in frequencies:
                    frequencies[element] += 1
                else:
                    frequencies[element] = 0
            return max(frequencies.items(), key=lambda it: it[1])[0]


def test_web_camera_thread():
    queue = Queue()
    web_camera_thread = WebCameraThread()
    visualisation_thread = VisualisationThread(queue)
    web_camera_thread.init()
    web_camera_thread.start()
    visualisation_thread.start()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    buffer = Buffer(10)
    tick_mark = cv.getTickCount()
    frame = None
    roi_frame = None
    finger_status = None
    while True:
        frame = web_camera_thread.frame()
        if not visualisation_thread.selected():
            sleep(0.01)
        if visualisation_thread.selected():
            roi = visualisation_thread.rectangle()
            roi_frame = frame[roi]
            roi_frame = segment_hand_ycrcb(roi_frame)
            roi_frame = np.flip(roi_frame, axis=1)
            try:
                finger_status = get_hand_attributes(roi_frame)
                buffer.add(tuple(finger_status))
                finger_status = buffer.get_majority_element()
            except Exception:
                pass
            fps = cv.getTickFrequency() / (cv.getTickCount() - tick_mark)
            tick_mark = cv.getTickCount()
            print('FPS: {}'.format(fps))
        queue.put((frame, roi_frame, finger_status))


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

    def shape(self):
        return self.current_frame.shape[:2]


class VisualisationThread(Thread):

    def __init__(self, queue):
        super().__init__()
        self.daemon = True
        self.queue = queue
        self.mouse = Mouse()
        self.is_callback_set = False

    def run(self):
        while True:
            frame, roi, finger_status = self.queue.get()
            if finger_status is not None:
                frame = show_finger_status(finger_status, frame)
            self.mouse.draw(frame)
            cv.imshow('Frame', frame)
            if roi is not None:
                cv.imshow('ROI', roi)
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


class DisplayThread(Thread):

    def __init__(self, queue):
        super().__init__()
        self.daemon = True
        self.queue = queue
        self.key = None

    def get_key(self):
        return self.key

    def run(self):
        while True:
            self.key = cv.waitKey(1) & 0xFF
            data = self.queue.get()
            current_state = data['current_state']
            frame = data['frame']
            camera_img = np.copy(frame)
            hand_mask = None if 'hand_mask' not in data else data['hand_mask']
            hand_box = None if 'hand_box' not in data else data['hand_box']
            chosen_fingers_states = None if 'chosen_fingers_states' not in data else data['chosen_fingers_states']
            fingers_states = None if 'fingers_states' not in data else data['fingers_states']
            hand_image = None
            instructions_img = np.zeros((300, 800, 3), dtype=np.uint8)
            if current_state == BACKGROUND_INIT_STATE:
                message = 'Initializing background, please wait'
                (width, height), baseline = cv.getTextSize(message, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                height += baseline
                cv.putText(instructions_img, message, (0, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
            elif current_state == HAND_DETECTION_AND_SEGMENTATION_STATE:
                message = 'Raise your right hand, press \'S\' when your hand is visible'
                (width, height), baseline = cv.getTextSize(message, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                height += baseline
                cv.putText(instructions_img, message, (0, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
                if (hand_mask is not None) and (hand_box is not None):
                    hand_mask_bgr = cv.merge((hand_mask, hand_mask, hand_mask))
                    hand_roi = get_box_roi(frame, hand_box)
                    hand_image = cv.bitwise_and(hand_roi, hand_mask_bgr)
                    x, y, w, h = hand_box
                    cv.rectangle(camera_img, (x, y), (x + w, y + h), (0, 0, 255))
            elif current_state == SHOW_GESTURE_STATE:
                instruction = 'Replicate gesture'
                chosen_fingers_states_text = 'Desired - Thumb: {}; Index: {}; Middle: {}; Ring: {}; Little: {}'.format(
                    *['Up' if finger_state else 'Down' for finger_state in chosen_fingers_states])
                fingers_states_text = 'Current - Thumb: {}; Index: {}; Middle: {}; Ring: {}; Little: {}'.format(
                    *['Up' if finger_state else 'Down' for finger_state in fingers_states])
                (width_instruction, height_instruction), baseline = cv.getTextSize(instruction, cv.FONT_HERSHEY_SIMPLEX,
                                                                                   0.7, 1)
                height_instruction += baseline
                (width_chosen, height_chosen), baseline = cv.getTextSize(chosen_fingers_states_text,
                                                                         cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                height_chosen += baseline
                (width_current, height_current), baseline = cv.getTextSize(fingers_states_text, cv.FONT_HERSHEY_SIMPLEX,
                                                                           0.7, 1)
                height_current += baseline
                cv.putText(instructions_img, instruction, (0, height_instruction), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 0, 0))
                cv.putText(instructions_img, fingers_states_text, (0, height_instruction + height_current),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
                cv.putText(instructions_img, chosen_fingers_states_text,
                           (0, height_instruction + height_current + height_chosen), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 0, 0), 1)
                if (hand_mask is not None) and (hand_box is not None):
                    hand_mask_bgr = cv.merge((hand_mask, hand_mask, hand_mask))
                    hand_roi = np.flip(get_box_roi(frame, hand_box), axis=1)
                    hand_image = cv.bitwise_and(hand_roi, hand_mask_bgr)
                    x, y, w, h = hand_box
                    cv.rectangle(camera_img, (x, y), (x + w, y + h), (255, 0, 0))
            elif current_state == SUCCESS_STATE:
                message = 'Success'
                (width, height), baseline = cv.getTextSize(message, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                height += baseline
                cv.putText(instructions_img, message, (0, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
            cv.imshow('Camera', camera_img)
            cv.imshow('Instructions', instructions_img)
            if hand_image is not None:
                cv.imshow('Hand', hand_image)


def test_background_subtractor():
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        return
    ret, frame = camera.read()
    if not ret:
        return
    height, width = frame.shape[:2]
    background_subtractor = BackgroundSubtractor(width, height)
    hand_detector = HandDetector()
    foreground = None
    hand_box = None
    hand_mask = None
    captured = False
    while camera.isOpened():
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('s') or key == ord('S'):
            hand_detector.capture()
            captured = True
        ret, frame = camera.read()
        if ret:
            if not background_subtractor.is_initialized():
                background_subtractor.update(frame)
            else:
                foreground = background_subtractor.foreground_mask(frame)
                detection = hand_detector.detect(frame, foreground) if not captured else hand_detector.track(frame)
                if detection is not None:
                    hand_box, hand_mask = detection
                else:
                    hand_box, hand_mask = (None, None)
            if hand_box is not None:
                x, y, w, h = hand_box
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
            if hand_mask is not None:
                cv.imshow('Hand', hand_mask)
            cv.imshow('Camera', frame)
            if foreground is not None:
                cv.imshow('Foreground', foreground)


BACKGROUND_INIT_STATE = 0
HAND_DETECTION_AND_SEGMENTATION_STATE = 1
CHOOSE_RANDOM_FINGERS_STATE = 2
SHOW_GESTURE_STATE = 3
SUCCESS_STATE = 4
NUM_SUCCESS_FRAMES = 180


def test():
    with open('fingers.json') as file:
        json = load(file)
        read_fingers_states = json['states']
        read_fingers_states = [tuple(states) for states in read_fingers_states]
    web_camera_thread = WebCameraThread()
    web_camera_thread.init()
    web_camera_thread.start()
    height, width = web_camera_thread.shape()
    background_subtractor = BackgroundSubtractor(width, height, 300)
    buffer = Buffer(15)
    hand_detector = HandDetector()
    current_state = BACKGROUND_INIT_STATE
    hand_box = None
    hand_mask = None
    num_success_frames = 0
    chosen_fingers_states = tuple([False] * 5)
    fingers_states = tuple([False] * 5)
    data = dict()
    queue = Queue()
    display_thread = DisplayThread(queue)
    display_thread.start()
    while True:
        cv.waitKey(1)
        key = display_thread.get_key()
        frame = web_camera_thread.frame()
        if key == ord('q') or key == ord('Q'):
            break
        if current_state == BACKGROUND_INIT_STATE:
            background_subtractor.update(frame)
            if background_subtractor.is_initialized():
                current_state = HAND_DETECTION_AND_SEGMENTATION_STATE
        elif current_state == HAND_DETECTION_AND_SEGMENTATION_STATE:
            foreground = background_subtractor.foreground_mask(frame)
            detection = hand_detector.detect(frame, foreground)
            if detection is None:
                hand_box, hand_mask = (None, None)
            else:
                hand_box, hand_mask = detection
            if key == ord('s') or key == ord('S'):
                hand_detector.capture()
                current_state = CHOOSE_RANDOM_FINGERS_STATE
        elif current_state == CHOOSE_RANDOM_FINGERS_STATE:
            chosen_fingers_states = random.choice([state for state in read_fingers_states
                                                   if state != chosen_fingers_states])
            current_state = SHOW_GESTURE_STATE
        elif current_state == SHOW_GESTURE_STATE:
            detection = hand_detector.track(frame)
            if detection is not None:
                hand_box, hand_mask = detection
            else:
                hand_box, hand_mask = (None, None)
            if (hand_box is not None) and (hand_mask is not None):
                hand_mask = np.flip(hand_mask, axis=1)
                try:
                    fingers_states = tuple(get_hand_attributes(hand_mask))
                except Exception:
                    pass
                buffer.add(fingers_states)
                fingers_states = buffer.get_majority_element()
                fingers_states = fingers_states if fingers_states is not None else [False] * 5
                if fingers_states == chosen_fingers_states:
                    current_state = SUCCESS_STATE
        elif current_state == SUCCESS_STATE:
            num_success_frames += 1
            if num_success_frames == NUM_SUCCESS_FRAMES:
                num_success_frames = 0
                current_state = CHOOSE_RANDOM_FINGERS_STATE
        data['current_state'] = current_state
        data['frame'] = frame
        data['hand_mask'] = hand_mask
        data['hand_box'] = hand_box
        data['chosen_fingers_states'] = chosen_fingers_states
        data['fingers_states'] = fingers_states
        queue.put(data)


BACKGROUND_INIT_COLLECT_STATE = 0
DETECT_SEGMENT_COLLECT_STATE = 1
TRACK_IDLE_COLLECT_STATE = 2
TRACK_RECORD_INIT_COLLECT_STATE = 3
TRACK_RECORD_COLLECT_STATE = 4


CONTROLS_WIN_NAME = 'Controls'
RECORD = 'Record'
THUMB = 'Thumb'
INDEX = 'Index'
MIDDLE = 'Middle'
RING = 'Ring'
LITTLE = 'Little'


def nothing(x):
    pass


def collect_test_images():
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        return
    ret, frame = camera.read()
    if not ret:
        return
    height, width = frame.shape[:2]
    background_subtractor = BackgroundSubtractor(width, height, 180)
    hand_detector = HandDetector()
    hand_box = None
    hand_mask = None
    hand_state = tuple([False] * 5)
    frames_per_state = dict()
    current_state = BACKGROUND_INIT_COLLECT_STATE
    while camera.isOpened():
        key = cv.waitKey(1) & 0xFF
        ret, frame = camera.read()
        if ret:
            if current_state == BACKGROUND_INIT_COLLECT_STATE:
                background_subtractor.update(frame)
                if background_subtractor.is_initialized():
                    current_state = DETECT_SEGMENT_COLLECT_STATE
                    print('Initialized background')
            elif current_state == DETECT_SEGMENT_COLLECT_STATE:
                foreground = background_subtractor.foreground_mask(frame)
                detection = hand_detector.detect(frame, foreground)
                if detection is None:
                    hand_box = None
                    hand_mask = None
                else:
                    hand_box, hand_mask = detection
                if key == ord('s') or key == ord('S'):
                    current_state = TRACK_IDLE_COLLECT_STATE
                    hand_detector.capture()
                    cv.namedWindow(CONTROLS_WIN_NAME)
                    cv.createTrackbar(RECORD, CONTROLS_WIN_NAME, 0, 1, nothing)
                    cv.createTrackbar(THUMB, CONTROLS_WIN_NAME, 0, 1, nothing)
                    cv.createTrackbar(INDEX, CONTROLS_WIN_NAME, 0, 1, nothing)
                    cv.createTrackbar(MIDDLE, CONTROLS_WIN_NAME, 0, 1, nothing)
                    cv.createTrackbar(RING, CONTROLS_WIN_NAME, 0, 1, nothing)
                    cv.createTrackbar(LITTLE, CONTROLS_WIN_NAME, 0, 1, nothing)
                    print('Captured hand')
            elif current_state == TRACK_IDLE_COLLECT_STATE:
                detection = hand_detector.track(frame)
                if detection is None:
                    hand_box = None
                    hand_mask = None
                else:
                    hand_box, hand_mask = detection
                is_record_on = cv.getTrackbarPos(RECORD, CONTROLS_WIN_NAME) == 1
                if is_record_on:
                    current_state = TRACK_RECORD_COLLECT_STATE
                    thumb_state = cv.getTrackbarPos(THUMB, CONTROLS_WIN_NAME) == 1
                    index_state = cv.getTrackbarPos(INDEX, CONTROLS_WIN_NAME) == 1
                    middle_state = cv.getTrackbarPos(MIDDLE, CONTROLS_WIN_NAME) == 1
                    ring_state = cv.getTrackbarPos(RING, CONTROLS_WIN_NAME) == 1
                    little_state = cv.getTrackbarPos(LITTLE, CONTROLS_WIN_NAME) == 1
                    hand_state = (thumb_state, index_state, middle_state, ring_state, little_state)
                    if hand_state not in frames_per_state:
                        frames_per_state[hand_state] = []
                    print('Recording frames...')
            elif current_state == TRACK_RECORD_COLLECT_STATE:
                detection = hand_detector.track(frame)
                if detection is None:
                    hand_box = None
                    hand_mask = None
                else:
                    hand_box, hand_mask = detection
                if hand_mask is not None:
                    frames_per_state[hand_state].append(hand_mask)
                is_record_on = cv.getTrackbarPos(RECORD, CONTROLS_WIN_NAME) == 1
                if not is_record_on:
                    current_state = TRACK_IDLE_COLLECT_STATE
                    print('Finished recording frames')
            #Visual
            if hand_box is not None:
                x, y, w, h = hand_box
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if hand_mask is not None:
                cv.imshow('Hand', hand_mask)
            cv.imshow('Camera', frame)
        if key == ord('q') or key == ord('Q'):
            break
    if frames_per_state:
        for hand_state, frames in frames_per_state.items():
            repr_state = '_'.join(['up' if it else 'down' for it in hand_state])
            if not os.path.exists(repr_state):
                os.mkdir(repr_state)
            for index, frame in enumerate(frames):
                path = '{}/{}.jpg'.format(repr_state, index)
                cv.imwrite(path, frame)


def get_accuracy():
    if os.path.exists('defects'):
        shutil.rmtree('defects')
    os.mkdir('defects')
    regular_expression = re.compile('(?:up|down)_(?:up|down)_(?:up|down)_(?:up|down)_(?:up|down)')
    test_images_dirs = []
    for path in os.listdir(os.getcwd()):
        if os.path.isdir(path) and regular_expression.match(path):
            test_images_dirs.append(path)
    total_tests = 0
    correct_tests = 0
    for test_images_dir in test_images_dirs:
        test_path = os.path.join(os.getcwd(), test_images_dir)
        for file_name in os.listdir(test_path):
            total_tests += 1
            file_path = os.path.join(test_path, file_name)
            image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            image = np.flip(image, axis=1)
            _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            image = fill_holes(image)
            hand_state = [False] * 5
            try:
                hand_state = get_hand_attributes(image)
            except Exception:
                pass
            repr_hand_state = '_'.join(['up' if it else 'down' for it in hand_state])
            if repr_hand_state == test_images_dir:
                correct_tests += 1
            else:
                new_file_name = '{}_{}'.format(test_images_dir, file_name)
                new_file_path = os.path.join('defects', new_file_name)
                cv.imwrite(new_file_path, image)
    accuracy = correct_tests / total_tests
    print('Accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    #main()
    #main_gray()
    third_main()
    #test_hand_detection()
    #test_rgb_to_ycb()
    #test_web_camera_thread()
    #test_camera_subtraction()
    #test_background_subtractor()
    #test()
    #collect_test_images()
    #get_accuracy()