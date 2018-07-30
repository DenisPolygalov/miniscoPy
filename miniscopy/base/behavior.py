#!/usr/bin/env python3

import os, sys
import cv2
import numpy as np
import warnings

"""
Copyright (C) 2018 Lilia Evgeniou,
Laboratory for Circuit and Behavioral Physiology,
RIKEN Center for Brain Science, Saitama, Japan.
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, a copy is available at
http://www.fsf.org/
"""

"""
An example of parameters for detection (for config.yaml file):
behavior:
  # Blob Detector parameters

  # Change thresholds
  blobdet_minThreshold: 10
  blobdet_maxThreshold: 200

  # Filter by Area
  blobdet_filterByArea: True
  blobdet_minArea: 300
  blobdet_maxArea: 700

  # Filter by Circularity
  blobdet_filterByCircularity: True
  blobdet_minCircularity: 0.1

  # Filter by Convexity
  blobdet_filterByConvexity: True
  blobdet_minConvexity: 0.5

  # Filter by Inertia
  blobdet_filterByInertia: False
  blobdet_minInertiaRatio: 0.8

  # Filter by light or dark color
  blobdet_filterByColor: True
  blobdet_blobColor: 255  # finds light colored blobs

  # detect_position method minmax of size of blob made by LED
  detpos_min_keypoint_sz: 15  # should be changed if different size blob
  detpos_max_keypoint_sz: 35

  # color channel to view gray scale - used for detection of position; can be 0, 1, or 2
  color_channel: 2
"""


class CBehavPositionDetector(object):
    def __init__(self, d_param):
        oc_blob_detector_params = cv2.SimpleBlobDetector_Params()

        # include parameters from yaml file

        # Threshold
        oc_blob_detector_params.minThreshold      = d_param["blobdet_minThreshold"]
        oc_blob_detector_params.maxThreshold      = d_param["blobdet_maxThreshold"]

        # Area
        oc_blob_detector_params.filterByArea      = d_param["blobdet_filterByArea"]
        oc_blob_detector_params.minArea           = d_param["blobdet_minArea"]
        oc_blob_detector_params.maxArea           = d_param["blobdet_maxArea"]
        # Circularity
        oc_blob_detector_params.filterByCircularity = d_param["blobdet_filterByCircularity"]
        oc_blob_detector_params.minCircularity      = d_param["blobdet_minCircularity"]
        # Convexity
        oc_blob_detector_params.filterByConvexity = d_param["blobdet_filterByConvexity"]
        oc_blob_detector_params.minConvexity      = d_param["blobdet_minConvexity"]
        # Inertia
        oc_blob_detector_params.filterByInertia   = d_param["blobdet_filterByInertia"]
        oc_blob_detector_params.minInertiaRatio   = d_param["blobdet_minInertiaRatio"]
        # Color
        oc_blob_detector_params.filterByColor     = d_param["blobdet_filterByColor"]
        oc_blob_detector_params.blobColor         = d_param["blobdet_blobColor"]

        self.oc_blob_detector = cv2.SimpleBlobDetector_create(oc_blob_detector_params)

        self.i_min_kp_sz = d_param["detpos_min_keypoint_sz"]
        self.i_max_kp_sz = d_param["detpos_max_keypoint_sz"]

        # used for detection
        self.i_c_channel = d_param["color_channel"]  # = 2

        self.l_call_cnt = []
        self.l_kp_id    = []
        self.l_kp_size  = []
        self.l_position = []

        self.i_call_cnt = 0
    #


    def detect_position(self, na_input, b_verbose=False):
        """
        Detects LED light where mouse is in behaviour videos
        by detecting blob on "red" gray scale image.
        :param na_input: numpy array of frame image
        :param detector: cv2.SimpleBlobDetector whose parameters are already defined
        :return: the best estimated coordinates of the center of the LED light circle.
                 (None, None) if no LED light detected
        Function also changes self.l_call_cnt, self.l_kp_id, self.l_kp_size,
        self.l_position, and self.i_call_cnt (adds 1 to it).
        - appends all possible keypoints found into lists.
        Later on, will make ndarray with a matrix as follows:
            ______________________________________________________
            | l_call_cnt | l_kp_id | l_kp_size | x_coor | y_coor |
            |     0      |    0    |   20.4    |  15.8  |  19.3  | <- only 1 kp found
            |     1      |    0    |   25.2    |  12.4  |  20.5  | <- 2 kps found
            |     1      |    1    |   27.5    |  50.3  |  94.1  |
            |     2      |    0    |   NaN     |  NaN   |  NaN   | <- handles no kp found
            .            .         .           .        .        .
            .            .         .           .   l_position    .
            .            .         .           .        .        .
        """

        # make the LED stand out brighter by converting to red gray scale
        # finds key points
        keypoints = self.oc_blob_detector.detect(na_input[:, :, self.i_c_channel])

        # if no key points are found, will append what is needed into lists,
        # and will return (None, None)
        if len(keypoints) == 0:
            if b_verbose:
                print("WARNING: failed to detect any key points. Check your detection parameters.")
            self.l_call_cnt.append(self.i_call_cnt)
            self.l_kp_id.append(0)
            self.l_kp_size.append(np.nan)
            self.l_position.append([np.nan, np.nan])
            # add 1 to call_cnt, which keeps track of frame number
            self.i_call_cnt += 1
            return (None, None)

        if b_verbose:
            for idx_kp, kp in enumerate(keypoints):
                print("call_id=%i\tkeypoint: id=%i\tsize=%.3f\tx=%.3f\ty=%.3f" % \
                      (self.i_call_cnt, idx_kp, kp.size, kp.pt[0], kp.pt[1]))

        # keeps one set of estimated coordinates (in this case, first keypoint)
        kp_xcoor, kp_ycoor = None, None
        # finds keypoints of right size,
        # and appends all into lists if more than one found.
        for idx_kp, kp in enumerate(keypoints):
            if self.i_min_kp_sz < kp.size < self.i_max_kp_sz:
                # keeps set of estimated coordinates
                if idx_kp == 0:
                    kp_xcoor = kp.pt[0]
                    kp_ycoor = kp.pt[1]
                # appends to lists
                self.l_call_cnt.append(self.i_call_cnt)
                self.l_kp_id.append(idx_kp)
                self.l_kp_size.append(kp.size)
                self.l_position.append([kp.pt[0], kp.pt[1]])

        # returning when keypoints found
        if (kp_xcoor, kp_ycoor) != (None, None):
            # add 1 to call_cnt, which keeps track of frame number
            self.i_call_cnt += 1
            return (kp_xcoor, kp_ycoor)

        # returning when a keypoint of the right size is not found.
        # appends what is needed to lists,
        # and returns (None, None)
        self.l_call_cnt.append(self.i_call_cnt)
        self.l_kp_id.append(0)
        self.l_kp_size.append(np.nan)
        self.l_position.append([np.nan, np.nan])
        # add 1 to call_cnt, which keeps track of frame number
        self.i_call_cnt   += 1
        return (None, None)
    #


    def draw_cross(self, t_xy, na_input, i_sz=10, t_color=(225, 255, 255)):
        """
        Modifies numpy array of image directly to draw a cross at coordinates t_xy

        * TO REMEMBER: The array's shape is organised into: (rows, columns, depth)
                       so the array's "coordinates" are (y, x)
                       Hence, coordinate values must be flipped.
                       In addition, point (0, 0) is at top left corner.
                       x values increase to the right ->
                       y values increase to the bottom |
                                                       v

        :param t_xy: coordinate tuple (x, y)
        :param na_input: numpy array of frame image
        :param i_sz: size of cross (in pixels)
        :param t_color: color of cross
        :return: this function will directly modify the numpy array
        """

        # handles when detector returns (None, None); returns unmodified image
        if len(t_xy) != 2:    return na_input
        if t_xy[0] == None:   return na_input
        if t_xy[1] == None:   return na_input
        if np.isnan(t_xy[0]): return na_input
        if np.isnan(t_xy[1]): return na_input

        # horizontal and vertical coordinates
        # e.g. (300, 50), but in numpy array, it is first 50 rows, then 300 cols
        i_hcoor, i_vcoor = map(int, t_xy)

        # horizontal and vertical size of array
        # na_input.shape = (101, 624, 3)
        i_vsz, i_hsz, *_ = na_input.shape  # i_vsz: vertical width of image (around 100),
                                           # i_hsz: horizontal length of image (around 600 pixels)

        # size of cross on top, bottom, left, right
        # if cross at the edge of image, modify to partial cross
        i_sz_t = i_vcoor if i_vcoor - i_sz < 0 else i_sz
        i_sz_b = abs(i_vsz - i_vcoor) if i_vcoor + i_sz > i_vsz else i_sz
        i_sz_l = i_hcoor if i_hcoor - i_sz < 0 else i_sz
        i_sz_r = abs(i_hsz - i_hcoor) if i_hcoor + i_sz > i_hsz else i_sz

        # modification of array image
        # na_input[row (vert), col (horiz), depth]
        na_input[i_vcoor - i_sz_t:i_vcoor + i_sz_b, i_hcoor, :] = t_color
        na_input[i_vcoor, i_hcoor - i_sz_l:i_hcoor + i_sz_r, :] = t_color

        return na_input
    #
#


def find_segments(na_x, f_thr_amp, i_thr_len):
    """
    Finds indices of 'na_x' where values of 'na_x' are
    equal or above the 'amplitude threshold' value 'f_thr_amp'
    AND equal or longer than the 'duration threshold' value 'i_thr_len'.
    :param na_x: ndarray of one dimension
    :param f_thr_amp: amplitude threshold value (float)
    :param i_thr_len: length threshold - must be specified in number
                      of elements of na_x (2,3,4, ... X.size - 1)
    :return: M row x 2 column ndarray of beginning and ending (+1) indices, -
             so called 'segments' where M is the number of segments.
             * To remember: end idx + 1 because Python range excludes last number
             ________________________
             | beg idx  | end idx  |    <- segment 1
             |          |          |    <- segment 2
             .          .          .    <- ...
    """

    # just in case, makes na_x in 1 "row" format: array([a, b, c, ...])
    na_x = np.squeeze(na_x)

    if len(na_x.shape) != 1:
        raise TypeError("find_segments: na_x must be a ndarray of single row or column")

    if np.isnan(na_x).any():
        raise TypeError("find_segments: na_x contain NaN values")

    if (na_x <= f_thr_amp).all():
        raise ValueError("find_segments: ﻿amplitude threshold (f_thr_amp) is too high")

    if (na_x >= f_thr_amp).all():
        raise ValueError("find_segments: ﻿amplitude threshold (f_thr_amp) is too low")

    i_thr_len = np.int(np.floor(i_thr_len))

    if i_thr_len < 2:
        raise ValueError("find_segments: length threshold (i_thr_len) is too short (less than 2)")

    if i_thr_len > (na_x.size - 1):
        raise ValueError("find_segments: ﻿length threshold (i_thr_len) is too long (longer than length(X) - 1)")

    # special case when thr_amp is exactly equal to min(x)
    if (na_x >= f_thr_amp).all():
        na_segments = np.array([0, len(na_x)], dtype=np.int64)
        warnings.warn("find_segments: all values are equal or above the f_thr_amp value")
        return na_segments

    # Ideally, we need to find indices of all continuous chunks of ones longer or equal to i_thr_len
    # In order to detect 0 -> 1 and 1 -> 0 transitions in x1, we use np.diff()

    na_x_gt_thr_amp_bool = na_x >= f_thr_amp  # boolean ndarray: where na_x is greater than f_thr_amp
    na_x_gt_thr_amp      = np.array(list(map(np.int, na_x_gt_thr_amp_bool)))  # boolean converted to 0s and 1s
    na_diff_x            = np.diff(na_x_gt_thr_amp)  # array of differences
                                                     # 0: no change; 1: increase; -1: decrease
    # example:
    # na_x:                  array([ 1,  3,  4,  6,  9, 10,  5,  3,  1])
    # f_thr_amp:             4
    # na_x_gt_thr_amp_bool:  array([False, False,  True,  True,  True,  True,  True, False, False])
    # na_x_gt_thr_amp:       array([0, 0, 1, 1, 1, 1, 1, 0, 0])
    # na_diff_x:             array([ 0,  1,  0,  0,  0,  0, -1,  0])

    # filter na_x by amplitude
    # slicing is done in the Python way: end is 1 more than the actual end index

    if   na_x[0] <  f_thr_amp and na_x[-1] <  f_thr_amp:  # 00111100 (for na_x_gt_thr_amp)
        na_begs = np.where(na_diff_x > 0)[0] + 1
        na_ends = np.where(na_diff_x < 0)[0] + 1

    elif na_x[0] >= f_thr_amp and na_x[-1] >= f_thr_amp:  # 11000011
        na_begs = np.append(0, np.where(na_diff_x > 0)[0] + 1)
        na_ends = np.append(np.where(na_diff_x < 0)[0] + 1, na_x.size)

    elif na_x[0] >= f_thr_amp and na_x[-1] <  f_thr_amp:  # 11011000
        na_begs = np.append(0, np.where(na_diff_x > 0)[0] + 1)
        na_ends = np.where(na_diff_x < 0)[0] + 1

    elif na_x[0] <  f_thr_amp and na_x[-1] >= f_thr_amp:  # 00011011
        na_begs = np.where(na_diff_x > 0)[0] + 1
        na_ends = np.append(np.where(na_diff_x < 0)[0] + 1, na_x.size)

    else:
        raise ValueError("find_segments: algorithm error (1)")  # should never happen

    if len(na_begs) != len(na_ends):
        raise ValueError("find_segments: algorithm error (2)")

    na_diff_I = np.where(((na_ends - na_begs + 1) >= (i_thr_len + 1)) == True)[0]
    na_segments = np.column_stack((na_begs[na_diff_I], na_ends[na_diff_I]))  # ndarray of shape (M, 2) where M = number of segments

    return na_segments.astype(np.int64)


if __name__ == '__main__':

    # os.sys.path.append("/Users/ path ... to your /miniscoPy")

    import cv2
    import yaml
    import matplotlib.pyplot as plt

    from miniscopy.base.ioutils import enum_video_files
    from miniscopy.base.mupamovie import CMuPaMovieCV
    from miniscopy.base.player import CSideBySidePlayer
    
    if len(sys.argv) < 3:
        print("ERROR: not enough input arguments. Usage:")
        print("%s folder_name_containing_video_files config.yaml" % sys.argv[0])
        sys.exit(0)
    #
    # make a tuple of strings - input file names
    t_files = enum_video_files(sys.argv[1], "behavCam*.avi")

    # load parameters
    d_param = yaml.load(open(sys.argv[2], "r"))

    # create a multi-part movie object
    oc_movie = CMuPaMovieCV(t_files)

    # create a side by side movie player
    oc_player = CSideBySidePlayer()
    
    # create position detector
    oc_pos_det = CBehavPositionDetector(d_param["behavior"])

    # check that position detector works
    i_brake = 0
    while(oc_movie.read_next_frame()):
        na_frame_out = oc_movie.na_frame.copy()

        # draw a cross at coordinates of LED directly onto na_frame_out
        t_cross_pos = oc_pos_det.detect_position(na_frame_out, b_verbose=True)
        oc_pos_det.draw_cross(t_cross_pos, na_frame_out)

        # set the frame data
        # left frame: original movie
        oc_player.set_Lframe_data(oc_movie.na_frame)
        # right frame: movie with detected LED
        oc_player.set_Rframe_data(na_frame_out)

        if oc_player.drawait(): break

        i_brake += 1
        if i_brake >= 500: break

    print(np.column_stack((oc_pos_det.l_call_cnt,
                           oc_pos_det.l_kp_id,
                           oc_pos_det.l_kp_size,
                           oc_pos_det.l_position)))

    # plot the x and y axes for all frames
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(np.array(oc_pos_det.l_position))
    plt.show()
    #
#
