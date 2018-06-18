#!/usr/bin/env python3

import os, sys, time
import collections
import numpy as np
import cv2

"""
Heavily based on:
miniscoPy (Guillaume Viejo):
https://github.com/PeyracheLab/miniscoPy
and CaImAn (Andrea Giovannucci et al.)
https://github.com/flatironinstitute/CaImAn
https://github.com/flatironinstitute/CaImAn/graphs/contributors
"""

"""
Copyright (C) 2018 Denis Polygalov,
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

def bin_median(na_input, i_win_sz=10, b_exclude_nans=True):
    """
    Compute median of 3D array in along axis o by binning values

    Parameters:
    ----------

    na_input: ndarray
        input 3D matrix, time along first dimension

    i_win_sz: int
        number of frames in a bin

    Returns:
    -------
    na_out: ndarray
        median image
    """

    T, d1, d2 = np.shape(na_input)
    if T < i_win_sz:
        i_win_sz = T
    num_windows = np.int(np.floor(T / i_win_sz)) # re-implementation of the old_div()
    num_frames = num_windows * i_win_sz
    if b_exclude_nans:
        na_out = np.nanmedian(np.nanmean(np.reshape(
            na_input[:num_frames], (i_win_sz, num_windows, d1, d2)), axis=0), axis=0)
    else:
        na_out = np.median(np.mean(np.reshape(
            na_input[:num_frames], (i_win_sz, num_windows, d1, d2)), axis=0), axis=0)
    return na_out
#

def bootstrap_template(oc_movie, i_tmpl_nframes, s_method="head", i_color_ch=0, b_verbose=False):
    """
    Read 'i_tmpl_nframes' frames from multi-part movie object 'oc_movie' 
    by using method 's_method'. Input frames will be converted to, and output is 
    returned as 3D array of (TIME x FRAME_HEIGHT x FRAME_WIDTH) shape and np.float32 type.
    Only single color/grayscale/np.float32 type of input supported.
    """
    i_max_nframes = oc_movie.df_info['frames'].sum()
    i_half_nframes = np.int(i_max_nframes/2)
    i_half_ntmpl   = np.int(i_tmpl_nframes/2)

    if i_tmpl_nframes >= i_max_nframes:
        raise ValueError("Requested template size(%d) is too big for this movie(%d)" % (i_tmpl_nframes,i_max_nframes) )

    if s_method == "head":
        na_indices = np.arange(i_tmpl_nframes)

    elif s_method == "tail":
        na_indices = np.arange(i_max_nframes - i_tmpl_nframes, i_max_nframes, 1)

    elif s_method == "middle":
        na_indices = np.arange(i_half_nframes - i_half_ntmpl, i_half_nframes + i_half_ntmpl, 1)

    elif s_method == "random":
        na_indices = np.random.randint(0, high=i_max_nframes, size=i_tmpl_nframes)

    else: raise ValueError("Unsupported method: %s" % s_method)

    oc_movie.read_frame(0)

    # make sure the shape of this array is (T x H x W) where
    # (H x W) is the shape of the frame and T is the time axis.
    na_template = np.zeros([na_indices.shape[0], oc_movie.na_frame.shape[0], oc_movie.na_frame.shape[1]], dtype=np.float32)

    if b_verbose:
        print("bootstrap_template: s_method=%s i_tmpl_nframes=%d i_color_ch=%d na_template.shape=%s" % \
            (s_method, i_tmpl_nframes, i_color_ch, repr(na_template.shape)) \
        )

    for tt, idx in enumerate(na_indices):

        oc_movie.read_frame(idx) # read the next requested frame
        
        if len(oc_movie.na_frame.shape) == 3:
            if i_color_ch >= oc_movie.na_frame.shape[2]:
                 raise ValueError("Color channel mismatch: i_color_ch=%d oc_movie.na_frame.shape=%s" % (i_color_ch, repr(oc_movie.na_frame.shape)))
            else:
                na_template[tt,:,:] = oc_movie.na_frame[:,:,i_color_ch].astype(np.float32)
        #
        elif len(oc_movie.na_frame.shape) == 2:
            na_template[tt,:,:] = oc_movie.na_frame[:,:].astype(np.float32)
        else:
            raise ValueError("Unsupported frame shape: %s" % repr(oc_movie.na_frame.shape))
        #
        # if b_verbose: print("bootstrap_template: %s" % oc_movie.get_frame_stat())
    #
    return na_template
#

def calc_squaring_margins(na_movie, b_verbose=False):
    """
    Return a tuple of two integers - best margins to
    convert the 'na_movie' array to square shape frame.
    input:  na_movie - (T x H x W) matrix of movie frames
    output: tuple(i_left_margin, i_right_margin) - best margins
            to slice the 'na_movie' to square (T x N x N) shape
            where N = min(H,W)
    """
    if len(na_movie.shape) != 3:
        raise ValueError("Unexpected shape: %s" % repr(na_movie.shape))
    #
    i_frame_number = 0
    oc_squaring_filter = CSquaringFilter(na_movie[i_frame_number,:,:].shape)
    while (i_frame_number < na_movie.shape[0]):
        oc_squaring_filter.process_frame(na_movie[i_frame_number,:,:], b_verbose=b_verbose)
        i_frame_number += 1
    #
    na_margins = np.array(oc_squaring_filter.l_all_margins, dtype=np.int32)
    return tuple( np.median(na_margins,axis=0).astype(np.int32) )
#

class CSquaringFilter(object):
    """
    Convert rectangular shape input frame into a square frame
    by cutting (left / right) or (top / bottom) margins.
    Margins to cut are calculated based in intensity changes along
    the longer size of the input frame. This can only be used for
    input data containing relatively bright FOV in the middle
    (not necessary center) of the frame surrounded by black sides.
    NOTE: only 2D arrays are acceptable as input!
    """
    def __init__(self, t_input_frame_shape):
        if len(t_input_frame_shape) != 2:
            raise ValueError("Unsupported frame shape: %s" % repr(t_input_frame_shape))

        nrows, ncols = t_input_frame_shape[0], t_input_frame_shape[1]
        self._t_init_shape = t_input_frame_shape

        if nrows < ncols:
            self.i_target_axis = 0
            self.i_target_sz = nrows
        elif nrows > ncols:
            self.i_target_axis = 1
            self.i_target_sz = ncols
        else: raise ValueError("Input is already a square")

        # main data exchange interface for this class
        self.t_curr_margins = None # a tuple consisting of two integers - margin width values
        self.l_all_margins  = []   # list of tuples, each tuple consist of self.t_curr_margins
        self.na_out = None
    #
    def _calc_margins(self, na_section):
        """
        'na_section' must be a 1D array (vector)
        'self.i_target_sz' must be less than na_section.size
        """
        # The left(top) and right(bottom) side margins
        i_Lm = 0
        i_Rm = na_section.size - 1 # point to the last element in the na_section vector
        while ((i_Rm - i_Lm) > self.i_target_sz):
            i_dLm = np.int32(na_section[i_Lm+1]) - np.int32(na_section[i_Lm])
            i_dRm = np.int32(na_section[i_Rm-1]) - np.int32(na_section[i_Rm])
            if i_dLm >= i_dRm: i_Rm -= 1
            if i_dRm >= i_dLm: i_Lm += 1
            # print(i_Lm, i_Rm)
        #
        self.t_curr_margins = (i_Lm, i_Rm)
        self.l_all_margins.append(self.t_curr_margins)
    #
    def process_frame(self, na_input, b_verbose=False):
        if na_input.shape != self._t_init_shape:
            raise ValueError("Unexpected frame shape: %s expecting: %s" % \
                (repr(na_input.shape), self._t_init_shape) \
            )
        self._calc_margins(na_input.sum(self.i_target_axis))

        # assign the output
        if self.i_target_axis == 0:
            self.na_out = na_input[:, self.t_curr_margins[0]:self.t_curr_margins[1]]

        elif self.i_target_axis == 1:
            self.na_out = na_input[self.t_curr_margins[0]:self.t_curr_margins[1], :]

        else: raise ValueError("Unexpected target axis: %s" % repr(self.i_target_axis))

        if b_verbose:
            print("na_input.shape=%s t_curr_margins=%s na_out.shape=%s" % \
                (repr(na_input.shape), repr(self.t_curr_margins), repr(self.na_out.shape)) \
            )
        #
    #
#

class CHighPassSpatialFilter(object):
    """
    Perform high pass spatial filtering of an input frame.
    """
    def __init__(self, i_filter_sz):
        t_flt_sz = (i_filter_sz, i_filter_sz)
        kernel_size = tuple([(3 * i) // 2 * 2 + 1 for i in t_flt_sz])
        kernel = cv2.getGaussianKernel(kernel_size[0], t_flt_sz[0])
        self.kernel2D = kernel.dot(kernel.T)
        nz = np.nonzero(self.kernel2D >= self.kernel2D[:, 0].max())
        zz = np.nonzero(self.kernel2D  < self.kernel2D[:, 0].max())
        self.kernel2D[nz] -= self.kernel2D[nz].mean()
        self.kernel2D[zz] = 0
        # main data exchange interface for this class
        self.na_out = None
    #
    def process_frame(self, na_input):
        self.na_out = cv2.filter2D(na_input, -1, self.kernel2D, borderType=cv2.BORDER_REFLECT)
    #
#

class CMoCorrFrameWiseRigid(object):
    """
    Perform rigid motion correction of a single input frame.
    The whole frame will be used as input for shift calculation
    related to input template frame.
    """
    def __init__(self, max_shift_w=25, max_shift_h=25):
        self.ms_w, self.ms_h = max_shift_w, max_shift_h
        # main data exchange interface for this class
        self.na_out          = None # output frame
        self.l_shift         = None # [x, y] amount of shift applied to the output frame
        self.f_f2t_xcorr     = None # float, scalar - input frame <-> template xcorr value
        self.na_out_template = None # updated template
    #
    def process_frame(self, na_input, na_input_template, frame_num):
        h_i, w_i = na_input_template.shape
        na_crop_template = na_input_template[self.ms_h:h_i - self.ms_h, self.ms_w:w_i - self.ms_w].astype(np.float32)
        res = cv2.matchTemplate(na_input, na_crop_template, cv2.TM_CCORR_NORMED)

        top_left = cv2.minMaxLoc(res)[3]
        sh_y, sh_x = top_left

        # FROM PYFLUO https://github.com/bensondaled/pyfluo
        if (0 < top_left[1] < 2 * self.ms_h - 1) & (0 < top_left[0] < 2 * self.ms_w - 1):
            # if max is internal, check for subpixel shift using gaussian
            # peak registration
            log_xm1_y = np.log(res[sh_x - 1, sh_y])
            log_xp1_y = np.log(res[sh_x + 1, sh_y])
            log_x_ym1 = np.log(res[sh_x, sh_y - 1])
            log_x_yp1 = np.log(res[sh_x, sh_y + 1])
            four_log_xy = 4 * np.log(res[sh_x, sh_y])

            sh_x_n = -(sh_x - self.ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
            sh_y_n = -(sh_y - self.ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

            # Seems like this is wrong way bacause shifts are always rounded to integers.
            # Implementation of the old_div(A,B) is: np.int(np.floor(A/B))
            # sh_x_n = -(sh_x - self.ms_h + np.int(np.floor((log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))))
            # sh_y_n = -(sh_y - self.ms_w + np.int(np.floor((log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))))
        else:
            sh_x_n = -(sh_x - self.ms_h)
            sh_y_n = -(sh_y - self.ms_w)

        M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
        min_, max_ = np.min(na_input), np.max(na_input)

        # assign output
        self.f_f2t_xcorr = np.mean(res) # this is equal to np.max(res) because the 'res' consist of only single element
        self.l_shift = [sh_x_n, sh_y_n]
        self.na_out = np.clip(cv2.warpAffine(na_input, M, (w_i, h_i), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), min_, max_)
        self.na_out_template = na_input_template * frame_num / (frame_num + 1) + 1. / (frame_num + 1) * self.na_out
    #
#

class CMoCorrRigid(object):
    """
    Perform rigid motion correction of a sequence of input frames i.e. a movie.
    """
    def __init__(self, oc_movie, d_param):
        # main data exchange interface for this class
        self.l_shifts_buf = []
        self.l_xcorrs_buf = []
        self.i_frame_num = 0
        self.b_fread_status = False
        self.na_out = None

        # private stuff
        self.oc_movie = oc_movie
        self.f_bias = d_param['bias_value_to_add_to_movie']
        self.i_tup_ival = d_param['template_update_interval']

        self.dq_frames    = collections.deque(maxlen=d_param['frame_queue_sz'])
        self.dq_templates = collections.deque(maxlen=d_param['template_queue_sz'])

        self.oc_corrector = CMoCorrFrameWiseRigid( \
            max_shift_w=d_param['max_shift_w'], \
            max_shift_h=d_param['max_shift_h'] \
        )
        self.max_w, self.max_h, self.min_w, self.min_h = 0, 0, 0, 0

        self.na_template = bin_median( \
            bootstrap_template( \
                self.oc_movie, \
                d_param['template_bootstrap_sz'], \
                s_method=d_param['template_bootstrap_method'], \
                b_verbose=True \
            )
        )

        # add bias to the template frame
        self.na_template += self.f_bias

        if np.percentile(self.na_template, 1) < - 10:
            raise ValueError("Movie data is too negative. Increase 'bias_value_to_add_to_movie' value")

        # perform correction for the first frame in the movie
        self.b_fread_status = self.oc_movie.read_frame(0)
        na_frame = self.oc_movie.na_frame[:,:,0].astype(np.float32)
        na_frame += self.f_bias
        self.oc_corrector.process_frame(na_frame, self.na_template, self.i_frame_num)

        self.max_h, self.max_w = np.ceil( np.maximum((self.max_h, self.max_w), self.oc_corrector.l_shift)).astype(np.int)
        self.min_h, self.min_w = np.floor(np.minimum((self.min_h, self.min_w), self.oc_corrector.l_shift)).astype(np.int)

        # self.oc_corrector.na_out and self.oc_corrector.na_out_template are
        # 2D float32 matrices here. Or at least expected to be in such shape.
        self.na_template =       self.oc_corrector.na_out_template.copy()
        self.dq_frames.append(   self.oc_corrector.na_out.copy() )
        self.l_shifts_buf.append(self.oc_corrector.l_shift.copy())
        self.l_xcorrs_buf.append(self.oc_corrector.f_f2t_xcorr)
        self.i_frame_num += 1
    #
    def process_frame(self, b_verbose=False):
        # read next frame
        self.b_fread_status = self.oc_movie.read_next_frame()
        if not self.b_fread_status: return self.b_fread_status

        # process frame
        na_frame = self.oc_movie.na_frame[:,:,0].astype(np.float32)
        na_frame += self.f_bias
        self.oc_corrector.process_frame(na_frame, self.na_template, self.i_frame_num)

        self.max_h, self.max_w = np.ceil( np.maximum((self.max_h, self.max_w), self.oc_corrector.l_shift)).astype(np.int)
        self.min_h, self.min_w = np.floor(np.minimum((self.min_h, self.min_w), self.oc_corrector.l_shift)).astype(np.int)

        if self.i_frame_num < self.dq_frames.maxlen:
            self.na_template = self.oc_corrector.na_out_template.copy() # need to make a copy
        #
        self.dq_frames.append( self.oc_corrector.na_out.copy() )        # need to make a copy

        if self.i_frame_num % self.i_tup_ival == 0:
            if self.i_frame_num >= self.dq_frames.maxlen:
                self.dq_templates.append(np.mean(self.dq_frames, 0))
                self.na_template = np.median(self.dq_templates, 0)
            #
            if b_verbose:
                print(self.oc_movie.get_frame_stat())
            #

        # assign the output
        self.na_out = self.oc_corrector.na_out                     # by reference
        self.l_xcorrs_buf.append(self.oc_corrector.f_f2t_xcorr)    # don't need a copy for scalar variables
        self.l_shifts_buf.append(self.oc_corrector.l_shift.copy()) # make a copy because l_shift is a list
        self.i_frame_num += 1

        if b_verbose:
            # NOTE: self.oc_corrector.f_f2t_xcorr is the value BEFORE correction
            # After correction it become 0.999999 or even 1.0
            print("frame2template_xcorr = %.9f shift=[ %+.4f, %+.4f ]" % ( \
                self.oc_corrector.f_f2t_xcorr, \
                self.oc_corrector.l_shift[0], \
                self.oc_corrector.l_shift[1]) \
            )
        #
        return self.b_fread_status
    #
#
