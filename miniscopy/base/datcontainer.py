#!/usr/bin/env python3

import os, sys
import numpy as np

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


class CDatContainer(object):
    def __init__(self, s_target_dir):
        self.s_target_dir = s_target_dir

        # name the files
        s_sett_fname = os.path.join(s_target_dir, "settings_and_notes.dat")
        s_ts_fname   = os.path.join(s_target_dir, "timestamp.dat")

        # make a list of lists: list of "\t"-separated strings in a line; 4 lines in larger list
        h_settings_and_notes = open(file=s_sett_fname, mode="r")
        # i[:-1] to remove "\n" at the end of each line
        l_settings_and_notes = [i[:-1].split("\t") for i in h_settings_and_notes.readlines()]
        # [['animal', 'excitation', 'msCamExposure', 'recordLength'],
        #  ['name', '60', '255', '0'],
        #  ['behav_ROI_x', 'behav_ROI_y', 'behav_ROI_w', 'behav_ROI_h'],
        #  ['8', '242', '624', '101'],
        #  ['elapsedTime', 'Note']]

        if len(l_settings_and_notes) < 4:
            raise ValueError("Unexpected file format: %s" % repr(l_settings_and_notes))

        # keys of d_param dictionary from first line
        l_param_keys = l_settings_and_notes[0]
        # items of d_param dictionary from second line (changes numeric values to integers)
        l_param_items = [int(i) if i.isnumeric() else i for i in l_settings_and_notes[1]]

        # dictionary with l_param_keys as keys corresponding to the l_param_items as the items (with same index)
        self.d_param = {key:l_param_items[item] for item, key in enumerate(l_param_keys)}

        # keys of d_behav_ROI
        l_behav_ROI_keys = ["x", "y", "w", "h"]
        # items of d_behav_ROI from fourth line
        l_behav_ROI_items = [int(i) for i in l_settings_and_notes[3]]

        # dictionary with l_behav_ROI_keys as keys corresponding to the l_behav_ROI_items as the items (with same index)
        self.d_behav_ROI = {key: l_behav_ROI_items[item] for item, key in enumerate(l_behav_ROI_keys)}


        # timestamp.dat file

        # initialize lists for each column
        self.l_camNum = []
        self.l_frameNum = []
        self.l_sysClock = []
        self.l_buffer = []

        with open(file=s_ts_fname, mode="r") as h_files:

            for idx, s_line in enumerate(h_files):

                s_line = s_line.strip()

                if s_line == '': continue

                # check that all lines are 4 tokens separated by tabs
                l_tokens = s_line.split('\t')
                if len(l_tokens) != 4:
                    raise ValueError("Unexpected file format in file %s at line %s: %s" % (s_ts_fname, idx, s_line))

                # check that first line is:
                # "camNum\tframeNum\tsysClock\tbuffer\n"
                if idx == 0:
                    if s_line != "camNum\tframeNum\tsysClock\tbuffer":
                        raise ValueError("Unexpected file format in file %s at line 0: %s" % (s_ts_fname, s_line))
                    else:
                        continue  # We know that this is the correct file, with the correct column names

                # check that after the first line, all lines are 4 integers and add them to appropriate lists
                try:
                    self.l_camNum.append(int(l_tokens[0]))
                    self.l_frameNum.append(int(l_tokens[1]))
                    self.l_sysClock.append(int(l_tokens[2]))
                    self.l_buffer.append(int(l_tokens[3]))
                except ValueError:
                    raise ValueError("Unexpected file format in file %s at line %s: %s" % (s_ts_fname, idx, s_line))

            # check if the self.l_camNum list contain only zeros and ones (regardless of order)
            na_camNum = np.array(self.l_camNum, np.int)
            if not ((0 <= na_camNum).all() and (na_camNum <= 1).all()):
                raise ValueError("Unexpected file format: number other than 0 or 1 found in camNum")

            # check if the self.l_camNum list is 101010... OR 010101...
            # THIS WILL NOT WORK BECAUSE THERE ARE MANY CASES OF WRONG ALTERNATION
            #if (na_camNum[::2] == 0).all() and (na_camNum[1::2] == 1).all():  # 01010101...
            #    b_status = True  # so that I know later that it is in the "even" configuration
            #elif (na_camNum[::2] == 1).all() and (na_camNum[1::2] == 0).all():  # 10101010...
            #    b_status = False  # "odd" configuration
            #else:
            #    raise ValueError("Unexpected file format")  # give more information

            # split 3 lists into two categories according to the camera number
            self.l_frameNum_cam0 = []
            self.l_sysClock_cam0 = []
            self.l_buffer_cam0 = []
            self.l_frameNum_cam1 = []
            self.l_sysClock_cam1 = []
            self.l_buffer_cam1 = []

            for i in range(len(self.l_camNum)):
                if self.l_camNum[i] == 0:
                    self.l_frameNum_cam0.append(self.l_frameNum[i])
                    self.l_sysClock_cam0.append(self.l_sysClock[i])
                    self.l_buffer_cam0.append(self.l_buffer[i])
                elif self.l_camNum[i] == 1:
                    self.l_frameNum_cam1.append(self.l_frameNum[i])
                    self.l_sysClock_cam1.append(self.l_sysClock[i])
                    self.l_buffer_cam1.append(self.l_buffer[i])


if __name__ == '__main__':

    # check if argument passed by user
    if len(sys.argv) < 2:
        print("Not enough arguments")
        sys.exit(-1)
    #

    # check if target directory (argument) does exist and it is actually directory (not a file for example)
    if not os.path.isdir(sys.argv[1]):
        print("Argument is not a directory")
        sys.exit(-2)

    # instantiate (create) an object of class CDatLoader
    oc_dat_container = CDatContainer(sys.argv[1])

    print(oc_dat_container.d_param)
    print(oc_dat_container.d_behav_ROI)
    for i in range(10):
        print(oc_dat_container.l_camNum[i],
              oc_dat_container.l_frameNum[i],
              oc_dat_container.l_sysClock[i],
              oc_dat_container.l_buffer[i])