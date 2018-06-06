#!/usr/bin/env python3

import os, sys, time
import matplotlib.pyplot as plt

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

class CSideBySidePlayer(object):
    def __init__(self, fig_size=(15,5), desired_fps=60):
        self.fig = plt.figure(figsize=fig_size)
        # https://matplotlib.org/api/_as_gen/matplotlib.figure.SubplotParams.html#matplotlib.figure.SubplotParams
        self.fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, bottom=0.01, top=0.99)
        
        self.f_timeout = 1.0/float(desired_fps)
        self.f_prev_frame_time = 0
        self.f_fps = 0
        
        self.axL = self.fig.add_subplot(1,2,1)
        plt.setp(self.axL, xticks=[], yticks=[])
        
        self.axR = self.fig.add_subplot(1,2,2)
        plt.setp(self.axR, xticks=[], yticks=[])
        
        self.imgL = None
        self.imgR = None
        
        self.fig.canvas.mpl_connect('key_press_event', self._cb_on_key_press_event)
        self.fig.canvas.mpl_connect('resize_event', self._cb_on_resize_event)
    #
    def _cb_on_key_press_event(self, event):
        try:
            key_code = hex(ord(event.key))
        except TypeError:
            key_code = event.key
        print("key_press_event: key_code=%s key=%s" % (key_code, repr(event.key)) )
        sys.stdout.flush()
    #
    def _cb_on_resize_event(self, event):
        fs_w, fs_h = float(event.width)/float(self.fig.dpi), float(event.height)/float(self.fig.dpi)
        print( "resize_event: %d %d figsize=(%.2f, %.2f)" % \
            (event.width, event.height, fs_w, fs_h) \
        )
        sys.stdout.flush()
    #
    def set_Lframe_data(self, na_frame):
        if self.imgL == None:
            self.imgL = self.axL.imshow(na_frame)
        else:
            self.imgL.set_data(na_frame)
        #
    #
    def set_Rframe_data(self, na_frame):
        if self.imgR == None:
            self.imgR = self.axR.imshow(na_frame)
        else:
            self.imgR.set_data(na_frame)
        #
    #
    def set_data(self, na_Lframe, na_Rframe):
        self.set_Lframe_data(na_Lframe)
        self.set_Rframe_data(na_Rframe)
    #
    def drawait(self):
        self.fig.canvas.draw_idle()
        f_curr_frame_time = time.time()
        self.f_fps = 1.0/(f_curr_frame_time - self.f_prev_frame_time)
        self.f_prev_frame_time = f_curr_frame_time
        return plt.waitforbuttonpress(timeout=self.f_timeout)
    #
#

if __name__ == '__main__':
    
    # os.sys.path.append("D:\\ ... \\miniscoPy\\")
    
    import numpy as np
    from miniscopy.base.ioutils import enum_video_files
    from miniscopy.base.mupamovie import CMuPaMovieCV
    from miniscopy.base.player import CSideBySidePlayer
    
    oc_player = CSideBySidePlayer()
    
    if len(sys.argv) < 2:
        print("ERROR: not enough input arguments. Usage:")
        print("%s folder_name_containing_video_files" % sys.argv[0])
        sys.exit(0)
    #
    
    # make a tuple of strings - input file names
    t_files = enum_video_files(sys.argv[1], "msCam*.avi")
    
    # create a multi-part movie object
    oc_movie = CMuPaMovieCV(t_files)
    
    brake = 0
    while(oc_movie.read_next_frame()):
        
        print("%s\tFPS: %.2f" % (oc_movie.get_frame_stat(), oc_player.f_fps))
        
        oc_player.set_Lframe_data(oc_movie.na_frame)
        oc_player.set_Rframe_data(np.flipud(oc_movie.na_frame[:,:,0].astype(np.float32)))
        if oc_player.drawait(): break
        
        brake += 1
        if brake >= 60: break
    #
#
