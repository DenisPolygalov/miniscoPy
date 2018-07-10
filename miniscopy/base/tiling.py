#!/usr/bin/env python3

import warnings
import numpy as np

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

class CTiledImage(object):
    def __init__(self, na_input, i_nrows, i_ncols, b_copy_input=False):
        if na_input.shape[0] % i_nrows != 0: warnings.warn("FRACTIONAL HORIZONTAL PARTITION")
        if na_input.shape[1] % i_ncols != 0: warnings.warn("FRACTIONAL VERTICAL PARTITION")
        
        self.shape = (i_nrows, i_ncols)
        
        if b_copy_input:
            self._na_data = na_input.copy()
        else:
            self._na_data = na_input
        #
        self._i_nrows = i_nrows
        self._i_ncols = i_ncols
        self._Ridx = np.linspace(0, na_input.shape[0], (i_nrows + 1), dtype=np.int)
        self._Cidx = np.linspace(0, na_input.shape[1], (i_ncols + 1), dtype=np.int)
        
        if   len(na_input.shape) == 3: self._b_have_color_data = True
        elif len(na_input.shape) == 2: self._b_have_color_data = False
        else: raise ValueError("Unsupported shape of the input array: %s" % repr(na_input.shape))
        #
    #
    def __getitem__(self, t_addr):
        i_row, i_col = t_addr[0], t_addr[1]
        
        if self._b_have_color_data:
            return self._na_data[ \
                self._Ridx[i_row]:self._Ridx[i_row + 1], \
                self._Cidx[i_col]:self._Cidx[i_col + 1], :]
        else:
            return self._na_data[ \
                self._Ridx[i_row]:self._Ridx[i_row + 1], \
                self._Cidx[i_col]:self._Cidx[i_col + 1] ]
            #
        #
    #
    def __setitem__(self, t_addr, na_data):
        i_row, i_col = t_addr[0], t_addr[1]
        
        if self._b_have_color_data:
            self._na_data[ \
                self._Ridx[i_row]:self._Ridx[i_row + 1], \
                self._Cidx[i_col]:self._Cidx[i_col + 1], :] = na_data[:,:,:]
        else:
            self._na_data[ \
                self._Ridx[i_row]:self._Ridx[i_row + 1], \
                self._Cidx[i_col]:self._Cidx[i_col + 1] ] = na_data[:,:]
            #
        #
    #
    def set_all(self, na_data):
        if self._b_have_color_data:
            self._na_data[:,:,:] = na_data[:,:,:]
        else:
            self._na_data[:,:] = na_data[:,:]
        #
    #
    def __str__(self):
        l_out = []
        l_row = []
        for i_row in range(self._i_nrows):
            l_row.clear()
            for i_col in range(self._i_ncols):
                l_row.append(repr(self[i_row,i_col].shape))
                l_row.append(" | ")
            l_row.append('\n')
            s_data_row = "".join(l_row)
            l_out.append(s_data_row)
            s_dashes = (len(s_data_row)-2) * '-' + '\n'
            l_out.append(s_dashes)
        return "".join(l_out)
    #
#

if __name__ == '__main__':
    
    # input image size:
    i_nrows, i_ncols = 8, 8
    
    # number of tiles to split the input image:
    i_nrow_tiles, i_ncol_tiles = 2, 2
    
    # the input image
    na_img = np.arange(i_nrows * i_ncols, dtype=np.int).reshape(i_nrows, i_ncols)
    
    print(na_img)
    print()
    
    # the tiled image:
    oc_timg = CTiledImage(na_img, i_nrow_tiles, i_ncol_tiles)
    
    # you can just print it in order to see resulting tile sizes:
    print(oc_timg)
    
    # the way how to process each tile:
    for ix, iy in np.ndindex(oc_timg.shape):
        oc_timg[ix, iy] = oc_timg[ix, iy] * 2
    #
    for ix, iy in np.ndindex(oc_timg.shape):
        print( oc_timg[ix, iy] )
        print()
    #
    
    na_ones = np.ones(i_nrows * i_ncols, dtype=np.int).reshape(i_nrows, i_ncols)
    oc_timg.set_all(na_ones)
    
    for ix, iy in np.ndindex(oc_timg.shape):
        print( oc_timg[ix, iy] )
        print()
    #
#
