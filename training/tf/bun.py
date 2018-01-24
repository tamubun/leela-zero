# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys

weight = 'c83e1b6e0ffbf8e684f2d8f6261853f14c553b29ee0e70ff6c34e87d28009c43'

sys.argv.append(weight)
from AlphaGo import util

from net_to_model import *

t = tfprocess
s = t.session

mycmap = colors.LinearSegmentedColormap.from_list(
    'bun', [(0.0, 'red'),(0.5,'black'),(1.0,'white')])

def show_array(a, fname=None, scale=1, figid=0, m = None):
    """
    Show array of shape=(x,y).
    When fname is given, output figure is to file, without displaying.
    When scale is given, enlargement scale is changed.
    When figid is given, the output is shown that ids window.
    m is normalize factor and if omitted, m is automatically set.
    """
    if m is None:
        m = np.abs(a).max()
    (x,y) = a.shape

    if a.min() >= 0:
        cmap = 'gray'
        norm = colors.Normalize(0, m)
    else:
        cmap = mycmap
        norm = colors.Normalize(-m, m)

    plt.figure(figid, figsize=(15.0*scale/x, 15.0*scale/y))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(a, interpolation='nearest', cmap=cmap, norm = norm)
    if fname is None:
        plt.show(block=False)
    else:
        plt.savefig(fname)

def get_input(m, fname = '20160312-Lee-Sedol-vs-AlphaGo.sgf'):
    """
    Create LeelaZero inputs planes from SGF file.
    m: move count. m = 0 means initial position.
    fname: SGF file name.

    returns tuple
      tuple[0]: LZ input planes
      tuple[1]: current position (for checking)
    """
    p = np.zeros((1,18,361), np.float32)

    with open(fname) as f:
        it = util.sgf_iter_states(f.read(), include_end=False)

    l = []
    try:
        for _ in range(max(m-7, 0)):
            it.next()
        for i in range(min(1+m, 8)):
            state, move, player = it.next()
            l.insert(0, (state.board.transpose().flatten(), player))
    except StopIteration:
        pass

    board = l[0][0].reshape(19,19)
    player = l[0][1]
    if player == 1:
        p[0,16] = np.ones(361)
    else:
        p[0,17] = np.ones(361)

    for i, (b, _) in enumerate(l):
        blk = b.copy()
        blk[blk<=0] = 0
        wht = -b.copy()
        wht[wht<=0] = 0
        if player == 1:
            p[0, i] = blk
            p[0, 8+i] = wht
        else:
            p[0, i] = wht
            p[0, 8+i] = blk
        
    return (p, board)
