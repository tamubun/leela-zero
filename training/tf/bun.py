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
    shape=(x,y)の 配列を表示。
    fnameを指定した時は、画面表示せず、ファイルに落とす。
    scaleで拡大率を変える。
    figid を指定すると、同時に複数のウィンドウに表示できる。
    m は、normalize用の最大値。
    mを渡さなければ、全体の絶対値の最大値を使う。
    """
    if m is None:
        m = np.abs(a).max()
    (x,y) = a.shape
#    a = a.T # 転置しないと盤面と合わない
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

def pos(p):
    if p is None:
        return 'pass'
    (x,y) = p
    return 'abcdefghjklmnopqrst'[x] + str(19-y)

def xy(s):
    s = s.lower()
    try:
        if len(s) < 2 or len(s) > 3:
            raise ValueError
        x = 'abcdefghjklmnopqrst'.index(s[0])
        y = 19 - int(s[1:])
        if y < 0 or y > 18:
            raise ValueError
    except ValueError:
        return None
    return (x,y)

def get_input(m, fname = '20160312-Lee-Sedol-vs-AlphaGo.sgf'):
    """
    fnameで指定したSGFの m番目の盤面からLeelaの入力を作る。
    m=0 が初期盤面。
    retval[0] が leelaの入力planes, retval[1]は、現在の盤面。
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
