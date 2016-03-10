from __future__ import division

import numpy as np

def calc_row_col(num_ex, num_items, max_items_per_row=9):
    num_rows_per_ex = int(np.ceil(num_items / max_items_per_row))
    if num_items > max_items_per_row:
        num_col = max_items_per_row
        num_row = num_rows_per_ex * num_ex
    else:
        num_row = num_ex
        num_col = num_items

    def calc(ii, jj):
        col = jj % max_items_per_row
        row = num_rows_per_ex * ii + int(jj / max_items_per_row)

        return row, col

    return num_row, num_col, calc


def set_axis_off(axarr, num_row, num_col):
    for row in xrange(num_row):
        for col in xrange(num_col):
            if num_col > 1:
                ax = axarr[row, col]
            else:
                ax = axarr[row]
            ax.set_axis_off()
    pass
