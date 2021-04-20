import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Adapted from https://gist.github.com/Andrewnetwork/a59f60daa019d99529f7622bef50a105


# EDIT THIS
# Find your dpi: https://www.infobyip.com/detectmonitordpi.php
# We are working with inches here via scaling by our dpi.
my_dpi = 96


def createCanvas(width, height, my_dpi):
    plt.close()
    plt.figure()
    fig, ax = plt.subplots(1)
    fig.dpi = my_dpi
    fig.set_size_inches(width, height)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def createRectangle(wh, xy, cwch, color="black"):
    """
    All units are in inches.

    Parameters
    ==========
    wh: (width,height)
    xy: (x,y)
    cwch: (canvasWidth,canvasHeight)
    """
    return patches.Rectangle((xy[0] / cwch[0], xy[1] / cwch[1]), wh[0] / cwch[0], wh[1] / cwch[1], color=color)


def createCheckersBoard(grid, captionLoc=True):
    n = len(grid[0])
    canvasSize = (n, n)
    fig, ax = createCanvas(canvasSize[0], canvasSize[1], my_dpi)
    color = "black"

    colordic = {'F': "paleturquoise", 'H': "thistle", 'S': 'aquamarine', 'G': 'lightpink'}

    for row in range(n):
        for col in range(n):
            num = grid[row, col]
            color = colordic[num]
            letter = num
            rect = createRectangle((1, 1), (row, col), canvasSize, color)
            ax.add_patch(rect)
            if captionLoc:
                locStr = letter
                ax.text((row + 0.23) / n, (col + 0.35) / n, locStr, fontsize=12, color="black")
    plt.savefig("stuff.png")
    plt.close()
    plt.figure()


# for i in range(2, 9, 2):
#     createCheckersBoard(i, True)

def main(grid):
    print(grid)
    grid = np.rot90(grid, 3)
    createCheckersBoard(grid)

if __name__ == '__main__':
    main()