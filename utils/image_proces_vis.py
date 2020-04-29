import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from skimage import exposure
# from skimage.exposure import match_histograms
import cv2

def colormap_transparent(R=1, G=0, B=0):
    colors = [(R, G, B, a) for a in np.linspace(0, 1, 100)]
    cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
    return cmap

def set_matplotlib_params():
    return plt.rc('font', **{'family': 'serif', 'serif': ['cmss10'], 'size': '12'})

def get_greens():
    return ['g', 'lime', 'limegreen', 'forestgreen', 'lightgreen']





def histogram_matching(image = '../../input/processed/trainA/pat_30_bSSFP_7.png', reference_img ='../../input/processed/trainB/pat_12_lge_8.png'):
    image = cv2.imread(image)
    reference_img = cv2.imread(reference_img)
    matched = match_histograms(image, reference_img, multichannel=True)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                        ncols=3,
                                        figsize=(8, 3),
                                        sharex=True,
                                        sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(image)
    ax1.set_title('Source (bSSFP)')
    ax2.imshow(reference_img)
    ax2.set_title('Reference (LGE)')
    ax3.imshow(matched)
    ax3.set_title('Matched (New bSSFP)')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))


    for i, img in enumerate((image, reference_img, matched)):
        for c, c_color in enumerate(('red')):
            img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
            axes[c, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img[..., c])
            axes[c, i].plot(bins, img_cdf)
            axes[c, 0].set_ylabel(c_color)

    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')

    plt.tight_layout()