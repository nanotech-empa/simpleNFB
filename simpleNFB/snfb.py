'''
Created: 14.06.23
Author: amsp
Description: Module of ipython widgets for quick user interface jupyter notebook

Classes:
    imageBrowser: view SXM data and export images with information
    spectrumBrowser: view and compile DAT files

## the browsers are intended to be used with the line: %matplotlib widget
## however, they will work with %matplotlib notebook

** this library will be replaced by Lysanders data browser when it is complete
'''

import SXM_browser
import DAT_browser
import _browser
import fileBrowser

import importlib
importlib.reload(SXM_browser)
importlib.reload(DAT_browser)
importlib.reload(_browser)
importlib.reload(fileBrowser)
sxmBrowser = SXM_browser.imageBrowser
datBrowser = DAT_browser.spectrumBrowser
Browser = _browser._Browser
fileSelection = fileBrowser._fileBrowser