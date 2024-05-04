# simpleNFB
Description: Module of ipython widgets for quick Nanonis File Browsing in jupyter notebook

## Classes:
    - imageBrowser: view SXM data and export images with information
    - spectrumBrowser: view and compile DAT files

### The browsers are intended to be used with the line: %matplotlib widget
### Relies on spmpy to read nanonis file data, periodic updates of this module should be performed.
## Optional: 

jupyter_launch_script.bat can be used to directly open jupyter notebooks from the windows file browser. 
 1. right-click on an .ipynb file
 2. "open with"
 3. "choose another app"
 4. "jupyter_launch_script.bat"
