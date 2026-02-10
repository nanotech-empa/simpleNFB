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
 4. "jupyter_launch_script.bat" (user or pc version depending on Anaconda installation)

## Installing simpleNFB on your computer and running it using a global Python environment
    - Install GitHub desktop on your computer (https://desktop.github.com/download/)
    - Clone this repository from the internet (https://github.com/nanotech-empa/simpleNFB)
    - Verfiy the repository was installed on your computer (C:\Users\username\Documents\GitHub\simpleNFB)
    - Open Visual Studio Code, run the first cell and select "Global Env" as your environement.
    - In a separate code cell, run the following commands to install the required packages / libraries
        !py -m pip install spiepy
        !py -m pip install ipywidgets
        !py -m pip install scikit-image 
        !py -m pip install PyYAML
        !py -m pip install untangle
        !py -m pip install xmltodict
        !py -m pip install ipympl
    - Run the first cell again to see if it works. If not, re-open the file in Visual Studio Code again (this should help).
    - Enjoy the tool :)

