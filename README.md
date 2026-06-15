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

## Installing simpleNFB on your computer
    * Simple Install
    - open command prompt
    - (optional) activate preferred envionrment -- currently works with python 3.13
        pip install "git+https://github.com/nanotech-empa/simpleNFB.git@refactor"
    
    * Involved Install
    - (optional?) Install GitHub desktop on your computer (https://desktop.github.com/download/)
    - Clone this repository from the internet (https://github.com/nanotech-empa/simpleNFB)
    - Verfiy the repository was installed on your computer (C:\Users\username\Documents\GitHub\simpleNFB)
    - (recommended) Create a virtual environment
    - Open terminal (cmd.exe / powershell / git-bash) (optional: activate your virtual environment)
        - Navigate to the directory of the repo downlaoded from github
        - pip install .
    - (required) ensure you have installed spmpy and its required dependencies
## Using the file browser
    - the sNFB_template.ipynb provides a minimal interface to access the browser tools
    - open the jupyter notebook in VS Code or internet browser
    - run the first cell to import the browser library
    - the second cell starts the sxm file browser
    - the third cell starts the dat file browser

