# snfb.py — re-export shim for notebook compatibility
# Notebooks use: import simpleNFB.snfb as snfb
#                importlib.reload(snfb)
#                snfb.sxmBrowser(...)  /  snfb.datBrowser(...)
from .SXM_browser import fileBrowser as sxmBrowser
from .DAT_browser import fileBrowser as datBrowser
