# simpleNFB

**simpleNFB** (simple Nanonis File Browser) is a set of interactive Jupyter
widgets for quickly browsing, visualising, and exporting Nanonis SPM data —
SXM scan images and DAT spectroscopy files — without leaving the notebook.

It is built for scientists and researchers already familiar with SPM data. Point
a browser at a measurement session's root folder and it recursively discovers
every SXM and DAT file, renders it with an interactive plotly figure, and gives
you publication-oriented export tools. File reading and channel/parameter access
are handled by [spmpy](https://github.com/nanotech-empa/spmpy), so the data you
see in the browser is exactly what spmpy exposes.

Two browsers are provided:

- **imageBrowser** (`sxmBrowser`) — view SXM scan channels as image plots.
- **spectrumBrowser** (`datBrowser`) — plot and compile DAT spectra in 1D or 2D.

---

## Installation

The browsers are intended for use inside a
Jupyter notebook. `spmpy` and its dependencies are installed automatically.

### Standard install

```pip
pip install git+https://github.com/nanotech-empa/simpleNFB
```

```github download + pip install
download from zip file from https://github.com/nanotech-empa/simpleNFB
extract files
open cmd prompt
    use command:
    cd <simpleNFB directory>
    (optional)
    conda activate <python environment>
    finally:
    pip install .
```

```git + bash
git clone https://github.com/nanotech-empa/simpleNFB.git
cd simpleNFB
pip install .
```

### Editable / developer install

For power users who want to modify their local copy of the library, clone the
repository and install it in editable mode. Changes to the source are picked up
without reinstalling:

```bash
git clone https://github.com/nanotech-empa/simpleNFB.git
cd simpleNFB
pip install -e .
```

> On Windows, add the optional `clipboard` extra for native PNG copy support:
> `pip install -e ".[clipboard]"`

---

## Recommended environment

**VS Code** is the recommended environment — its notebook interface renders the
widgets cleanly and the code-export tools inject cells reliably. Launching the
notebook in a standard **web browser** (JupyterLab / Notebook 7) also works and
should be fully supported.

Minimal usage:

```python
%gui asyncio # enables asynchronous funtions for file loading
import simpleNFB as snfb

sxm = snfb.sxmBrowser(home_directory='./')
sxm                        # display the SXM image browser

dat = snfb.datBrowser(home_directory='./', sxmBrowser=sxm)
dat                        # display the DAT spectrum browser
```

Passing `sxmBrowser=sxm` into the DAT browser links the two (see
**Connectivity** below). The `sNFB_template.ipynb` notebook provides a standard
ready-to-run interface.

---

## Features

### Browsing SXM files — image plot viewer
The imageBrowser renders any SXM scan channel as an interactive image plot with
selectable colormap, colour limits, and per-line / plane leveling corrections.

### Browsing DAT files — 1D and 2D plotting
The spectrumBrowser plots one or many DAT spectra together in a 1D overlay, or
stacks them into a 2D map keyed by index or any measurement parameter. Legends
can be auto-generated from a chosen parameter or replaced with custom labels.

### Recursive file discovery
Point a browser at a root directory and it walks the tree automatically,
listing every SXM and DAT file it finds (skipping output and cache folders) so
an entire measurement session is browsable from one entry point.

### Plot export
A **copy figure** button places the current plot on the clipboard as a PNG. A
save toggle (on by default) simultaneously writes the figure to a
`browser_outputs/` folder, with an optional note appended to the filename.

### File data processing functions
The numerical routines behind the browsers are available as standalone,
widget-free imports from `simpleNFB.process_utils` — including Savitzky-Golay and boxcar
smoothing, modified Z-score despiking, group averaging, line-by-line leveling,
nm→eV spectral rebinning, and tip-position resolution:

```python
from simpleNFB.process_utils import smooth_data, despike_z_score, remove_line_average
```

### Code export
Each browser can export a self-contained Python snippet that reproduces the
current plot from raw files. The snippet is copied to the clipboard and injected
into a text-edit widget — a convenient path to persistent, reproducible dataset
management and figure generation outside the notebook.

### Connectivity between browsers
The SXM browser's **context** feature lists the DAT files recorded 
after the the currently selected scan and the next (matched by acquisition time) 
and allows locations to be displayed on the image along with a marker legend.
When the DAT browser is connect to the SXM browser, a **location** button places markers on the SXM image at
the tip positions of the DAT files selected in the spectrum browser.

### spmpy wrapper functions
The browsers wrap `spmpy` and expose its capabilities directly: the loaded
`Spm` object, processed channel data, and channel/parameter lookups are
reachable through the browser's `get_channel(...)` and `get_parameter(...)`
methods, so anything spmpy can read is available programmatically.

---

## License

See [LICENSE](LICENSE).
