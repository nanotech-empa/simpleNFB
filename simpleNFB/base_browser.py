"""
base_browser.py
---------------
Abstract base class shared by imageBrowser (SXM_browser) and spectrumBrowser (DAT_browser).

Extracted in Phase-1 refactor to eliminate ~8 methods that were copy-pasted identically
across both browser files.

No ipywidgets import is required here: widget attributes are accessed through ``self``
at call-time, so this module is importable without a running Jupyter kernel (important
for unit tests).
"""

import os
import subprocess
from pathlib import Path

import plotly.express as px


class BaseBrowser:
    """
    Shared functionality for all file-browser widgets.

    Subclasses are expected to set up the following instance attributes in their
    own ``__init__`` before any of these methods are called:

    * ``self.errors``              – list[str]
    * ``self.errorText``           – widget with ``.options``
    * ``self.directories``         – list[Path]
    * ``self.active_dir``          – Path
    * ``self.directorySelection``  – widget with ``.index``, ``.options``, ``.value``
    * ``self.directoryDisplayDepth`` – widget with ``.value`` ('full' | int)
    * ``self.rootFolder``          – widget with ``.value``
    * ``self.selectionList``       – widget with ``.value``, ``.index``
    * ``self.refreshBtn``          – Button widget (used for type-identity checks)
    * ``self.copyBtn``             – Button widget with ``.icon``
    * ``self.figure``              – plotly FigureWidget (set by subclass ``__init__``)
    * ``self.last_save_fname``     – str  (set by subclass ``save_figure``)
    * ``self.v_settings_layout``   – container widget with ``.children``, ``.layout``
    """

    # Directories containing these strings are never recursed into.
    # Subclasses may override this frozenset to customise the list.
    _SKIP_DIRS: frozenset = frozenset({
        'browser_outputs',
        'ipynb',
        'raw_stml_data',
        'spmpy',
        '__pycache__',
    })

    # Suppresses reactive observers while widget values are set programmatically
    # (e.g. during file load or navigation).  Instance assignment shadows this default.
    _loading: bool = False

    # ------------------------------------------------------------------
    # Layout utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _layout_helpers():
        """Return (L, FL, FLB, FLH) Layout factory lambdas used throughout widget construction."""
        import ipywidgets as widgets
        L   = lambda w: widgets.Layout(visibility='visible', width=f'{w}px')
        FL  = lambda w: widgets.Layout(display='flex', width=f'{w}%')
        FLB = lambda w: widgets.Layout(display='flex', width=f'{w}%',
                                       align_items='center', justify_content='center')
        FLH = lambda w: widgets.Layout(visibility='hidden', display='flex', width=f'{w}%')
        return L, FL, FLB, FLH

    def _build_common_file_widgets(self) -> None:
        """Create file-browser widgets shared by all subclasses."""
        import ipywidgets as widgets
        from .widget_helpers import Text_Widget, Selection_Widget
        _, FL, FLB, _ = self._layout_helpers()

        self.rootFolder = widgets.Text(
            description='', layout=widgets.Layout(display='flex', width='95%'))
        self.directorySelection = widgets.Select(
            options=self.directories, rows=8, layout=FL(98))
        self.directoryDisplayDepth = widgets.Dropdown(
            description='depth', value=1, options=['full', 1, 2, 3, 4, 5],
            tooltip='Depth of folder structure shown in selection menu',
            layout=FLB(75), style={'description_width': '40px'})
        self.filenameText = Text_Widget('')
        self.indexText    = Text_Widget('0')
        self.errorText    = Selection_Widget([], 'Out:', rows=5)

    def _build_main_layout(self, file_col, img_col, session_label_width: int = 5) -> None:
        """Assemble the top-level 3-column layout and set it as self.h_main_layout."""
        import ipywidgets as widgets
        from .widget_helpers import HBox, VBox
        _, FL, _, _ = self._layout_helpers()

        self.h_main_layout = VBox(children=[
            HBox(children=[
                widgets.Label('Session', layout=widgets.Layout(
                    display='flex', justify_content='flex-start',
                    width=f'{session_label_width}%')),
                self.rootFolder], layout=FL(99)),
            HBox(children=[file_col, img_col, self.v_settings_layout], layout=FL(99))],
            layout=FL(100))
        self.v_settings_layout.layout.min_width = '200px'
        file_col.layout.min_width = '200px'

    # ------------------------------------------------------------------
    # Figure defaults and utilities
    # ------------------------------------------------------------------

    def _apply_font_defaults(self) -> None:
        """Set global figure font to Arial with black text."""
        self.figure.update_layout(font=dict(family='Arial', color='black'))

    @staticmethod
    def _resolve_colorscale(name: str):
        """Convert a qualitative or sequential colormap name to a Plotly colorscale value."""
        if hasattr(px.colors.qualitative, name):
            colors = getattr(px.colors.qualitative, name)
            n = len(colors)
            return [[i / max(n - 1, 1), c] for i, c in enumerate(colors)]
        return name

    def _setup_figure_autosize(self) -> None:
        """Observe JS→Python relayout syncs to keep height proportional to width."""
        self._aspect = None
        self._fig_resizing = False
        self.figure.observe(self._on_figure_relayout, names=['_js2py_relayout'])

    def _on_figure_relayout(self, change) -> None:
        if self._fig_resizing:
            return
        data = change.get('new') or {}
        relayout_data = data.get('relayout_data', {})
        w = relayout_data.get('width')
        if not w or w <= 0:
            return
        if self._aspect is None:
            h = relayout_data.get('height') or self.figure.layout.height
            if h:
                self._aspect = h / w
            return
        target_h = round(w * self._aspect)
        if abs((self.figure.layout.height or 0) - target_h) > 2:
            self._fig_resizing = True
            try:
                self.figure.update_layout(height=target_h)
            finally:
                self._fig_resizing = False

    def _build_font_widgets(self) -> None:
        """Create figure font/layout control widgets as self.fig* attributes."""
        import ipywidgets as widgets
        W98 = widgets.Layout(display='flex', width='98%')
        W48 = widgets.Layout(display='flex', width='48%')
        fWidth = lambda percent: widgets.Layout(display='flex', width=f'{percent}%')
        ds  = lambda px: {'description_width': f'{px}px'}

        self.figFontFamily = widgets.Dropdown(
            options=['Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Georgia', 'Verdana'],
            value='Arial', description='Font:', layout=W98, style=ds(40))
        self.figTitleSize = widgets.BoundedIntText(
            value=14, min=4, max=48, description='Title:', layout=fWidth(62), style=ds(38))
        self.figTitleColor = widgets.ColorPicker(
            value='black', description='', concise=True, layout=fWidth(34))
        self.figAxesLabelSize = widgets.BoundedIntText(
            value=12, min=4, max=36, description='Axes:', layout=fWidth(62), style=ds(38))
        self.figAxesLabelColor = widgets.ColorPicker(
            value='black', description='', concise=True, layout=fWidth(34))
        self.figTickSize = widgets.BoundedIntText(
            value=10, min=4, max=28, description='Ticks:', layout=fWidth(62), style=ds(38))
        self.figTickColor = widgets.ColorPicker(
            value='black', description='', concise=True, layout=fWidth(34))
        self.figLegendSize = widgets.BoundedIntText(
            value=11, min=4, max=28, description='Legend:', layout=fWidth(62), style=ds(50))
        self.figLegendColor = widgets.ColorPicker(
            value='black', description='', concise=True, layout=fWidth(34))
        self.figFontApplyBtn = widgets.Button(description='Apply Font', layout=W98)

    def _font_settings_tab(self):
        """Return a VBox suitable for an accordion tab with font/layout controls."""
        import ipywidgets as widgets
        W98 = widgets.Layout(display='flex', width='98%')
        HB  = lambda *ws: widgets.HBox(children=list(ws),
                                       layout=widgets.Layout(display='flex', width='98%'))
        return widgets.VBox(children=[
            self.figFontFamily,
            widgets.Label('── Title ──', layout=W98),
            HB(self.figTitleSize, self.figTitleColor),
            widgets.Label('── Axes labels ──', layout=W98),
            HB(self.figAxesLabelSize, self.figAxesLabelColor),
            widgets.Label('── Tick labels ──', layout=W98),
            HB(self.figTickSize, self.figTickColor),
            widgets.Label('── Legend ──', layout=W98),
            HB(self.figLegendSize, self.figLegendColor),
            self.figFontApplyBtn,
        ], layout=widgets.Layout(visibility='hidden', display='flex',
                                 width='98%', flex_flow='column'))

    def _apply_figure_layout(self, _=None) -> None:
        """Apply current font widget values to the figure."""
        family = self.figFontFamily.value
        self.figure.update_layout(
            font=dict(family=family),
            title=dict(font=dict(size=self.figTitleSize.value,
                                 color=self.figTitleColor.value)),
            legend=dict(font=dict(size=self.figLegendSize.value,
                                  color=self.figLegendColor.value)),
        )
        self.figure.update_xaxes(
            title_font=dict(size=self.figAxesLabelSize.value,
                            color=self.figAxesLabelColor.value),
            tickfont=dict(size=self.figTickSize.value, color=self.figTickColor.value),
        )
        self.figure.update_yaxes(
            title_font=dict(size=self.figAxesLabelSize.value,
                            color=self.figAxesLabelColor.value),
            tickfont=dict(size=self.figTickSize.value, color=self.figTickColor.value),
        )

    def _connect_font_observers(self) -> None:
        """Wire the Apply Font button."""
        self.figFontApplyBtn.on_click(self._apply_figure_layout)

    def _refresh_info(self, *_) -> None:
        """Tier-2 observer: rebuild title/info text then redraw. No data reprocessing."""
        self.update_scan_info()
        self._redraw()

    def _figure_stem(self, dir_name: str) -> str:
        """Return the filename stem (without note or extension) for save_figure."""
        raise NotImplementedError

    def save_figure(self, a) -> None:
        """Save the current figure to browser_outputs/ as a high-res PNG (requires kaleido)."""
        self.saveBtn.icon = 'hourglass-start'
        out_dir = self.active_dir / 'browser_outputs'
        out_dir.mkdir(exist_ok=True)
        dir_name = str(self.directorySelection.value).split(chr(92))[-1]
        stem = self._figure_stem(dir_name)
        if self.saveNote.value:
            stem += f'_{self.saveNote.value}'
        self.last_save_fname = str(out_dir / f'{stem}.png')
        try:
            self._last_img_bytes = self.figure.to_image(format='png', scale=5)
            Path(self.last_save_fname).write_bytes(self._last_img_bytes)
            self.updateErrorText('Figure Saved')
        except Exception as err:
            self.updateErrorText(f'Save error: {err}')
        self.saveNote.value = ''
        self.saveBtn.icon = 'file-image-o'

    # ------------------------------------------------------------------
    # Error / status output
    # ------------------------------------------------------------------

    def updateErrorText(self, text: str) -> None:
        """Append *text* to the error log and refresh the output widget."""
        self.errors.append(f'{len(self.errors)} {text}')
        self.errorText.options = self.errors

    # ------------------------------------------------------------------
    # Directory discovery
    # ------------------------------------------------------------------

    def find_directories(self, _path: Path) -> list:
        """
        Recursively collect sub-directories of *_path*, skipping any whose
        name contains a string from ``_SKIP_DIRS`` and any bare data files
        that appear at the top level.

        Side-effect: extends ``self.directories`` with the found paths.

        Returns
        -------
        list[Path]
            The directories found at this level of recursion (not the full
            accumulated list).
        """
        directories = []
        try:
            entries = os.listdir(_path)
        except PermissionError:
            return directories

        for entry in entries:
            # Skip bare .dat / .sxm files that appear in os.listdir
            if entry[-4:] in ('.dat', '.sxm'):
                continue
            entry_path = Path(_path) / entry
            if not os.path.isdir(entry_path):
                continue
            if any(skip in entry for skip in self._SKIP_DIRS):
                continue
            directories.append(entry_path)
            self.find_directories(entry_path)  # recurse

        self.directories.extend(directories)
        return directories

    def update_directories(self, _event=None) -> None:
        """
        Rebuild ``directorySelection.options`` with paths trimmed to the
        depth chosen in ``directoryDisplayDepth``.
        """
        depth_val = self.directoryDisplayDepth.value
        depth = 0 if depth_val == 'full' else -int(depth_val)

        display_dirs = [
            '\\'.join(str(d).split('\\')[depth:]) for d in self.directories
        ]
        if display_dirs:
            display_dirs[0] = 'session folder'
        self.directorySelection.options = display_dirs

    # ------------------------------------------------------------------
    # File output helpers
    # ------------------------------------------------------------------

    def copy_figure(self, _event=None) -> None:
        """
        Save the current figure to disk, then copy it to the Windows clipboard.

        ``save_figure`` caches its rendered PNG bytes in ``self._last_img_bytes``,
        so this method reuses them for the clipboard — only one kaleido render total.

        Two clipboard strategies are attempted in order:

        1. **win32clipboard** (primary) — converts the cached PNG bytes to a DIB
           bitmap and writes directly to the clipboard.  Requires ``pywin32``
           and ``Pillow``.

        2. **PowerShell fallback** — calls ``Set-Clipboard -Path`` on the file
           written by ``save_figure``.  10-second timeout; non-zero exit code
           reported via ``updateErrorText``.
        """
        import io

        self.copyBtn.icon = 'hourglass-half'
        try:
            # Disk save — also populates self._last_img_bytes and last_save_fname.
            self.save_figure(_event)

            # Reuse the bytes already rendered by save_figure (no second kaleido call).
            buf = io.BytesIO(self._last_img_bytes)

            # ------------------------------------------------------------------
            # Primary path: win32clipboard — instant, no subprocess
            # ------------------------------------------------------------------
            try:
                import win32clipboard
                from PIL import Image

                img = Image.open(buf).convert('RGB')
                # Downsample to a screen-friendly size before DIB conversion
                img.thumbnail((2400, 2400), Image.LANCZOS)
                dib_buf = io.BytesIO()
                img.save(dib_buf, 'BMP')
                # Strip the 14-byte BMP file header to obtain a raw DIB block
                dib_data = dib_buf.getvalue()[14:]

                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib_data)
                finally:
                    win32clipboard.CloseClipboard()

            except ImportError:
                # ----------------------------------------------------------
                # Fallback: PowerShell Set-Clipboard using the saved file path
                # ----------------------------------------------------------
                ps_cmd = f'Set-Clipboard -Path "{self.last_save_fname}"'
                result = subprocess.run(
                    ['powershell', '-Command', ps_cmd],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    self.updateErrorText(
                        f'clipboard error: {result.stderr.decode(errors="replace").strip()}'
                    )

        except Exception as err:
            self.updateErrorText(f'copy_figure error: {err}')
        finally:
            self.copyBtn.icon = 'clipboard'

    # ------------------------------------------------------------------
    # Root-folder / refresh handler
    # ------------------------------------------------------------------

    def handler_root_folder_update(self, event=None) -> None:
        """
        Called when ``rootFolder`` text changes or ``refreshBtn`` is clicked.

        * Validates the typed path.
        * If valid, updates ``active_dir`` and rebuilds the directory tree.
        * When triggered by ``refreshBtn``, restores the previously selected
          directory after the rebuild (was a latent NameError in the original
          code if the path was invalid — fixed here).
        """
        new_root = self.rootFolder.value

        # Only capture current selection when the refresh button fires
        current_directory = None
        if type(event) == type(self.refreshBtn):
            current_directory = self.directorySelection.value

        if os.path.exists(new_root) and os.path.isdir(new_root):
            self.directorySelection.options = [self.active_dir]
            self.directories = [self.active_dir]
            self.active_dir = Path(new_root)
            self.find_directories(self.active_dir)
            self.update_directories(event)

        # Restore selection only when a valid refresh was requested
        if current_directory is not None:
            try:
                self.directorySelection.value = current_directory
            except Exception:
                pass  # selection may no longer be in the list after a root change

    # ------------------------------------------------------------------
    # Settings panel visibility
    # ------------------------------------------------------------------

    def _set_settings_visibility(self, visible: bool) -> None:
        """
        Show or hide ``v_settings_layout`` and all of its children.

        Uses duck-typing (``hasattr(child, 'children')``) so this method
        works without importing ipywidgets directly.

        Parameters
        ----------
        visible : bool
            ``True`` → visibility = 'visible',  ``False`` → 'hidden'
        """
        vis = 'visible' if visible else 'hidden'
        self.v_settings_layout.layout.visibility = vis
        for child in self.v_settings_layout.children:
            # Recursively show/hide nested containers (e.g. HBox inside Accordion)
            if hasattr(child, 'children'):
                for grandchild in child.children:
                    grandchild.layout.visibility = vis
            child.layout.visibility = vis
