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

    # Reference counter for nested busy calls; status clears only when it reaches 0
    _busy_count: int = 0

    # Subclasses set this to 'sxm' or 'dat' to pick their template subdirectory
    _TEMPLATES_SUBDIR: str = ''

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
        """Assemble the top-level 3-column layout and set it as self.h_main_layout.

        Layout strategy (ipywidgets flex):
          file_col        — flex: '0 0 auto', fixed 250 px; wide enough for typical paths
          img_col         — flex: '1 1 auto'; expands to fill all remaining space
          v_settings_layout — flex: '0 0 auto', fixed 310 px; wide enough for all tab widgets
        """
        import ipywidgets as widgets
        from .widget_helpers import HBox, VBox

        self._status_widget = widgets.HTML(
            value=self._status_html(False),
            layout=widgets.Layout(flex='0 0 auto', margin='0 0 0 8px'),
        )

        # File column: fixed width, does not grow or shrink
        file_col.layout.flex      = '0 0 auto'
        file_col.layout.width     = '250px'
        file_col.layout.min_width = '200px'

        # Image column: fills all space left after the two fixed-width side panels
        img_col.layout.flex      = '1 1 auto'
        img_col.layout.width     = 'auto'
        img_col.layout.min_width = '300px'

        # Settings panel: fixed width, does not grow or shrink
        self.v_settings_layout.layout.flex      = '0 0 auto'
        self.v_settings_layout.layout.width     = '310px'
        self.v_settings_layout.layout.min_width = '280px'

        _row = widgets.Layout(display='flex', flex_flow='row',
                              width='100%', align_items='stretch')
        self.h_main_layout = VBox(children=[
            HBox(children=[
                widgets.Label('Session', layout=widgets.Layout(
                    flex='0 0 auto', width=f'{session_label_width}%')),
                self.rootFolder,
                self._status_widget],
                layout=widgets.Layout(display='flex', flex_flow='row',
                                      width='100%', align_items='center')),
            HBox(children=[file_col, img_col, self.v_settings_layout], layout=_row)],
            layout=widgets.Layout(display='flex', flex_flow='column', width='100%',justify_content='space-between',align_items='stretch'),)

    # ------------------------------------------------------------------
    # Figure defaults and utilities
    # ------------------------------------------------------------------

    def _figure_layout_update(
        self,
        *,
        margin: dict,
        autosize: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Apply all figure appearance settings in one atomic call.

        Invoked at startup (once widgets exist) and by the 'Apply Settings' button.
        Browser subclasses supply their fixed structural values; the remaining
        parameters are read from the Figure Settings widgets.

        Parameters
        ----------
        margin   : l/r/t/b pixel margins — differs per browser.
        autosize : True  → fill HTML container (DAT, default).
                   False → explicit pixel dimensions (SXM).
        width    : pixel width; applied only when autosize=False.
        height   : pixel height; defaults to figHeight.value.
        """
        h         = height if height is not None else self.figHeight.value
        show_line = self.figAxesBorderToggle.value
        size_kw   = {'width': width} if not autosize and width is not None else {}

        self.figure.update_layout(
            autosize=autosize, height=h, **size_kw,
            margin=margin,
            paper_bgcolor=self.figBgColor.value,
            plot_bgcolor=self.figBgColor.value,
            font=dict(family=self.figFontFamily.value, color='black'),
            title=dict(font=dict(size=self.figTitleSize.value, color=self.figTitleColor.value)),
            legend=dict(font=dict(size=self.figLegendSize.value, color=self.figLegendColor.value)),
        )
        axis_kw = dict(
            title_font=dict(size=self.figAxesLabelSize.value,
                            color=self.figAxesLabelColor.value),
            tickfont=dict(size=self.figTickSize.value, color=self.figTickColor.value),
            ticks=self.figTicksMode.value, showgrid=self.figGridToggle.value,
            showline=show_line, linewidth=1, linecolor='black', mirror=show_line,
        )
        self.figure.update_xaxes(**axis_kw, nticks=self._parse_tick_count(self.figXTickCount.value))
        self.figure.update_yaxes(**axis_kw, nticks=self._parse_tick_count(self.figYTickCount.value))

    @staticmethod
    def _resolve_colorscale(name: str):
        """Convert a qualitative or sequential colormap name to a Plotly colorscale value."""
        if hasattr(px.colors.qualitative, name):
            colors = getattr(px.colors.qualitative, name)
            n = len(colors)
            return [[i / max(n - 1, 1), c] for i, c in enumerate(colors)]
        return name


    def _setup_figure_autosize(self, fig_width: int = None, fig_height: int = None) -> None:
        """Register the JS->Python relayout observer used by subclass hooks (e.g. STML axis sync)."""
        self.figure.observe(self._on_figure_relayout, names=['_js2py_relayout'])

    def _on_figure_relayout(self, change) -> None:
        """Hook for subclasses (e.g. STML axis sync). Base class is a no-op."""
        pass

    def _build_figure_settings_widgets(self, fig_width: int = 600, fig_height: int = 500) -> None:
        """Create figure settings widgets (size, bg, axes, fonts, templates)."""
        import ipywidgets as widgets
        W98    = widgets.Layout(display='flex', width='98%')
        fWidth = lambda p: widgets.Layout(display='flex', width=f'{p}%')
        ds     = lambda px: {'description_width': f'{px}px'}

        # Size
        self.figWidth  = widgets.BoundedIntText(
            value=fig_width,  min=100, max=4000, description='W (px):',
            layout=fWidth(48), style=ds(46))
        self.figHeight = widgets.BoundedIntText(
            value=fig_height, min=100, max=4000, description='H (px):',
            layout=fWidth(48), style=ds(46))
        # Background
        self.figBgColor = widgets.ColorPicker(
            value='white', description='BG:', concise=False, layout=W98, style=ds(28))

        # Axes display
        self.figAxesBorderToggle = widgets.ToggleButton(
            value=True, description='Axes Border', layout=fWidth(48))
        self.figGridToggle = widgets.ToggleButton(
            value=False, description='Grid', layout=fWidth(48))
        self.figTicksMode = widgets.ToggleButtons(
            options=['inside', 'outside'], value='outside', description='Ticks:',
            layout=W98, style={'description_width': '42px', 'button_width': 'auto'})
        self.figXTickCount = widgets.Text(
            value='auto', description='X ticks:', layout=fWidth(48), style=ds(50))
        self.figYTickCount = widgets.Text(
            value='auto', description='Y ticks:', layout=fWidth(48), style=ds(50))

        # Font
        self.figFontFamily = widgets.Dropdown(
            options=['Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Georgia', 'Verdana'],
            value='Arial', description='Font:', layout=W98, style=ds(40))
        self.figTitleSize = widgets.BoundedIntText(
            value=8, min=4, max=48, description='Title:', layout=fWidth(62), style=ds(38))
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
        self.figFontApplyBtn = widgets.Button(description='Apply Settings', layout=W98)

        # Templates
        self._templates: dict = {}
        self._templates_dir = (
            Path(__file__).parent / 'templates' / self._TEMPLATES_SUBDIR)
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        self.figTemplateName = widgets.Text(
            value='', placeholder='template name', description='Name:',
            layout=W98, style=ds(40))
        self.figSaveTemplateBtn  = widgets.Button(description='Save as Template', layout=W98)
        self.figTemplateList     = widgets.Select(options=[], rows=4, layout=W98)
        self.figApplyTemplateBtn = widgets.Button(description='Apply Template', layout=W98)
        self._load_templates_dir()  # needs figTemplateList to exist first

    # keep old name as alias so any external callers don't break
    def _build_font_widgets(self, fig_width: int = 600, fig_height: int = 500) -> None:
        self._build_figure_settings_widgets(fig_width, fig_height)

    def _figure_settings_tab(self):
        """Return a VBox for the Figure Settings accordion tab."""
        import ipywidgets as widgets
        W98 = widgets.Layout(display='flex', width='98%')
        HB  = lambda *ws: widgets.HBox(children=list(ws), layout=W98)
        return widgets.VBox(children=[
            widgets.Label('-- Size --', layout=W98),
            self.figHeight,
            widgets.Label('-- Background --', layout=W98),
            self.figBgColor,
            widgets.Label('-- Axes --', layout=W98),
            HB(self.figAxesBorderToggle, self.figGridToggle),
            self.figTicksMode,
            HB(self.figXTickCount, self.figYTickCount),
            widgets.Label('-- Font --', layout=W98),
            self.figFontFamily,
            widgets.Label('-- Title --', layout=W98),
            HB(self.figTitleSize, self.figTitleColor),
            widgets.Label('-- Axes labels --', layout=W98),
            HB(self.figAxesLabelSize, self.figAxesLabelColor),
            widgets.Label('-- Tick labels --', layout=W98),
            HB(self.figTickSize, self.figTickColor),
            widgets.Label('-- Legend --', layout=W98),
            HB(self.figLegendSize, self.figLegendColor),
            self.figFontApplyBtn,
            widgets.Label('-- Templates --', layout=W98),
            self.figTemplateName,
            self.figSaveTemplateBtn,
            self.figTemplateList,
            self.figApplyTemplateBtn,
        ], layout=widgets.Layout(visibility='hidden', display='flex',
                                 width='98%', flex_flow='column'))

    # keep old name as alias
    def _font_settings_tab(self):
        return self._figure_settings_tab()

    @staticmethod
    def _parse_tick_count(val: str) -> int:
        """Return nticks int (0 = plotly auto) from 'auto' or numeric string."""
        try:
            return max(0, int(val))
        except (ValueError, TypeError):
            return 0

    def _apply_figure_layout(self, _=None) -> None:
        """'Apply Settings' callback. Subclasses override to supply browser-specific args."""
        self._figure_layout_update(margin=dict(l=60, r=30, t=60, b=50), autosize=True)

    def _template_extra_save(self) -> dict:
        """Hook: return extra key/value pairs to store alongside the plotly template.
        Subclasses (e.g. SXM_browser) override to capture widget state beyond fonts/axes."""
        return {}

    def _template_extra_apply(self, entry: dict) -> None:
        """Hook: apply any extra data stored in *entry* back to the browser widgets.
        Subclasses override to restore widget state captured by _template_extra_save."""
        pass

    def _template_safe_name(self, name: str) -> str:
        """Return a filesystem-safe stem derived from the template name."""
        safe = ''.join(c if c.isalnum() or c in '-_ ' else '_' for c in name)
        return safe.replace(' ', '_')

    def _save_template_file(self, name: str) -> None:
        """Write a single template to its own JSON file in the templates directory."""
        import json
        entry = self._templates[name]
        fpath = self._templates_dir / f'{self._template_safe_name(name)}.json'
        data = {
            'name': name,
            'template': entry['template'].to_plotly_json(),
            **{k: v for k, v in entry.items() if k != 'template'},
        }
        try:
            fpath.write_text(json.dumps(data, indent=2))
        except Exception as err:
            self.updateErrorText(f'template save error: {err}')

    def _load_templates_dir(self) -> None:
        """Scan the templates directory and load every JSON file on startup."""
        import json
        import plotly.graph_objects as go
        for fpath in sorted(self._templates_dir.glob('*.json')):
            try:
                data = json.loads(fpath.read_text())
                name = data.get('name', fpath.stem)
                tmpl_dict = data.get('template', {})
                tmpl = go.layout.Template(
                    layout=go.Layout(**tmpl_dict.get('layout', {})))
                self._templates[name] = {
                    'template': tmpl,
                    **{k: v for k, v in data.items() if k not in ('template', 'name')},
                }
            except Exception as err:
                self.updateErrorText(f'template load error ({fpath.name}): {err}')
        if self._templates:
            self.figTemplateList.options = list(self._templates.keys())

    def _save_as_template(self, _=None) -> None:
        """Capture current Figure Settings as a named template entry."""
        import plotly.graph_objects as go
        name = self.figTemplateName.value.strip()
        if not name:
            self.updateErrorText('Enter a template name before saving')
            return
        ticks     = self.figTicksMode.value
        show_grid = self.figGridToggle.value
        show_line = self.figAxesBorderToggle.value
        nticks_x  = self._parse_tick_count(self.figXTickCount.value)
        nticks_y  = self._parse_tick_count(self.figYTickCount.value)
        axis_layout = dict(
            title_font=dict(size=self.figAxesLabelSize.value,
                            color=self.figAxesLabelColor.value),
            tickfont=dict(size=self.figTickSize.value, color=self.figTickColor.value),
            ticks=ticks, showgrid=show_grid,
            showline=show_line, linewidth=1, linecolor='black', mirror=show_line,
        )
        plotly_tmpl = go.layout.Template(layout=go.Layout(
            font=dict(family=self.figFontFamily.value),
            title=dict(font=dict(size=self.figTitleSize.value,
                                 color=self.figTitleColor.value)),
            legend=dict(font=dict(size=self.figLegendSize.value,
                                  color=self.figLegendColor.value)),
            paper_bgcolor=self.figBgColor.value,
            plot_bgcolor=self.figBgColor.value,
            xaxis=dict(**axis_layout, nticks=nticks_x),
            yaxis=dict(**axis_layout, nticks=nticks_y),
            width=self.figWidth.value, height=self.figHeight.value,
        ))
        self._templates[name] = {'template': plotly_tmpl, **self._template_extra_save()}
        self.figTemplateList.options = list(self._templates.keys())
        self.figTemplateName.value   = ''
        self._save_template_file(name)

    def _apply_selected_template(self, _=None) -> None:
        """Apply the selected template to the live figure and sync size widgets."""
        name = self.figTemplateList.value
        if not name or name not in self._templates:
            return
        entry = self._templates[name]
        tmpl  = entry['template']
        self.figure.update_layout(template=tmpl)
        w = tmpl.layout.width
        h = tmpl.layout.height
        if w:
            self.figWidth.value = w
        if h:
            self.figHeight.value = h
        self._template_extra_apply(entry)

    def _connect_figure_settings_observers(self) -> None:
        """Wire the Apply Settings, Save Template, and Apply Template buttons."""
        self.figFontApplyBtn.on_click(self._apply_figure_layout)
        self.figSaveTemplateBtn.on_click(self._save_as_template)
        self.figApplyTemplateBtn.on_click(self._apply_selected_template)
        # Auto-apply 'default' template if it was loaded from disk
        if 'default' in self._templates:
            self.figTemplateList.value = 'default'
            self._apply_selected_template()

    # keep old name as alias
    def _connect_font_observers(self) -> None:
        self._connect_figure_settings_observers()

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
            import plotly.graph_objects as go
            fig_export = go.Figure(self.figure)
            fig_export.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)')
            self._last_img_bytes = fig_export.to_image(format='png', scale=5)
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

    @staticmethod
    def _status_html(busy: bool, msg: str = 'Processing...') -> str:
        """HTML for the status dot shown in the session row."""
        if busy:
            return (f'<span style="font-size:11px;font-family:sans-serif;'
                    f'white-space:nowrap;color:#e67e22;">&#9679; {msg}</span>')
        return ('<span style="font-size:11px;font-family:sans-serif;'
                'white-space:nowrap;color:#27ae60;">&#9679; Ready</span>')

    def _set_busy(self, busy: bool, msg: str = 'Processing...') -> None:
        """Increment/decrement busy counter; update status widget at 0/1 transitions.

        Using a counter means nested calls (handler -> _redraw) clear correctly:
        the indicator stays orange until the outermost caller finishes.
        """
        if busy:
            self._busy_count += 1
            self._status_widget.value = self._status_html(True, msg)
        else:
            self._busy_count = max(0, self._busy_count - 1)
            if self._busy_count == 0:
                self._status_widget.value = self._status_html(False)

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

        1. **win32clipboard** (primary) — writes the PNG bytes directly to the
           clipboard using the registered PNG format.  Requires ``pywin32``.

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

                # Use the PNG clipboard format — preserves the alpha channel
                # and is understood by Word, PowerPoint, Illustrator, etc.
                fmt_png = win32clipboard.RegisterClipboardFormat('PNG')
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardData(fmt_png, self._last_img_bytes)
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
    # Notebook cell injection
    # ------------------------------------------------------------------

    def _inject_cell(self, code: str) -> None:
        """Insert code as a new Jupyter cell directly below the current one.

        JupyterLab/Notebook 7: insert-cell-below then replace-selection.
        Classic Notebook: Jupyter.notebook API fallback.
        json.dumps handles all escaping regardless of quotes or newlines.
        """
        import json
        from IPython.display import display, Javascript
        code_json = json.dumps(code)
        js = (
            "(function() {"
            "  var code = " + code_json + ";"
            "  var app = window.jupyterapp || window.jupyterlab;"
            "  if (app && app.commands) {"
            "    app.commands.execute('notebook:insert-cell-below').then(function() {"
            "      app.commands.execute('notebook:replace-selection', { text: code });"
            "    });"
            "    return;"
            "  }"
            "  if (window.Jupyter && window.Jupyter.notebook) {"
            "    var nb = window.Jupyter.notebook;"
            "    var cell = nb.insert_cell_below('code');"
            "    cell.set_text(code);"
            "    cell.select();"
            "  }"
            "})();"
        )
        display(Javascript(js))

    # ------------------------------------------------------------------
    # Settings panel visibility
    # ------------------------------------------------------------------

    def _set_settings_visibility(self, visible: bool) -> None:
        """Show or hide the settings accordion and all its descendant widgets."""
        import ipywidgets as widgets
        target = 'visible' if visible else 'hidden'
        self.v_settings_layout.layout.visibility = target
        for child in self.v_settings_layout.children:
            child.layout.visibility = target
            if hasattr(child, 'children'):
                for grandchild in child.children:
                    grandchild.layout.visibility = target
