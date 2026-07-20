"""
base_browser.py
---------------
Abstract base class shared by imageBrowser (SXM_browser) and spectrumBrowser (DAT_browser).

Rendering is matplotlib on the ipympl (widget) backend. The Canvas is built
directly from a Figure — no pyplot, no ``%matplotlib`` magic — which is the
construction path that works identically in VS Code and JupyterLab:

    fig    = Figure(...)          # plain matplotlib Figure
    canvas = Canvas(fig)          # ipympl Canvas IS an ipywidget
    FigureManager(canvas, 0)      # gives the canvas its toolbar

Widget attributes are accessed through ``self`` at call-time, so this module
is importable without a running Jupyter kernel (important for unit tests).
"""

import asyncio
import io
import os
import subprocess
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, MaxNLocator


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
    * ``self.rootFolder``          – widget with ``.value``
    * ``self.selectionList``       – widget with ``.value``, ``.index``
    * ``self.refreshBtn``          – Button widget (used for type-identity checks)
    * ``self.copyBtn``             – Button widget with ``.icon``
    * ``self.figure`` / ``self.canvas`` / ``self.ax`` – matplotlib Figure,
      ipympl Canvas widget, and main Axes (created via ``_make_figure``)
    * ``self.last_save_fname``     – str  (set by subclass ``save_figure``)
    * ``self.v_settings_layout``   – container widget with ``.children``, ``.layout``
    """

    #: One place to change raster resolution. figWidth/figHeight widgets are in
    #: pixels; matplotlib thinks in inches, so px / _DPI = inches everywhere.
    _DPI: int = 100
    _figure_format: str | None = None  # 'png' or 'svg' (None = default == png)
    # Directories containing these strings are never recursed into.
    _SKIP_DIRS: frozenset = frozenset({
        'browser_outputs',
        'ipynb',
        'raw_stml_data',
        'spmpy',
        '__pycache__',
    })

    # Suppresses reactive observers while widget values are set programmatically.
    _loading: bool = False

    # Reference counter for nested busy calls; status clears only at 0.
    _busy_count: int = 0

    # Debounce state (I1): shared 150 ms coalescing of observer-triggered redraws.
    _redraw_handle = None
    _data_dirty: bool = False

    # Export cache: True whenever the figure may differ from _last_img_bytes.
    _export_dirty: bool = True
    _last_img_bytes: bytes | None = None

    # Subclasses set this to 'sxm' or 'dat' to pick their template subdirectory
    _TEMPLATES_SUBDIR: str = ''

    # ------------------------------------------------------------------
    # Figure / canvas construction
    # ------------------------------------------------------------------

    def _make_figure(self, width_px: int, height_px: int) -> None:
        """Create self.figure (Figure), self.canvas (ipympl widget), self.ax.

        Direct Canvas construction avoids pyplot's global figure registry and
        backend magics — the failure modes seen in VS Code notebooks.
        """
        # _DPI is unconditionally 100. It is NOT an export knob — it is the
        # single inch↔pixel conversion the whole GUI is built on (SXM pixel
        # budgets, the _fig_*_px helpers, figsize math). Vector export ignores
        # savefig dpi entirely (verified: PDF at dpi=72 vs 300 is byte-
        # identical), so switching it for SVG/PDF changed layout for nothing.
        from ipympl.backend_nbagg import Canvas, FigureManager
        self.figure = Figure(figsize=(width_px / self._DPI, height_px / self._DPI),
                             dpi=self._DPI)
        self.canvas = Canvas(self.figure)
        FigureManager(self.canvas, 0)          # attaches the interactive toolbar
        self.canvas.header_visible = False     # hide "Figure 0" strip
        # Backend-authoritative sizing. resizable=False removes the drag
        # handle, but ipympl's JS still sends 'resize' messages (its
        # ResizeObserver fires on any layout reflow). The backend receiver —
        # FigureCanvasWebAggCore.handle_resize — does int(css_width)×DPR and
        # writes it into fig.set_size_inches [verified against ipympl 0.10 /
        # mpl 3.10 source]: the int() truncation shaves fractional pixels on
        # every round trip, ratcheting the figure smaller. Size flows one way
        # here (W/H widgets → figure), so drop frontend resizes entirely.
        self.canvas.resizable = False
        self.canvas.handle_resize = lambda *args, **kwargs: None
        self.ax = self.figure.add_subplot()

    @staticmethod
    def _ax_rect(margin: dict, w_px: int, h_px: int) -> list:
        """[left, bottom, width, height] figure fractions from pixel margins.

        Explicit positioning (instead of tight/constrained layout) keeps the
        plot box constant across redraws — no layout-engine drift."""
        l = margin['l'] / w_px
        b = margin['b'] / h_px
        return [l, b,
                max(0.05, 1 - (margin['l'] + margin['r']) / w_px),
                max(0.05, 1 - (margin['t'] + margin['b']) / h_px)]

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
        self.filenameText = Text_Widget('')
        self.errorText    = Selection_Widget([], 'Out:', rows=5)
        self.errorText.layout.width = '95%'

    def _build_main_layout(self, file_col, img_col, session_label_width: int = 5) -> None:
        """Assemble the top-level 3-column layout and set it as self.h_main_layout."""
        import ipywidgets as widgets
        from .widget_helpers import HBox, VBox

        self._status_widget = widgets.HTML(
            value=self._status_html(False),
            layout=widgets.Layout(flex='0 0 auto', margin='0 0 0 8px'),
        )

        file_col.layout.flex      = '0 0 auto'
        file_col.layout.width     = '250px'
        file_col.layout.min_width = '200px'

        img_col.layout.flex      = '1 1 auto'
        img_col.layout.width     = 'auto'
        img_col.layout.min_width = '300px'

        self.v_settings_layout.layout.flex      = '0 0 auto'
        self.v_settings_layout.layout.width     = '310px'
        self.v_settings_layout.layout.min_width = '280px'

        self._error_accordion = widgets.Accordion(
            children=[self.errorText], titles=['Messages'], selected_index=None,
            layout=widgets.Layout(width='98%'))

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
            HBox(children=[file_col, img_col, self.v_settings_layout], layout=_row),
            self._error_accordion],
            layout=widgets.Layout(display='flex', flex_flow='column', width='100%',
                                  justify_content='space-between', align_items='stretch'),)

    # ------------------------------------------------------------------
    # Figure styling
    # ------------------------------------------------------------------

    def _style_axes(self, ax) -> None:
        """Apply all Figure Settings widget values to *ax* (fonts, ticks,
        spines, grid, tick counts). Idempotent — safe after every rebuild."""
        fam       = self.figFontFamily.value
        show_line = self.figAxesBorderToggle.value

        ax.set_facecolor(self.figBgColor.value)
        ax.title.set_fontsize(self.figTitleSize.value)
        ax.title.set_color(self.figTitleColor.value)
        ax.title.set_fontfamily(fam)
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            lbl.set_fontsize(self.figAxesLabelSize.value)
            lbl.set_color(self.figAxesLabelColor.value)
            lbl.set_fontfamily(fam)
        # tick_params colors= sets both tick marks and tick labels
        ax.tick_params(direction='in' if self.figTicksMode.value == 'inside' else 'out',
                       labelsize=self.figTickSize.value, colors=self.figTickColor.value)
        for spine in ax.spines.values():
            spine.set_visible(show_line)
            spine.set_linewidth(1)
            spine.set_color('black')
        ax.grid(self.figGridToggle.value)
        for axis, count in ((ax.xaxis, self.figXTickCount.value),
                            (ax.yaxis, self.figYTickCount.value)):
            n = self._parse_tick_count(count)
            axis.set_major_locator(MaxNLocator(nbins=n) if n > 0 else AutoLocator())

    def _fig_w_px(self) -> int:
        """figWidth widget (inches) → pixels."""
        return int(round(self.figWidth.value * self._DPI))

    def _fig_h_px(self) -> int:
        """figHeight widget (inches) → pixels."""
        return int(round(self.figHeight.value * self._DPI))

    def _figure_layout_update(
        self,
        *,
        margin: dict | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Apply all figure appearance settings in one call.

        Invoked at startup and by the 'Apply Settings' button. *margin* (px)
        is only meaningful for fixed-margin browsers (SXM); browsers with a
        layout engine (DAT: constrained) omit it — the engine owns placement.
        Size defaults to the figWidth/figHeight widgets (inches × _DPI).
        Fonts default per-figure via rcParams family (one lab, one font).
        """
        w = width  if width  is not None else self._fig_w_px()
        h = height if height is not None else self._fig_h_px()
        mpl.rcParams['font.family'] = self.figFontFamily.value
        self._margin_px = margin   # last-applied pixel margins (debug/inspection)
        self.figure.set_size_inches(w / self._DPI, h / self._DPI, forward=True)
        self.figure.set_facecolor(self.figBgColor.value)
        if getattr(self, 'ax', None) is not None:
            # Manual set_position would detach the axes from a layout engine.
            if margin is not None and self.figure.get_layout_engine() is None:
                self.ax.set_position(self._ax_rect(margin, w, h))
            self._style_axes(self.ax)
        self._export_dirty = True
        self.canvas.draw_idle()

    @staticmethod
    def _resolve_cmap(name: str, reverse: bool = False):
        """Return a matplotlib Colormap from *name* (case-insensitive);
        reverse=True appends '_r'. Falls back to 'Greys' on unknown names."""
        if name not in mpl.colormaps:
            name = next((c for c in mpl.colormaps if c.lower() == name.lower()),
                        'Greys')
        if reverse:
            name = name[:-2] if name.endswith('_r') else name + '_r'
        return mpl.colormaps[name]

    def _build_figure_settings_widgets(self, fig_width: int = 600, fig_height: int = 500) -> None:
        """Create figure settings widgets (size, bg, axes, fonts, templates)."""
        import ipywidgets as widgets
        W98    = widgets.Layout(display='flex', width='98%')
        fWidth = lambda p: widgets.Layout(display='flex', width=f'{p}%')
        ds     = lambda px: {'description_width': f'{px}px'}

        # Size — displayed in INCHES (matplotlib's native unit, so
        # set_size_inches receives the widget value directly). Constructor
        # params arrive in px for backward compatibility; ÷ _DPI converts.
        self.figWidth  = widgets.BoundedFloatText(
            value=round(fig_width / self._DPI, 2), min=1, max=40, step=0.1,
            description='W (in):', layout=fWidth(48), style=ds(46))
        self.figHeight = widgets.BoundedFloatText(
            value=round(fig_height / self._DPI, 2), min=1, max=40, step=0.1,
            description='H (in):', layout=fWidth(48), style=ds(46))
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

        # Export format. Copy always puts a PNG on the clipboard (the OS
        # clipboard can't hold SVG/PDF as an image); the saved file uses this.
        # Seed from any constructor-supplied _figure_format so both paths agree.
        _fmt = (self._figure_format or 'png').lower()
        self.figFormat = widgets.Dropdown(
            options=['png', 'svg', 'pdf', 'eps'],
            value=_fmt if _fmt in ('png', 'svg', 'pdf', 'eps') else 'png',
            description='Save as:', layout=W98, style=ds(60))
        self._figure_format = self.figFormat.value

        # Templates
        self._templates: dict = {}
        self._templates_dir = (
            Path(__file__).parent / 'templates' / self._TEMPLATES_SUBDIR)
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        self.figTemplateName = widgets.Text(
            value='', placeholder='template name', description='Tmpl:',
            layout=fWidth(64), style=ds(34))
        self.figSaveTemplateBtn  = widgets.Button(
            description='Save', tooltip='Save current settings as a template',
            layout=fWidth(34))
        self.figTemplateList     = widgets.Select(options=[], rows=4, layout=W98)
        self.figApplyTemplateBtn = widgets.Button(description='Apply Template', layout=W98)
        self._load_templates_dir()

    def _figure_settings_extras(self) -> list:
        """Hook: browser-specific widgets inserted before the Apply button.

        SXM contributes the ctx-legend width; DAT contributes the slack-margin
        row. Returning widgets here (instead of hasattr probing in the base)
        keeps one tab layout compatible with every browser."""
        return []

    def _figure_settings_tab(self):
        """Compact Figure Settings tab.

        Every control carries its own description, so the old '-- Section --'
        Label rows are dead weight — dropped. Related size+color pairs share a
        row; template name and Save share a row. Reading order is unchanged:
        size → background → axes → fonts → extras → apply → templates."""
        import ipywidgets as widgets
        W98 = widgets.Layout(display='flex', width='98%')
        HB  = lambda *ws: widgets.HBox(children=list(ws), layout=W98)
        return widgets.VBox(children=[
            HB(self.figWidth, self.figHeight),
            self.figBgColor,
            HB(self.figAxesBorderToggle, self.figGridToggle),
            self.figTicksMode,
            HB(self.figXTickCount, self.figYTickCount),
            self.figFontFamily,
            HB(self.figTitleSize, self.figTitleColor),
            HB(self.figAxesLabelSize, self.figAxesLabelColor),
            HB(self.figTickSize, self.figTickColor),
            HB(self.figLegendSize, self.figLegendColor),
            self.figFormat,
            *self._figure_settings_extras(),
            self.figFontApplyBtn,
            HB(self.figTemplateName, self.figSaveTemplateBtn),
            self.figTemplateList,
            self.figApplyTemplateBtn,
        ], layout=widgets.Layout(visibility='hidden', display='flex',
                                 width='98%', flex_flow='column'))

    @staticmethod
    def _parse_tick_count(val: str) -> int:
        """Return tick-count int (0 = auto) from 'auto' or numeric string."""
        try:
            return max(0, int(val))
        except (ValueError, TypeError):
            return 0

    def _apply_figure_layout(self, _=None) -> None:
        """'Apply Settings' callback. Subclasses override to supply browser-specific args."""
        self._figure_layout_update(margin=dict(l=60, r=30, t=60, b=50))

    # ------------------------------------------------------------------
    # Templates — plain widget-settings dicts (JSON)
    #
    # v0.2 format: {'name': ..., 'settings': {...}, <subclass extras>}.
    # Legacy v0.1 files carried a plotly 'template' blob; that key is now
    # simply ignored, so old files still load (extras like SXM labels apply).
    # ------------------------------------------------------------------

    _SETTINGS_WIDGETS = (
        ('font_family',      'figFontFamily'),
        ('title_size',       'figTitleSize'),
        ('title_color',      'figTitleColor'),
        ('axes_label_size',  'figAxesLabelSize'),
        ('axes_label_color', 'figAxesLabelColor'),
        ('tick_size',        'figTickSize'),
        ('tick_color',       'figTickColor'),
        ('legend_size',      'figLegendSize'),
        ('legend_color',     'figLegendColor'),
        ('bg_color',         'figBgColor'),
        ('axes_border',      'figAxesBorderToggle'),
        ('grid',             'figGridToggle'),
        ('ticks_mode',       'figTicksMode'),
        ('x_ticks',          'figXTickCount'),
        ('y_ticks',          'figYTickCount'),
        ('width',            'figWidth'),
        ('height',           'figHeight'),
        ('save_format',      'figFormat'),
    )

    def _collect_settings(self) -> dict:
        """Snapshot the Figure Settings widget values into a JSON-safe dict."""
        return {key: getattr(self, wname).value
                for key, wname in self._SETTINGS_WIDGETS}

    def _apply_settings(self, settings: dict) -> None:
        """Write *settings* back to the widgets (observers suppressed)."""
        self._loading = True
        try:
            for key, wname in self._SETTINGS_WIDGETS:
                if key in settings:
                    val = settings[key]
                    # Legacy templates stored width/height in px; the widgets
                    # are now inches. Values above any plausible inch count
                    # are px — convert. (40 in = widget max.)
                    if key in ('width', 'height') and isinstance(val, (int, float)) \
                            and val > 60:
                        val = round(val / self._DPI, 2)
                    try:
                        getattr(self, wname).value = val
                    except Exception:
                        pass   # e.g. font no longer in dropdown options
        finally:
            self._loading = False

    def _template_extra_save(self) -> dict:
        """Hook: extra key/value pairs stored alongside the settings dict."""
        return {}

    def _template_extra_apply(self, entry: dict) -> None:
        """Hook: apply extra data stored in *entry* back to the browser widgets."""
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
        try:
            fpath.write_text(json.dumps({'name': name, **entry}, indent=2))
        except Exception as err:
            self.updateErrorText(f'template save error: {err}')

    def _load_templates_dir(self) -> None:
        """Scan the templates directory and load every JSON file on startup."""
        import json
        for fpath in sorted(self._templates_dir.glob('*.json')):
            try:
                data = json.loads(fpath.read_text())
                name = data.pop('name', fpath.stem)
                data.pop('template', None)   # legacy plotly blob: ignored
                data.setdefault('settings', {})
                self._templates[name] = data
            except Exception as err:
                self.updateErrorText(f'template load error ({fpath.name}): {err}')
        if self._templates:
            self.figTemplateList.options = list(self._templates.keys())

    def _save_as_template(self, _=None) -> None:
        """Capture current Figure Settings as a named template entry."""
        name = self.figTemplateName.value.strip()
        if not name:
            self.updateErrorText('Enter a template name before saving')
            return
        self._templates[name] = {'settings': self._collect_settings(),
                                 **self._template_extra_save()}
        self.figTemplateList.options = list(self._templates.keys())
        self.figTemplateName.value   = ''
        self._save_template_file(name)

    def _apply_selected_template(self, _=None) -> None:
        """Apply the selected template: widgets first, then one layout pass."""
        name = self.figTemplateList.value
        if not name or name not in self._templates:
            return
        entry = self._templates[name]
        self._apply_settings(entry.get('settings', {}))
        self._template_extra_apply(entry)
        self._apply_figure_layout()

    def _connect_figure_settings_observers(self) -> None:
        """Wire the Apply Settings, Save Template, and Apply Template buttons."""
        self.figFontApplyBtn.on_click(self._apply_figure_layout)
        self.figSaveTemplateBtn.on_click(self._save_as_template)
        self.figApplyTemplateBtn.on_click(self._apply_selected_template)
        self.figFormat.observe(self._on_format_change, names='value')
        if 'default' in self._templates:
            self.figTemplateList.value = 'default'
            self._apply_selected_template()

    def _on_format_change(self, change) -> None:
        """Sync _figure_format and invalidate the byte cache.

        The cache is format-agnostic, so a stale PNG must not be written into
        an .svg/.pdf file — force a re-render on the next export."""
        self._figure_format = change['new']
        self._export_dirty = True

    def _export_format(self) -> str:
        """Current export format ('png' when unset)."""
        return (self._figure_format or 'png').lower()

    @staticmethod
    def _editable_font_rcparams():
        """Context manager pinning SVG/PDF text to remain editable text (not
        outlined paths) in Illustrator/Inkscape.

        svg.fonttype='none'  → real <text> elements instead of vector outlines.
        pdf.fonttype=42      → embedded TrueType (selectable/editable) instead
                               of the default Type-3 outlines.
        """
        import matplotlib as mpl
        return mpl.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    # ------------------------------------------------------------------
    # Redraw debounce (I1) — shared by both browsers
    # ------------------------------------------------------------------

    def _schedule_redraw(self, *_, data_dirty: bool = False) -> None:
        """Debounced entry point for observer callbacks.

        Coalesces rapid widget events into a single render after a 150 ms
        quiescence window on the kernel's asyncio loop. data_dirty=True signals
        that update_image_data() must run before the render.
        """
        if self._loading:
            return
        if data_dirty:
            self._data_dirty = True
        if self._redraw_handle is not None:
            self._redraw_handle.cancel()
            self._redraw_handle = None
        try:
            loop = asyncio.get_running_loop()
            self._redraw_handle = loop.call_later(0.15, self._execute_redraw)
        except RuntimeError:
            self._execute_redraw()

    def _schedule_redraw_dirty(self, *_) -> None:
        """Convenience observer target for data-reload (tier-3) widgets."""
        self._schedule_redraw(data_dirty=True)

    def _execute_redraw(self) -> None:
        """Run after the debounce window; reprocess data if dirty, then redraw."""
        self._redraw_handle = None
        try:
            if self._data_dirty:
                self._data_dirty = False
                self.update_image_data()
            self._redraw()
        except Exception as err:
            # asyncio callbacks bypass ipywidgets' handler try/excepts (N19)
            self.updateErrorText(f'render error: {err}')

    def _refresh_info(self, *_) -> None:
        """Tier-2 observer: rebuild title/info text then redraw. No data reprocessing."""
        self.update_scan_info()
        self._redraw()

    def _figure_stem(self, dir_name: str) -> str:
        """Return the filename stem (without note or extension) for save_figure."""
        raise NotImplementedError

    def _current_dir_name(self) -> str:
        """Name of the currently selected directory for use in filenames.

        Taken from the internal Path list (self.directories), NOT from
        directorySelection.value — the widget shows decorated display strings
        (tree glyphs ├─ │ 📁) that must never reach the filesystem."""
        try:
            return Path(self.directories[self.directorySelection.index]).resolve().name
        except Exception:
            return Path(self.active_dir).resolve().name

    # ------------------------------------------------------------------
    # Export / copy — Agg savefig, no external renderer
    # ------------------------------------------------------------------

    def _render_export_bytes(self, fmt: str | None = None) -> bytes:
        """Render the live figure to *fmt* bytes (default: the chosen format),
        trimmed to content.

        Figure.savefig always rasterises through Agg for PNG regardless of the
        interactive canvas. bbox_inches='tight' makes the renderer compute the
        union bounding box of every artist (title, colorbar labels, figure
        texts included) and crop to it — the on-screen slack margins never
        reach the export. The plot area itself stays figWidth×figHeight.

        Vector formats (svg/pdf/eps) ignore dpi entirely and are written at the
        figsize physical size; dpi still governs any embedded raster (e.g. the
        SXM heatmap). SVG/PDF text is kept editable (see _editable_font_rcparams)."""
        fmt = (fmt or self._export_format()).lower()
        buf = io.BytesIO()
        with self._editable_font_rcparams():
            self.figure.savefig(buf, format=fmt, dpi=self.figure.dpi,
                                transparent=True,
                                bbox_inches='tight', pad_inches=0.03)
        return buf.getvalue()

    def _export_bytes_cached(self) -> bytes:
        """Bytes for the currently selected format, reusing the last render
        when nothing changed AND the format matches.

        _export_dirty is set by every render/style path and by a format change,
        so a cached PNG can never be written into an .svg/.pdf file."""
        fmt = self._export_format()
        if (not self._export_dirty and self._last_img_bytes is not None
                and getattr(self, '_last_img_fmt', None) == fmt):
            return self._last_img_bytes
        self._last_img_bytes = self._render_export_bytes(fmt)
        self._last_img_fmt   = fmt
        self._export_dirty   = False
        return self._last_img_bytes

    def _save_path(self) -> str:
        """browser_outputs/<stem>[_note].<format> for the current selection."""
        out_dir = self.active_dir / 'browser_outputs'
        out_dir.mkdir(exist_ok=True)
        stem = self._figure_stem(self._current_dir_name())
        if self.saveNote.value:
            stem += f'_{self.saveNote.value}'
        return str(out_dir / f'{stem}.{self._export_format()}')

    def save_figure(self, a=None) -> None:
        """Save the current figure to browser_outputs/ in the chosen format."""
        self.saveBtn.icon = 'hourglass-start'
        try:
            self.last_save_fname = self._save_path()
            Path(self.last_save_fname).write_bytes(self._export_bytes_cached())
            self.updateErrorText(f'Figure Saved ({self._export_format()})')
        except Exception as err:
            self.updateErrorText(f'Save error: {err}')
        finally:
            self.saveNote.value = ''
            self.saveBtn.icon = 'file-image-o'

    # ------------------------------------------------------------------
    # Error / status output
    # ------------------------------------------------------------------

    def updateErrorText(self, text: str) -> None:
        """Append *text* to the error log and refresh the output widget."""
        self.errors.append(f'{len(self.errors)} {text}')
        self.errorText.options = self.errors
        acc = getattr(self, '_error_accordion', None)
        if acc is not None:
            acc.selected_index = 0

    @staticmethod
    def _status_html(busy: bool, msg: str = 'Processing...') -> str:
        """HTML for the status dot shown in the session row."""
        if busy:
            return (f'<span style="font-size:11px;font-family:sans-serif;'
                    f'white-space:nowrap;color:#e67e22;">&#9679; {msg}</span>')
        return ('<span style="font-size:11px;font-family:sans-serif;'
                'white-space:nowrap;color:#27ae60;">&#9679; Ready</span>')

    def _set_busy(self, busy: bool, msg: str = 'Processing...') -> None:
        """Increment/decrement busy counter; update status widget at 0/1 transitions."""
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

    def _scan_directory_tree(self, root: Path, _depth: int = 0,
                             max_depth: int = 3) -> list:
        """Return [(Path, depth), …] for *root* and every non-skipped
        subdirectory, in pre-order, alphabetical at each level. Recursion is
        capped at *max_depth* (deep network shares must not freeze the kernel);
        truncation warns once per scan via the _tree_truncated latch."""
        result = [(root, _depth)]
        if _depth >= max_depth:
            if not getattr(self, '_tree_truncated', False):
                self._tree_truncated = True
                self.updateErrorText(f'Directory tree truncated at depth {max_depth}')
            return result
        try:
            children = sorted(
                (Path(e.path) for e in os.scandir(root)
                 if e.is_dir() and not any(s in e.name for s in self._SKIP_DIRS)),
                key=lambda p: p.name.lower(),
            )
        except PermissionError:
            return result
        for child in children:
            result.extend(self._scan_directory_tree(child, _depth + 1, max_depth))
        return result

    def _build_tree_options(self, tree: list) -> list:
        """Convert [(Path, depth), …] into display strings with Unicode
        tree-drawing characters so parent–child relationships are visible."""
        if not tree:
            return []
        display = ['📁 session folder']
        for i, (path, depth) in enumerate(tree[1:], 1):
            is_last = True
            for j in range(i + 1, len(tree)):
                if tree[j][1] == depth:
                    is_last = False
                    break
                if tree[j][1] < depth:
                    break
            prefix = ''
            for d in range(1, depth):
                has_continuation = False
                for j in range(i + 1, len(tree)):
                    if tree[j][1] == d:
                        has_continuation = True
                        break
                    if tree[j][1] < d:
                        break
                prefix += '│  ' if has_continuation else '   '
            connector = '└─ ' if is_last else '├─ '
            display.append(prefix + connector + path.name)
        return display

    def _refresh_directory_tree(self) -> None:
        """Rescan from active_dir and update self.directories + widget options."""
        self._tree_truncated = False
        tree = self._scan_directory_tree(self.active_dir)
        self.directories = [path for path, _ in tree]
        self.directorySelection.options = self._build_tree_options(tree)

    # ------------------------------------------------------------------
    # File output helpers
    # ------------------------------------------------------------------

    def copy_figure(self, _event=None) -> None:
        """Copy the current figure to the clipboard, and — when saveBtn is on
        (default) — also save it to browser_outputs/ in the chosen format.

        The OS clipboard cannot hold SVG/PDF as an image, so the clipboard
        always receives a PNG: when the export format IS png the saved bytes
        are reused; for a vector format a separate PNG is rendered just for the
        clipboard while the file on disk stays vector."""
        self.copyBtn.icon = 'hourglass-half'
        try:
            fmt = self._export_format()
            save_bytes = self._export_bytes_cached()           # chosen format
            png_bytes  = save_bytes if fmt == 'png' else self._render_export_bytes('png')
            fname_for_ps = None
            if getattr(getattr(self, 'saveBtn', None), 'value', True):
                self.last_save_fname = self._save_path()
                Path(self.last_save_fname).write_bytes(save_bytes)
                # PowerShell fallback needs a PNG path; a vector file won't do.
                fname_for_ps = self.last_save_fname if fmt == 'png' else None
                self.updateErrorText(f'Figure copied and saved ({fmt})')
            else:
                self.updateErrorText('Figure copied')
            # --- clipboard: in-process win32 first, PowerShell fallback ---
            try:
                import win32clipboard
                fmt_png = win32clipboard.RegisterClipboardFormat('PNG')
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardData(fmt_png, png_bytes)
                finally:
                    win32clipboard.CloseClipboard()
            except ImportError:
                if fname_for_ps is None:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    tmp.write(png_bytes); tmp.close()
                    fname_for_ps = tmp.name
                ps_cmd = f'Set-Clipboard -Path "{fname_for_ps}"'
                result = subprocess.run(['powershell', '-Command', ps_cmd],
                                        capture_output=True, timeout=10)
                if result.returncode != 0:
                    self.updateErrorText(
                        f'clipboard error: {result.stderr.decode(errors="replace").strip()}')
        except Exception as err:
            self.updateErrorText(f'copy error: {err}')
        finally:
            self.saveNote.value = ''
            self.copyBtn.icon = 'clipboard'

    # ------------------------------------------------------------------
    # Root-folder / refresh handler
    # ------------------------------------------------------------------

    def handler_root_folder_update(self, event=None) -> None:
        """Called when ``rootFolder`` text changes or ``refreshBtn`` is clicked.

        Validates the typed path; if valid, updates ``active_dir`` and rebuilds
        the directory tree. A refreshBtn trigger restores the previous selection."""
        new_root = self.rootFolder.value

        current_directory = None
        if type(event) == type(self.refreshBtn):
            current_directory = self.directorySelection.value

        if os.path.exists(new_root) and os.path.isdir(new_root):
            self.active_dir = Path(new_root)
            self._refresh_directory_tree()
        elif new_root:
            self.updateErrorText(f'Invalid session path: {new_root}')

        if current_directory is not None:
            try:
                self.directorySelection.value = current_directory
            except Exception:
                pass  # selection may no longer exist after a root change

    # ------------------------------------------------------------------
    # Notebook cell injection
    # ------------------------------------------------------------------

    def _inject_cell(self, code: str) -> None:
        """Insert code as a new Jupyter cell directly below the current one."""
        import json
        from IPython.display import display, Javascript
        code_json = json.dumps(code)
        js = (
            "(function() {"
            "  var code = " + code_json + ";"
            "  var app = window.jupyterapp || window.jupyterlab;"
            "  if (app) {"
            "    var nb = app.shell.currentWidget;"
            "    if (nb && nb.content && nb.content.activeCell) {"
            "      nb.content.activeCell.model.sharedModel.setSource(code);"
            "    }"
            "  } else if (window.Jupyter) {"
            "    var cell = Jupyter.notebook.get_selected_cell();"
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
        """Show or hide the settings column (display:none actually collapses
        the 310 px column back to the figure)."""
        self.v_settings_layout.layout.display    = 'flex' if visible else 'none'
        self.v_settings_layout.layout.visibility = 'visible' if visible else 'hidden'
        target = 'visible' if visible else 'hidden'
        for child in self.v_settings_layout.children:
            child.layout.visibility = target
            if hasattr(child, 'children'):
                for grandchild in child.children:
                    grandchild.layout.visibility = target
