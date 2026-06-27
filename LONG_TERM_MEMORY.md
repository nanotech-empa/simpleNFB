# simpleNFB — Long Term Memory

---

## 2026-06-14

### process_utils.py (new)
Standalone processing functions extracted from browser classes and added:
`rebin_intensity_nm_to_ev`, `smooth_data`, `group_average`, `relative_position`,
`remove_line_average`, `despike_z_score`, `moving_average`.

### DAT_browser.py
- Methods `relative_position`, `group_average`, `rebin_intensity_nm_to_ev` removed; delegated to `process_utils`
- `smooth_data` kept as thin widget-param wrapper
- `despike_z_score` and `moving_average` added as new filter options (widgets, accordion panel, observers, pipeline)
- Filter pipeline order: threshold → despike → SVG → moving average → median

### SXM_browser.py
- `remove_line_average` method removed; delegated to `process_utils`

### base_browser.py — `copy_figure`
- Hourglass icon moved before `save_figure`
- In-memory 150 dpi render via `BytesIO` for clipboard
- Primary path: `win32clipboard` + Pillow DIB (no subprocess)
- Fallback: PowerShell f-string with `timeout=10` and `updateErrorText` on failure
- `try/except/finally` ensures icon always resets

### Tests
- `tests/conftest.py` deleted
- `tests/test_phase1_base_browser.py` deleted
- `tests/run_tests.py` deleted (hardcoded path to deleted test file)

---

## 2026-06-14 — Pip library conversion

### pyproject.toml (new)
PEP 621 metadata; spmpy sourced via `git+https://github.com/nanotech-empa/spmpy.git`;
optional `[clipboard]` extra for pywin32; excludes `support_scripts*` and `spmpy*`.

### Internal imports → relative
`DAT_browser.py`, `SXM_browser.py`: removed `sys.path.append('./spmpy')`, converted
sibling imports to `.` prefix. `spe_browser.py`: `import spe_loader` → `from . import spe_loader`,
removed stale `importlib.reload(spe_loader)`. Removed unused `import sys` from SXM/DAT.

### simpleNFB/spmpy/ deleted
Vendored copy removed; replaced by GitHub URL dependency in pyproject.toml.

### simpleNFB/__init__.py rewritten
Exports `imageBrowser`, `spectrumBrowser`, `Spe_Browser`, and all `process_utils` functions.
Proper `__all__`.

### simpleNFB/snfb.py cleaned
Stripped all bare absolute imports and `importlib.reload` calls. Now a two-line re-export shim:
`sxmBrowser = imageBrowser`, `datBrowser = spectrumBrowser` via relative imports.

### Notebooks/sNFB_Template.ipynb patched
`import snfb` → `import simpleNFB.snfb as snfb`; `import spmpy` line removed.
`importlib.reload(snfb)` unchanged — works correctly since `snfb` is now a module object.

---

## 2026-06-14 — SXM_browser.py + DAT_browser.py full rewrite

### SXM_browser.py
- `__init__` decomposed into `_build_widgets`, `_build_layout`, `_connect_observers`
- Renames: `updateDisplayImage→_redraw`, `updateChannelSelection→_update_channel_selection`,
  `updateInfoText→_update_info_text`, `addFigureLabels→_add_figure_labels`
- `_updating_limits` flag replaces unobserve/set/reobserve pattern on vmin/vmax
- `_scan_cache` dict replaces repeated `get_param` calls (cleared on file load)
- `_ALIGN_PARAMS` class dict replaces 4-branch if-chain in `_add_figure_labels`
- `Path.mkdir(exist_ok=True)` in `save_figure`; `mouse_click` stub (body was broken)
- Bug fixed: `names='values'` → `names='value'` on limit observers

### DAT_browser.py
- `__init__` decomposed into `_build_widgets`, `_build_layout`, `_connect_observers`
- mpl.rcParams block moved from module level into `__init__`
- Renames: `updateDisplayImage→_redraw`, `updateChannelSelection→_update_channel_selection`,
  `updateInfoText→_update_info_text`, `updateAxesLimitSliders→_sync_axis_limit_sliders`,
  `updateLegendParameterSelection→_update_legend_parameters`
- `update_axes` decomposed: `_apply_filters`, `_build_legend_labels`, `_plot_spectra`, `_add_stml_axis`
- `update_scan_info` uses `_INFO_BUILDERS` dispatch dict + per-experiment `_info_*` methods
- `_cache_scan_param` replaces repeated `get_param` calls
- `bisect.bisect_left([4,16], len(ax.lines))` for legend fontsize selection
- `np.column_stack` for CSV export; `Path.mkdir(exist_ok=True)` for output dirs
- Bugs fixed: `referenceLocBtn.disabled=True` (was called as method), `csvBtn.icon` (was saveBtn),
  `assert` → error text + return, removed `print('update axes called')`,
  `y_values = [a.copy() for a in self.spec_data]`, `self.wfAxes = None` guard,
  `spec_index` always treated as list, `a['owner'] == self.parameterLegendToggle` (not `.value`)
- All `!= None`/`== None` → `is not None`/`is None`; dead `else: pass` blocks removed
- Dead `return spec, spec_data, spec_x` from STML branch removed

---

## 2026-06-26 — DAT_browser legend tool replacement

### DAT_browser.py
- Removed: `defaultLegendToggle`, `customLegendToggle`, `parameterLegendToggle`, `update_legend_mode`
- Added: `legendModeToggle` (ToggleButtons: Parameter / Custom / Condensed)
- `parameterLegendList` stays; `legendText`, `legendEntry`, `legendUpdate` kept but initially hidden (`display='none'`); shown only in Custom mode
- `_apply_legend_mode_visibility(mode)`: toggles `display` on parameter vs custom section widgets
- `_on_legend_mode_change(change)`: calls visibility helper + `_redraw()` (guarded by `_loading`)
- `_detect_numbering_pattern()`: returns `(min_name, max_name)` if ≥10 selected files all contain an embedded number; else `None`
- `_condensed_range_labels()`: calls `_detect_numbering_pattern`, falls back to first/last filename
- `update_axes`: in Condensed mode, suppresses per-trace legend and appends a dummy `go.Scatter` with `marker.showscale=True` / colorbar ticked at `[min_name, max_name]`
- `_update_legend_on_load`: auto-activates Condensed mode when pattern detected (under `_loading` so no premature redraw)
- `parameterLegendList` observer moved from `_update_display` (display-only) → `_redraw` (tier 1); needed because condensed mode uses full rebuild
- `import re` added at top of file

---

## 2026-06-27 — SXM_browser colorbar simplification

### SXM_browser.py
- Removed: `_colorbar_params`, `_reposition_colorbar`, `_on_figure_relayout`, `_rendered_fig_w`, `scaleanchor`/`constrain` axis attrs
- Root insight: with `scaleanchor + constrain='domain'` the image doesn't fill the full plot area, so computing the image's position in paper coords (and tracking rendered width via JS relayout) was required. Eliminated by shaping the figure correctly instead.
- New approach: `autosize=False`; `_update_fig_width(h_crop, w_crop)` computes `width = 60 + plot_h*(w_crop/h_crop) + 60` from the known `figHeight.value`. Plot area aspect matches image → image fills plot area entirely → default colorbar `x=1.02` and `len=1.0` are correct with no geometry calculation.
- Added `_apply_figure_layout` override: calls `super()` then reapplies `autosize=False, width=...` (base sets `autosize=True` which is overridden).
- Colorbar now uses `len=1.0, y=0.5, yanchor='middle'`; no `x` override needed.

## 2026-06-26 — SXM_browser Filter Settings panel expansion

### SXM_browser.py
- `linebylineBtn`, `flattenBtn`, `fixZeroBtn` removed from toolbar (`h_process_btn_layout`) — now live exclusively in the Filter Settings accordion tab
- Toolbar shrunk to `directionBtn` + `invertBtn` (view/sign controls, not filters)
- Filter Settings VBox rebuilt: section labels ('── Corrections ──', '── Filters ──'), then three labeled HBox rows for the correction buttons, then three HBox rows for Gaussian/Median/Laplace with their size inputs
- All six filter widgets already wired to `redraw_image` — no observer changes needed

---

## 2026-06-26 — DAT_browser '1D Plot' panel

### DAT_browser.py
- `'Axes Controls'` accordion tab replaced with `'1D Plot'` (index 5)
- New widgets: `xLabel1D`, `yLabel1D` (Text, label override); `xScaleMode`, `yScaleMode` (ToggleButtons: Linear/Log/Custom); `xCustomFormula`, `yCustomFormula` (Text, hidden until Custom selected)
- `update_axes`: applies scale mode (`xaxis.type='log'` or `'linear'`); evals custom formula via `_apply_axis_scale`; uses label override when non-empty, auto-generates otherwise
- `_apply_axis_scale(data_list, formula, var)`: `eval(formula, {'np': np, var: arr})` per array; results cast to ndarray
- `_toggle_formula_visibility(change)`: flips `layout.display` on formula Text when scale mode changes
- All 6 new widgets wired to `_redraw` (tier 1); scale mode widgets additionally wired to visibility toggle

---

## 2026-06-26 — DAT_browser file selection audit fixes

### DAT_browser.py
- `_update_display`: added `if self._loading: return` guard — prevents patching stale traces during file load
- `update_image_data`: removed dead `filename` parameter and else-branch; hoisted `get_channel(channelX)` out of multi-channel Y loop (was called N× for the same channel on the same file; now called once)
- `_update_legend_parameters` deleted; replaced by `_update_legend_on_load()` which combines header-key refresh + legendText sync under a single `_loading` block
- `update_legend_settings`: removed `selectionList` owner path (now in `_update_legend_on_load`); guard dropped, always runs when called by `groupSize`/`averageToggle` observers
- `handler_file_selection`, `nextDisplay`, `previousDisplay`: all now call `_update_legend_on_load()` after `load_new_image()` — navigation paths previously skipped legend parameter sync on experiment-type change

## 2026-06-25 — Observer tier + file selection optimizations

### base_browser.py
- `_loading: bool = False` class attribute suppresses reactive observers during programmatic widget changes
- `_refresh_info()` helper: tier-2 observer (`update_scan_info() + _redraw()`, no data pipeline)

### SXM_browser.py
- Replaced catch-all settings loop with explicit tier groups (tier 1: display; tier 2: info text; tier 3: filters)
- Removed dead observers: `edgesBtn.observe`, `gaussianBtn.observe`; deleted `handler_settingsChange`
- `load_new_image`: wraps `_update_channel_selection()` with `_loading=True` — stops spurious channel cascade
- Navigation + folder-selection: suppress `handler_file_selection` with `_loading`; call load+redraw once
- Result: 2× `update_image_data` + 2× `_redraw` per file load → 1× each

### DAT_browser.py
- Same tier-group pattern; removed catch-all loop and double `stmlToggle` binding
- `legendEntry`: `continuous_update=False` (no keystroke-by-keystroke redraws)
- `cmapSelection` → tier 1; `plasmonReference` single observer with `_loading` guard + `_redraw()` at end
- `handler_settingsChange` deleted; `update_scan_info` guards `loaded_experiments is None`
- `handler_folder_selection`: `os.listdir` + per-file `getmtime` → `os.scandir` (cached stats)
- Result: 3× `update_image_data` + 3-4× `_redraw` per file load → 1× each

---

## 2026-06-24 — matplotlib → plotly migration

### pyproject.toml
- Removed `matplotlib>=3.5`; added `plotly>=5.18`; added `[export]` extra: `kaleido>=0.2`

### base_browser.py — copy_figure
- `self.figure.savefig(buf, dpi=150, ...)` → `io.BytesIO(self.figure.to_image(format='png', scale=2))`
- Docstring updated: "plotly FigureWidget" replaces "matplotlib Figure"

### SXM_browser.py
- Imports: removed mpl/mpl_toolkits; added plotly.graph_objects, plotly.express
- `plt.subplots()` + `widgets.Output` → `go.FigureWidget(data=[go.Heatmap(...)])`
- `cmapSelection` options: `px.colors.named_colorscales()` (lowercase plotly names)
- Added `reverseScaleToggle` ToggleButton (replaces matplotlib `_r` suffix convention)
- `cmap` __init__ param: strips `_r` suffix → sets initial reversescale flag
- `update_axes`: uses `fig.batch_update()`, updates Heatmap trace in-place
- Scan direction: `scan_dir='down'` → `z = data[::-1]` (replaces `origin='upper'`)
- `_add_figure_labels`: plotly `add_annotation`/`add_shape` with `xref/yref='paper'`
- `save_figure`: `fig.write_image(fname, scale=5)` with kaleido error hint
- `scan_info` uses `<br>` separators (plotly HTML title rendering)
- `imageBrowser = fileBrowser` alias at module bottom

### DAT_browser.py
- Imports: removed mpl/mpl_toolkits/AutoMinorLocator; added plotly
- `mpl.rcParams` block, `_aspect`/`_resizing` attrs, `_on_figure_draw` removed
- `plt.subplots()` → `go.FigureWidget()` empty figure, single initial Scatter trace
- `self.axes`, `self.axes2`, `self.cb` removed; layout via `fig.update_layout`
- `figure.canvas` in VBox → `self.figure` (FigureWidget is itself a widget)
- `_plot_spectra(ax, ...)` → `_plot_spectra(...)` using `fig.add_trace(go.Scatter(...))`
- `_add_stml_axis(ax)` → `_add_stml_axis()` using `fig.update_layout(xaxis2=dict(...))`
- `update_axes`: clears with `fig.data = ()`, minor ticks via `minor=dict(ticks='inside')`
- `_sync_axis_limit_sliders`: reads `fig.layout.xaxis.range`; falls back to data range
- `handler_update_axes_limits`: `fig.update_layout(xaxis/yaxis=dict(range=[...]))`
- `_render_2d`: `ax.pcolormesh` → `go.Heatmap`; colorbar built into trace
- `generateWaterFall`: redirects to 2D view (sets plot2DToggle + plot2DYParam)
- `plotSpectrumLocations`: uses `sxmBrowser.figure.add_trace(go.Scatter(...))`
- `markerSelection` options updated to plotly symbol names
- `make_figure`: simplified (no plt.close)
- `spec_label` uses `<br>` separators
- `spectrumBrowser = fileBrowser` alias at module bottom
