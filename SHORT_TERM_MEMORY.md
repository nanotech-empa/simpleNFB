# simpleNFB — Short Term Memory

**Last updated:** 2026-06-25  
**Status:** matplotlib → plotly migration COMPLETE. Observer + file-selection optimization planned.

---

## Plotly migration plan

Replace `matplotlib` with `plotly` across all browser files.  
Core pattern: `go.FigureWidget` embeds directly in ipywidgets layouts — no `widgets.Output` wrapper needed.

### Concept map

| matplotlib | plotly |
|---|---|
| `plt.subplots()` + `widgets.Output` | `go.FigureWidget()` (is itself a widget) |
| `ax.imshow(data, ...)` | `go.Heatmap(z=data, ...)` updated via `batch_update()` |
| `ax.plot(x, y, label=l)` | `fig.add_trace(go.Scatter(x, y, name=l, mode='lines'))` |
| `ax.clear()` | `fig.data = ()` |
| `ax.twiny()` | `fig.update_layout(xaxis2=dict(overlaying='x', side='top'))` |
| `ax.legend(draggable=True)` | `fig.update_layout(legend=dict(...))` — draggable by default |
| `AutoMinorLocator` | `fig.update_xaxes(minor=dict(ticks='inside', ticklen=3))` |
| colorbar via `make_axes_locatable` | `go.Heatmap(colorbar=dict(...))` built-in |
| `AnchoredSizeBar` | `fig.add_shape(type='line')` + `fig.add_annotation()` |
| `figure.tight_layout()` | not needed |
| `figure.canvas.draw()` | not needed — FigureWidget updates reactively |
| `figure.savefig(fname, dpi=500)` | `fig.write_image(fname, scale=5)` (kaleido) |
| `figure.savefig(buf, dpi=150)` | `buf = io.BytesIO(fig.to_image(format='png', scale=2))` |
| `plt.colormaps()` dropdown | `px.colors.named_colorscales()` |
| `plt.cm.get_cmap(name)(linspace)` | `px.colors.sample_colorscale(name, linspace)` |
| `ax.axis()` → xmin,xmax,ymin,ymax | `fig.layout.xaxis.range`, `fig.layout.yaxis.range` |

### Efficient update pattern
```python
with self.figure.batch_update():
    self.figure.data[0].z    = new_data   # Heatmap
    self.figure.data[0].zmin = vmin
    self.figure.data[0].zmax = vmax
```
Batched updates push a single diff to the frontend — no flicker.

---

## Implementation order (one read + one edit pass per file)

### 1. `pyproject.toml`
- Add `plotly>=5.18` to `dependencies`
- Add `[export]` optional extra: `kaleido>=0.2` (needed for `write_image` / `to_image`)
- Remove `matplotlib` from `dependencies` (still imported in `process_utils` for colormaps — evaluate)
- Keep `Pillow` in `[clipboard]` extra (still needed for DIB clipboard path)

### 2. `base_browser.py`
- `copy_figure`: replace `self.figure.savefig(buf, ...)` with `io.BytesIO(self.figure.to_image(format='png', scale=2))`
- `save_figure` callers: update docstring reference from "matplotlib Figure" to "plotly FigureWidget"
- No structural changes needed

### 3. `SXM_browser.py` (image browser)
**Imports**: drop `matplotlib.pyplot`, `mpl_toolkits`, `matplotlib.font_manager`; add `plotly.graph_objects as go`, `plotly.express as px`

**`__init__`**:
- `self.figure = go.FigureWidget(data=[go.Heatmap(z=self.image_data, colorscale='greys', reversescale=True)])`
- Remove `self.figure_display = widgets.Output(...)` — use `self.figure` directly in layout
- Remove `with plt.ioff(): with self.figure_display: plt.show(self.figure)`
- Add `self.figure.update_layout(margin=dict(l=60,r=20,t=60,b=60))`

**`_build_layout`**: replace `self.figure_display` with `self.figure` in `v_image_layout`

**`_redraw`**:
```python
def _redraw(self, *_):
    self.update_axes()
    # no canvas.draw() needed
```

**`update_axes`**:
```python
with self.figure.batch_update():
    self.figure.data[0].z          = data
    self.figure.data[0].x          = np.linspace(0, w, data.shape[1])
    self.figure.data[0].y          = np.linspace(0, h, data.shape[0])
    self.figure.data[0].zmin       = self.vmin.value
    self.figure.data[0].zmax       = self.vmax.value
    self.figure.data[0].colorscale = self.cmapSelection.value
    self.figure.data[0].colorbar   = dict(title=f'{channel} ({unit})', thickness=12, len=0.9)
    self.figure.update_layout(
        title=dict(text=self.scan_info, font=dict(size=self.titleFontSize.value), x=0),
        xaxis=dict(title='x (nm)', tickvals=[0, w], ticktext=['0', f'{w:.2f}']),
        yaxis=dict(title='y (nm)', tickvals=[0, h], ticktext=['0', f'{h:.2f}']),
    )
```

**`_add_figure_labels`**: replace `ax.text(...)` / `AnchoredSizeBar` with:
```python
fig.add_annotation(x=x_frac, y=y_frac, xref='paper', yref='paper', text=label, ...)
fig.add_shape(type='line', x0=..., x1=..., xref='paper', yref='paper', ...)  # scalebar
```

**`save_figure`**: `self.figure.write_image(fname, scale=5)`

**`cmapSelection` options**: `px.colors.named_colorscales()` — note user cmap prefs reset

### 4. `DAT_browser.py` (spectrum browser)
**Imports**: drop `matplotlib`, `mpl_toolkits`; add `plotly.graph_objects as go`, `plotly.express as px`; keep `scipy` (filters still in `process_utils`)

**`__init__`**:
- `self.figure = go.FigureWidget()`
- Remove `self.figure_display` widget; remove `with plt.ioff():` block
- Remove `mpl.rcParams` block (no longer relevant)

**`_build_layout`**: replace `self.figure_display` with `self.figure` in `v_image_layout`

**`_redraw`**:
```python
def _redraw(self, *_):
    self.update_axes()
```

**`update_axes`**:
```python
# 1. clear traces
self.figure.data = ()
# 2. build
x_values, y_values = self._apply_filters(...)
labels = self._build_legend_labels(...)
self._plot_spectra(x_values, y_values, labels)
# 3. layout
self.figure.update_layout(
    title=dict(text=self.spec_label, font=dict(size=self.titlesize), x=0),
    xaxis_title=..., yaxis_title=...,
    showlegend=self.legendToggle.value,
    legend=dict(font=dict(size=self.legendFontsize[bisect.bisect_left([4,16], len(self.figure.data))]))
)
```

**`_plot_spectra`**:
```python
colors = px.colors.sample_colorscale(self.cmapSelection.value, np.linspace(0,1,len(y_values)))
for x, y, lbl, c in zip(x_values, y_values, labels, colors):
    self.figure.add_trace(go.Scatter(x=x, y=y, name=lbl, mode='lines', line=dict(color=c)))
```

**`_add_stml_axis`**:
```python
self.figure.update_layout(
    xaxis2=dict(title='Wavelength (nm)', overlaying='x', side='top',
                tickvals=energy_ticks, ticktext=[f'{1240/e:.0f}' for e in energy_ticks])
)
# add invisible traces on xaxis2 to anchor the scale
```

**`_sync_axis_limit_sliders`**:
```python
xr = self.figure.layout.xaxis.range or [-1, 1]
yr = self.figure.layout.yaxis.range or [-1, 1]
```

**`plot2D`**: replace `ax.imshow(...)` with `go.Heatmap(...)` added to figure

**`handler_update_axes_limits`**:
```python
self.figure.update_layout(xaxis=dict(range=[xmin, xmax]))
```

**`save_figure`**: `self.figure.write_image(fname, scale=5)`

**`cmapSelection` options**: `px.colors.named_colorscales()`

---

## Open decisions resolved
1. **matplotlib removed**: process_utils has no mpl import; scipy handles filtering. Using `px.colors.named_colorscales()` for dropdowns.
2. **kaleido**: Add `[export]` extra. Both `save_figure` and `copy_figure` wrap `write_image`/`to_image` in try/except reporting error via `updateErrorText`.
3. **cmap names**: Use plotly lowercase names. Accept `cmap='Greys_r'` in __init__, convert to `('greys', reversed=True)`. Add `reverseScaleToggle` button to SXM browser.
4. **DAT-specific gaps found in verification**:
   - `_render_2d()` → `go.Heatmap(z=z, x=x_ref, y=y_vals)` replaces pcolormesh; colorbar built-in
   - `_on_figure_draw()` → removed; plotly FigureWidget handles resize natively
   - `self.axes2` / `self.cb` → removed; secondary axis via `xaxis2` layout key
   - `figure.canvas` in VBox → replaced with `self.figure` (FigureWidget is itself a widget)
   - `mpl.rcParams` block → removed
   - `make_figure()` → simplified (no plt.close needed)
   - `generateWaterFall` → redirects to 2D view (plot2DToggle + _render_2d)
   - `plotSpectrumLocations` → uses `sxmBrowser.figure.add_trace(go.Scatter(...))`
   - `_add_stml_axis` signature change: remove `ax` param; use `fig.update_layout(xaxis2=...)`
   - `_plot_spectra` signature change: remove `ax` param; use `fig.add_trace(go.Scatter(...))`
   - `_sync_axis_limit_sliders` → reads `fig.layout.xaxis.range`; falls back to data range if None
   - `handler_update_axes_limits` → `fig.update_layout(xaxis=dict(range=[...]))`
   - `markerSelection` options → plotly symbol names: 'N', 'circle', 'star', 'square', 'triangle-up', 'x'
5. **SXM-specific**: `imageBrowser = fileBrowser` alias at end; same for DAT `spectrumBrowser = fileBrowser`
6. **__init__.py**: exports `imageBrowser`/`spectrumBrowser` — aliases fix this without changing public API

---

## Observer optimization plan

**Goal**: maximize speed, minimize redundant callbacks. Three cost tiers.

### Helper to add to both browsers
```python
def _refresh_info(self, *_):
    """Tier 2: rebuild title text + redraw. No data processing."""
    self.update_scan_info()
    self._redraw()
```

### SXM — `_connect_observers` changes

Remove catch-all tab loop (lines 276-282). Replace with explicit groups:

```python
# Tier 1 — layout update only
for w in (self.titleToggle, self.labelToggle,
          self.upperLeftSelect, self.upperRightSelect,
          self.lowerLeftSelect, self.lowerRightSelect,
          self.labelColorSelect, self.labelFontSize, self.titleFontSize):
    w.observe(self._redraw, names='value')

# Tier 2 — update_scan_info + _redraw
for w in (self.channelToggle, self.setpointToggle, self.feedbackToggle,
          self.locationToggle, self.depthSelection,
          self.nameToggle, self.directionToggle, self.dateToggle):
    w.observe(self._refresh_info, names='value')

# Tier 3 — full pipeline (already explicit, keep as-is)
for w in (self.gaussianToggle, self.gaussianSize,
          self.medianToggle,   self.medianSize,
          self.laplacToggle,   self.laplaceSize):
    w.observe(self.redraw_image, names='value')
```

Remove dead observers (lines 293-294): `edgesBtn.observe`, `gaussianBtn.observe`.

### DAT — `_connect_observers` changes

Remove catch-all accordion loop (lines 419-423). Replace with:

```python
# Tier 1
for w in (self.legendText, self.parameterLegendList, self.legendToggle,
          self.plot2DYLabel, self.plot2DXLabel, self.plot2DClimMode,
          self.cmapSelection):
    w.observe(self._redraw, names='value')

# Tier 2
for w in (self.titleToggle, self.nameToggle, self.setpointToggle,
          self.feedbackToggle, self.locationToggle, self.depthSelection,
          self.dateToggle):
    w.observe(self._refresh_info, names='value')

# Tier 3 (consolidate existing explicit loops here)
for w in (self.offsetToggle, self.offsetSize,
          self.svgToggle, self.svgSize, self.svgOrder,
          self.medFiltBtn, self.medFiltSize,
          self.despikeBtn, self.despikeWindow, self.despikeThreshold,
          self.movAvgBtn, self.movAvgSize,
          self.thresholdToggle, self.thresholdValue, self.averageToggle,
          self.flattenBtn, self.legendBtn, self.fixZeroBtn,
          self.stmlToggle,
          self.normalizeTimeBtn, self.normalizeCurrentBtn,
          self.normalizeEnergyBtn, self.normalizePlasmonBtn,
          self.plot2DToggle, self.plot2DYParam):
    w.observe(self.redraw_image, names='value')
```

**Additional DAT fixes:**
- Delete line 477 (`stmlToggle.observe(handler_settingsChange)`) — double binding removed by replacing loop
- `plasmonReference`: remove from any generic tier; keep only `handler_update_plasmonic_reference`; have that handler call `self._redraw()` at its end
- `legendEntry`: set `continuous_update=False` in widget constructor; observe with `_redraw` only
- `cmapSelection`: moved from `handler_update_axes` group to tier 1 `_redraw` (removes unnecessary `update_scan_info()` call)
- Axes Controls panel widgets (`xLimitsMin`, `xLimitsMax`, `yLimitsMin`, `yLimitsMax`): no value observer — only respond to button `on_click`. Loop removal fixes this automatically.

---

## File selection optimization plan

See WORKING_MEMORY.md for full cascade analysis. Summary of fixes:

### Root cause
`load_new_image()` calls `_update_channel_selection()` which changes channel widget values,
which fire `handler_channel_selection` → extra `update_image_data() + _redraw()`.
Navigation (next/prev, folder change) sets `selectionList.value` programmatically → fires
`handler_file_selection` → full cascade again.

**SXM**: 2× `update_image_data()` + 2× `_redraw()` per file load  
**DAT**: 3× `update_image_data()` + 3-4× `_redraw()` per file load

### Fix: `_loading` flag (add to `BaseBrowser.__init__`)
```python
self._loading = False
```

Guard in `handler_channel_selection` (both browsers):
```python
def handler_channel_selection(self, update) -> None:
    if self._loading:
        return
    ...
```

Guard in `handler_file_selection` (both browsers) for programmatic nav:
```python
def handler_file_selection(self, update) -> None:
    if self._loading:
        return
    ...
```

Wrap `_update_channel_selection()` in `load_new_image()`:
```python
self._loading = True
self._update_channel_selection()
self._loading = False
self.update_image_data()   # now runs exactly once
```

Wrap `selectionList.value` assignments in `nextDisplay`/`previousDisplay` and
`handler_folder_selection`:
```python
self._loading = True
self._update_info_text()        # or selectionList.value = ...
self._loading = False
# then call load_new_image() + _redraw() explicitly
```

### DAT: `os.scandir` for mtime-sorted file listing
Replace `os.listdir` + per-file `os.path.getmtime` with:
```python
with os.scandir(directory) as it:
    entries = [e for e in it if e.name.endswith('.dat') and filter_fn(e.name)]
entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
self.dat_files = [e.name for e in entries]
```
`DirEntry.stat()` reuses cached filesystem metadata — no extra syscall per file.
