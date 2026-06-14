# simpleNFB — Short Term Memory

**Last updated:** 2026-06-14  
**Status:** SXM_browser.py and DAT_browser.py rewrite complete. All tasks done.

---

## SXM_browser.py — full change list (621 lines)

**Bugs**
- L270 `mouse_click`: `ix`/`iy` used before assignment — delete method body, leave stub
- L328,332 `update_image_data`: `names='values'` → `names='value'`

**`__init__` decomposition** (L46–244)
Extract from `__init__`, calling order: state-init → each → `display()`
- `_build_widgets()` — all widget instantiation (L67–150)
- `_build_layout()` — all HBox/VBox assembly (L154–198)
- `_connect_observers()` — all `.observe()` / `.on_click()` (L201–230)
- Remove unused: `smallLayout`, `mediumLayout`, `largeLayout`, `extraLargeLayout`, `layout_h`

**Renames** (update all call sites in same pass)
- `updateDisplayImage` → `_redraw`
- `updateChannelSelection` → `_update_channel_selection`
- `updateInfoText` → `_update_info_text`
- `addFigureLabels` → `_add_figure_labels`

**Optimisations**
- `save_figure` L251–254: `(self.active_dir/'browser_outputs').mkdir(exist_ok=True)`; `Path/` for fname
- `update_scan_info`: replace 10 `get_param` calls with `_cache_scan_param(key)` (cache cleared in `load_new_image`)
- `update_image_data`: `self._updating_limits` flag replaces 4-line unobserve/set/reobserve
- `_add_figure_labels`: `_ALIGN_PARAMS = {'upper left':('left','top'), ...}` dict replaces 4-branch if-chain
- `!= None` / `== None` → `is not None` / `is None` (L281, L605)
- `for i in range(len(...))` → `enumerate` in `_add_figure_labels` and `update_axes`

**Type hints + docstrings**: add to `__init__`, `update_axes`, `update_image_data`, `save_figure`, `load_new_image`, `display`

---

## DAT_browser.py — full change list (1046 lines)

**Bugs**
- L71: `referenceLocBtn.disabled(True)` → `referenceLocBtn.disabled = True`
- L391: `self.saveBtn.icon` → `self.csvBtn.icon`
- L521: `assert experiments.count(...) == len(experiments)` → `if ... : self.updateErrorText(...); return`
- L619: remove `print('update axes called')`
- L635: `y_values = self.spec_data` → `y_values = [a.copy() for a in self.spec_data]`
- L767: add `self.wfAxes = None` in `__init__`; guard `generateWaterFall` with `if self.wfAxes is None: return`
- L876–886 `nextDisplay`/`previousDisplay`: align `spec_index` treatment (scalar vs list)
- L906: `a['owner'] == self.parameterLegendToggle.value` → `a['owner'] == self.parameterLegendToggle`

**`__init__` decomposition** (L63–364)
- `_build_widgets()` — all widget instantiation (L93–201)
- `_build_layout()` — all HBox/VBox/Accordion assembly (L203–294)
- `_connect_observers()` — all `.observe()` / `.on_click()` (L301–357)
- Remove unused: `smallLayout`, `mediumLayout`, `largeLayout`, `extraLargeLayout`, `layout_h`
- Move `mpl.rcParams` block (L40–49) into `__init__` before `_build_widgets`

**Renames** (update all call sites in same pass)
- `updateDisplayImage` → `_redraw`
- `updateChannelSelection` → `_update_channel_selection`
- `updateInfoText` → `_update_info_text`
- `updateAxesLimitSliders` → `_sync_axis_limit_sliders`
- `updateLegendParameterSelection` → `_update_legend_parameters`

**Decompose `update_axes`** (~130 lines → sequencing shell)
```
_apply_filters(x_values, y_values) -> tuple[list,list]
_build_legend_labels(y_values) -> list[str]
_plot_spectra(ax, x_values, y_values, labels) -> None
_add_stml_axis(ax) -> None
```

**`update_scan_info` experiment dispatch**
```python
_INFO_BUILDERS = {
    'STML': '_info_stml', 'bias spectroscopy': '_info_bias_spec',
    'THz amplitude sweep': '_info_thz', 'Z spectroscopy': '_info_z_spec',
    'History Data': '_info_history',
}
```
Each `_info_*` appends strings to `label`; `update_scan_info` dispatches and assembles.

**Optimisations**
- `save_figure` L375–378, `save_data` L392–395: `Path.mkdir(exist_ok=True)`; `Path/` for all paths
- `save_data`: replace zip+nested-listcomp+`np.transpose` with `np.column_stack`
- `update_scan_info`: `_cache_scan_param(key)` replaces 12 `get_param` calls (cache cleared in `load_new_image`)
- Legend fontsize (L708–713): `bisect.bisect_left([4,16], len(ax.lines))` indexes `self.legendFontsize`
- Remove dead `else: pass` (L383, L583, L716, L838)
- Remove dead `return spec, spec_data, spec_x` from `update_image_data` STML branch
- `!= None` / `== None` → `is not None` / `is None` (11 sites)
- `for i in range(len(...))` → `enumerate` / `zip` throughout

**Type hints + docstrings**: add to `__init__`, `update_axes`, `update_image_data`, `update_scan_info`, `save_figure`, `save_data`, `load_new_image`, `display`

---

## Implementation order

1. `SXM_browser.py` — one read, one edit pass
2. `DAT_browser.py` — one read, one edit pass
3. Verify: grep for `!= None`, `range(len(`, `names='values'`, `print(` across both files

---

## Post-install

```bash
pip install -e ".[clipboard]"
```
```python
import simpleNFB.snfb as snfb; importlib.reload(snfb)
imageBrowser = snfb.sxmBrowser(home_directory='./data')
specBrowser  = snfb.datBrowser(home_directory='./data', sxmBrowser=imageBrowser)
```
