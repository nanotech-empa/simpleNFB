"""
pipeline_panel.py -- reorderable process pipeline widget (DYNAMIC_PIPELINE_PLAN.md §2.4).

PipelinePanel wraps a VBox of PipelineRows: a dropdown + Add button appends a
row for any discovered ProcessSpec whose KIND is in the browser's allowed set;
each row owns independent param widgets and renders as two stacked lines --
header ([enable] Name ... [up][down][x]) over a horizontally scrolling strip
of `name: [input]` pairs.
Browsers wire `on_change` to their existing debounced redraw (`_schedule_redraw_dirty`)
so every edit/reorder triggers exactly one redraw, matching the rest of the app.

JSON contract (templates + code export both consume `to_list()`):
    [{'process': stem, 'enabled': bool, 'params': {name: value, ...}}, ...]
"""

import ipywidgets as widgets

from .widget_helpers import HBox, VBox

# Geometry. Captions are natural-width (the scroll strip absorbs any overflow,
# so nothing is truncated); only the entry width is fixed. Tune here.
_INPUT_W = '90px'   # one parameter entry widget
_BTN_W   = '32px'    # up/down/remove buttons


def _nowrap_label(text: str, tooltip: str = '', bold: bool = False) -> widgets.HTML:
    """Natural-width single-line label; full text/tooltip on hover via title=."""
    title = (tooltip or text).replace('"', '&quot;')
    weight = 'font-weight:600;' if bold else ''
    return widgets.HTML(
        value=f'<div title="{title}" style="white-space:nowrap;{weight}">{text}</div>',
        layout=widgets.Layout(flex='0 0 auto'))


class PipelineRow:
    """One pipeline step: its own enable checkbox + param widget instances.

    Two stacked rows:
        [✓] Process Name                              [↑][↓][✕]
        ← name: [input] │ name: [input] │ ... (horizontal scroll) →
    The header never scrolls; only the parameter strip does."""

    def __init__(self, spec, on_change_cb):
        self.spec = spec
        self._on_change_cb = on_change_cb

        # -- header row: enable + name (grows) + reorder/remove buttons --------
        self.enabled_widget = widgets.Checkbox(value=True, indent=False,
                                                layout=widgets.Layout(width='24px', flex='0 0 auto'))
        self.label_widget = _nowrap_label(spec.name, spec.description, bold=True)
        spacer = widgets.HTML('', layout=widgets.Layout(flex='1 1 auto'))
        btn_layout = widgets.Layout(width=_BTN_W, flex='0 0 auto')
        self.up_btn = widgets.Button(description='↑', layout=btn_layout)
        self.down_btn = widgets.Button(description='↓', layout=btn_layout)
        self.remove_btn = widgets.Button(description='✕', layout=btn_layout)
        header = HBox([self.enabled_widget, self.label_widget, spacer,
                        self.up_btn, self.down_btn, self.remove_btn],
                       layout=widgets.Layout(align_items='center', width='100%'))

        # -- param row: <name> <input> pairs in one horizontal scroller --------
        # The widget's own description is dropped (it stole space from the
        # entry); the caption sits to the left of each input instead.
        self.param_widgets = spec.make_widgets()  # dict, PARAMS order preserved
        pairs = []
        for i, p in enumerate(spec.params):
            w = self.param_widgets[p['name']]
            w.description = ''
            w.style.description_width = '0px'
            w.layout = widgets.Layout(width=_INPUT_W, flex='0 0 auto',
                                      margin='0 0 0 4px')      # gap after caption
            caption = _nowrap_label(f"{p.get('label', p['name'])}:",
                                    p.get('tooltip', p['name']))
            pair = HBox([caption, w], layout=widgets.Layout(
                flex='0 0 auto', align_items='center', padding='0 8px',
                border_left='1px solid #b0b0b0' if i else ''))  # │ separator
            pairs.append(pair)
        self.params_box = HBox(pairs, layout=widgets.Layout(
            width='100%', min_width='0',                       # shrinkable → scrolls
            flex_flow='row nowrap', overflow='auto hidden'))   # x-scroll (ipw8 shorthand)

        for w in (self.enabled_widget, *self.param_widgets.values()):
            w.observe(self._fire, names='value')
        # no params → skip the empty second row entirely
        children = [header, self.params_box] if pairs else [header]
        self.box = VBox(children, layout=widgets.Layout(
            width='100%', margin='0 0 4px 0',
            border_bottom='1px solid #d5d5d5'))                # visual row divider

    def _fire(self, _change):
        self._on_change_cb()

    def set_params(self, params):
        for name, val in (params or {}).items():
            if name in self.param_widgets:
                self.param_widgets[name].value = val

    def to_dict(self):
        return {'process': self.spec.stem,
                'enabled': self.enabled_widget.value,
                'params': {name: w.value for name, w in self.param_widgets.items()}}


class PipelinePanel:
    """specs: {stem: ProcessSpec} from processes.discover(). kinds: allowed KIND
    set for this browser ({'1d','xy','batch'} for DAT, {'2d'} for SXM).
    on_change: no-arg callable fired on every add/remove/reorder/param edit
    (suppressed while rebuilding from a template via from_list)."""

    def __init__(self, specs, kinds, on_change):
        self.specs = specs
        self.kinds = set(kinds)
        self._on_change = on_change
        self._loading = False
        self.rows = []
        self.warnings = []

        options = sorted(((s.name, stem) for stem, s in specs.items() if s.kind in self.kinds),
                          key=lambda o: o[0])
        self.dropdown = widgets.Dropdown(options=options, description='',style={'description_width': '0px'},)
        self.add_btn = widgets.Button(description='Add', icon='plus',layout=widgets.Layout(width='60px', flex='0 0 auto'))
        self.add_btn.on_click(self._on_add_click)
        self._sync_dropdown_tooltip()
        self.dropdown.observe(lambda _ch: self._sync_dropdown_tooltip(), names='value')

        self.rows_box = VBox([])
        self.container = VBox([HBox([self.dropdown, self.add_btn]), self.rows_box])

    def _sync_dropdown_tooltip(self):
        stem = self.dropdown.value
        if stem is not None and stem in self.specs:
            self.dropdown.tooltip = self.specs[stem].description

    def _on_add_click(self, _btn):
        if self.dropdown.value is None:
            return
        self.add_row(self.dropdown.value)
        self._fire_change()

    def add_row(self, stem, enabled=True, params=None):
        """Append one row for `stem` (duplicates allowed -- each gets fresh widgets)."""
        spec = self.specs[stem]
        row = PipelineRow(spec, self._fire_change)
        row.enabled_widget.value = enabled
        row.set_params(params)
        row.up_btn.on_click(lambda _b, r=row: self._move(r, -1))
        row.down_btn.on_click(lambda _b, r=row: self._move(r, +1))
        row.remove_btn.on_click(lambda _b, r=row: self._remove(r))
        self.rows.append(row)
        self._rebuild_box()
        return row

    def _move(self, row, direction):
        i = self.rows.index(row)
        j = i + direction
        if 0 <= j < len(self.rows):
            self.rows[i], self.rows[j] = self.rows[j], self.rows[i]
            self._rebuild_box()
            self._fire_change()

    def _remove(self, row):
        self.rows.remove(row)
        self._rebuild_box()
        self._fire_change()

    def _rebuild_box(self):
        self.rows_box.children = tuple(r.box for r in self.rows)

    def _fire_change(self):
        if not self._loading and self._on_change is not None:
            self._on_change()

    def to_list(self):
        """JSON-safe pipeline config; consumed by templates and code export."""
        return [r.to_dict() for r in self.rows]

    def from_list(self, cfg):
        """Rebuild rows from a saved config (on_change suppressed, mirrors `_loading`).
        Unknown stems (process file deleted since save) are skipped; a message is
        collected in `self.warnings` for the caller to push to `updateErrorText`."""
        self._loading = True
        try:
            self.rows = []
            self.warnings = []
            for entry in (cfg or []):
                stem = entry.get('process')
                if stem not in self.specs:
                    self.warnings.append(f"pipeline: unknown process '{stem}' skipped")
                    continue
                self.add_row(stem, enabled=entry.get('enabled', True), params=entry.get('params'))
        finally:
            self._loading = False
