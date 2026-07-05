"""
widget_helpers.py
-----------------
Shared ipywidgets factory functions and layout helpers used by all browser classes.

Previously these were copy-pasted identically into DAT_browser.py and SXM_browser.py.
Import from here instead:

    from widget_helpers import HBox, VBox, Btn_Widget, Text_Widget, Selection_Widget
"""

import ipywidgets as widgets


# ---------------------------------------------------------------------------
# Layout containers
# ---------------------------------------------------------------------------

def HBox(*pargs, **kwargs):
    """Horizontal flex box. align_items defaults to 'stretch' but a value set
    on a caller-supplied Layout is respected (was silently overwritten before,
    defeating e.g. align_items='center' on SXM's image column)."""
    box = widgets.Box(*pargs, **kwargs)
    box.layout.display = 'flex'
    if box.layout.align_items is None:
        box.layout.align_items = 'stretch'
    return box


def VBox(*pargs, **kwargs):
    """Vertical flex box. Same align_items rule as HBox."""
    box = widgets.Box(*pargs, **kwargs)
    box.layout.display = 'flex'
    box.layout.flex_flow = 'column'
    if box.layout.align_items is None:
        box.layout.align_items = 'stretch'
    return box


# ---------------------------------------------------------------------------
# Widget factories
# ---------------------------------------------------------------------------

def Btn_Widget(displayText: str, **kwargs):
    """Simple button widget."""
    return widgets.Button(description=displayText, **kwargs)


def Text_Widget(text: str, **kwargs):
    """Simple text input widget."""
    return widgets.Text(value=text, **kwargs)


def Selection_Widget(selection_list: list, label: str, rows=30):
    """Single-select list widget."""
    return widgets.Select(options=selection_list, description=label, disabled=False, rows=rows)

# R1: make_layouts removed (zero callers — BaseBrowser._layout_helpers is the
# single source of the layout lambda factories).
