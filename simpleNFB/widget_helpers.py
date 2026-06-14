"""
widget_helpers.py
-----------------
Shared ipywidgets factory functions and layout helpers used by all browser classes.

Previously these were copy-pasted identically into DAT_browser.py and SXM_browser.py.
Import from here instead:

    from widget_helpers import HBox, VBox, Btn_Widget, Text_Widget, Selection_Widget, make_layouts
"""

import ipywidgets as widgets


# ---------------------------------------------------------------------------
# Layout containers
# ---------------------------------------------------------------------------

def HBox(*pargs, **kwargs):
    """Horizontal flex box."""
    box = widgets.Box(*pargs, **kwargs)
    box.layout.display = 'flex'
    box.layout.align_items = 'stretch'
    return box


def VBox(*pargs, **kwargs):
    """Vertical flex box."""
    box = widgets.Box(*pargs, **kwargs)
    box.layout.display = 'flex'
    box.layout.flex_flow = 'column'
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


def Image_Widget(filename: str, format='png', width=720, height=720, **kwargs):
    """Image widget loaded from file."""
    return widgets.Image(
        value=open(filename, 'rb').read(),
        format=format,
        width=width,
        height=height,
        **kwargs,
    )


def Selection_Widget(selection_list: list, label: str, rows=30):
    """Single-select list widget."""
    return widgets.Select(options=selection_list, description=label, disabled=False, rows=rows)


# ---------------------------------------------------------------------------
# Layout lambda factory
# ---------------------------------------------------------------------------

def make_layouts():
    """
    Return a dict of standard layout lambda factories used by the browsers.

    Usage::

        L = make_layouts()
        btn = widgets.Button(layout=L['layout'](30))
        panel = widgets.VBox(layout=L['flex'](50))
    """
    return {
        # fixed-pixel layouts (visibility=visible / hidden)
        'layout':   lambda x: widgets.Layout(visibility='visible', width=f'{x}px'),
        'layout_h': lambda x: widgets.Layout(visibility='hidden',  width=f'{x}px'),
        # flex-percent layouts
        'flex':     lambda x: widgets.Layout(display='flex', width=f'{x}%'),
        'flex_btn': lambda x: widgets.Layout(
                        display='flex', width=f'{x}%',
                        align_items='center', justify_content='center'),
        'flex_h':   lambda x: widgets.Layout(
                        visibility='hidden', display='flex', width=f'{x}%'),
    }
