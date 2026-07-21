"""
simpleNFB/processes/ -- dynamic process discovery + ProcessSpec wrapper.

Each sibling module is a *pipeline process*: pure numpy/scipy, no widget or
browser state. Contract (strict, validated by discover()):
    NAME        str   -- dropdown label
    KIND        str   -- 'xy' | 'batch' -> DAT, '2d' -> SXM (see docstring below)
    DESCRIPTION str   -- tooltip text
    PARAMS      list[dict] -- one dict per keyword param of process(), each
                {'name','label','type': int|float|bool|choice, 'default',
                 optional: 'min','max','step','options','tooltip'}
    process     callable -- signature dictated by KIND (see _KIND_SIGNATURES)

Drop a new file in this folder -- no other code changes anywhere are needed
for it to show up in both browsers' pipeline dropdowns (that is the whole
point of the redesign, see DYNAMIC_PIPELINE_PLAN.md §2.3).

KIND dispatch signatures (also documented in the plan, §2.2):
    '1d'    process(y, **p) -> y             (DAT, looped per trace)
    'xy'    process(x, y, **p) -> (x, y)      (DAT, looped per trace)
    'batch' process(xs, ys, **p) -> (xs, ys)  (DAT, whole selection at once)
    '2d'    process(img, **p) -> img          (SXM)
"""

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field

import ipywidgets as widgets

_KNOWN_KINDS = {'1d', 'xy', 'batch', '2d'}

# For a given KIND: (input variable names, output variable names) -- used
# only for pretty-printing exported code in ProcessSpec.emit_code().
_KIND_VARS = {
    '1d':    (('y',),      ('y',)),
    'xy':    (('x', 'y'),  ('x', 'y')),
    'batch': (('xs', 'ys'), ('xs', 'ys')),
    '2d':    (('img',),    ('img',)),
}


@dataclass
class ProcessSpec:
    """Wraps one discovered process module for use by PipelinePanel + browsers."""
    stem: str
    name: str
    kind: str
    description: str
    params: list = field(default_factory=list)
    fn: callable = None
    module: object = None

    def make_widgets(self) -> dict:
        """Build one ipywidget per PARAMS entry, keyed by param name."""
        out = {}
        for p in self.params:
            kwargs = {'description': p.get('label', p['name'])}
            if 'tooltip' in p:
                kwargs['tooltip'] = p['tooltip']
            t = p['type']
            if t == 'int':
                kwargs.update(value=p['default'], min=p.get('min', 0),
                              max=p.get('max', 100), step=p.get('step', 1))
                w = widgets.BoundedIntText(**kwargs)
            elif t == 'float':
                kwargs.update(value=p['default'], step=p.get('step', 0.1))
                w = widgets.FloatText(**kwargs)
            elif t == 'bool':
                kwargs.update(value=p['default'])
                w = widgets.Checkbox(**kwargs)
            elif t == 'choice':
                kwargs.update(options=p['options'], value=p['default'])
                w = widgets.Dropdown(**kwargs)
            else:
                raise ValueError(f'{self.stem}: unknown PARAMS type {t!r}')
            out[p['name']] = w
        return out

    def call(self, *data, **params):
        """Run process() with only the recognised kwargs (extras ignored)."""
        names = {p['name'] for p in self.params}
        clean = {k: v for k, v in params.items() if k in names}
        return self.fn(*data, **clean)

    def emit_code(self, params: dict) -> str:
        """Return the source line reproducing this step (used by code export)."""
        ins, outs = _KIND_VARS[self.kind]
        names = {p['name'] for p in self.params}
        kw = ', '.join(f'{k}={params[k]!r}' for k in params if k in names)
        args = ', '.join(ins) + (f', {kw}' if kw else '')
        lhs = ', '.join(outs)
        return f'{lhs} = {self.stem}_process({args})'


def discover(package: str = __name__) -> tuple:
    """Scan sibling modules, validate the contract, return (specs, warnings).

    specs is {stem: ProcessSpec}; warnings is a list of human-readable strings
    for one bad file -- never an exception, so one malformed drop-in cannot
    break browser startup (see KNOWN_GOTCHAS G10: warnings must reach a widget
    the user actually sees, i.e. updateErrorText -- the caller's job).
    """
    specs, warnings = {}, []
    pkg = importlib.import_module(package)
    for info in pkgutil.iter_modules(pkg.__path__):
        stem = info.name
        if stem.startswith('_'):
            continue
        try:
            mod = importlib.import_module(f'.{stem}', package=package)
            for attr in ('NAME', 'KIND', 'PARAMS', 'process'):
                if not hasattr(mod, attr):
                    raise AttributeError(f'missing {attr}')
            if mod.KIND not in _KNOWN_KINDS:
                raise ValueError(f'unknown KIND {mod.KIND!r}')
            sig_params = set(inspect.signature(mod.process).parameters)
            for p in mod.PARAMS:
                if p['name'] not in sig_params:
                    raise ValueError(f"PARAMS {p['name']!r} not a process() kwarg")
            specs[stem] = ProcessSpec(
                stem=stem, name=mod.NAME, kind=mod.KIND,
                description=getattr(mod, 'DESCRIPTION', ''),
                params=mod.PARAMS, fn=mod.process, module=mod)
        except Exception as err:
            warnings.append(f'processes/{stem}.py skipped: {err}')
    return specs, warnings
