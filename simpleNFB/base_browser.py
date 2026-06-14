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
    * ``self.figure``              – matplotlib Figure (set by subclass ``__init__``)
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

        Two clipboard strategies are attempted in order:

        1. **win32clipboard** (primary) — renders the figure into a
           screen-quality in-memory PNG buffer (150 dpi), converts it to a
           DIB bitmap, and writes it directly to the clipboard with no
           subprocess overhead.  Requires ``pywin32`` and ``Pillow``.

        2. **PowerShell fallback** — if ``pywin32`` is not installed, calls
           ``Set-Clipboard -Path`` on the file written by ``save_figure``.
           A 10-second timeout prevents the UI from hanging indefinitely.
           A non-zero exit code is reported via ``updateErrorText``.

        The hourglass icon is shown immediately (before the disk write) so
        the user receives feedback during the full operation.  The clipboard
        icon is always restored in the ``finally`` block, even on failure.
        """
        import io

        self.copyBtn.icon = 'hourglass-half'
        try:
            # Disk save first — maintains existing behaviour and populates
            # last_save_fname for the PowerShell fallback path.
            self.save_figure(_event)

            # Render a screen-quality copy into memory.  150 dpi is sufficient
            # for clipboard paste and avoids re-reading the 500 dpi saved file.
            buf = io.BytesIO()
            self.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)

            # ------------------------------------------------------------------
            # Primary path: win32clipboard — instant, no subprocess
            # ------------------------------------------------------------------
            try:
                import win32clipboard
                from PIL import Image

                img = Image.open(buf).convert('RGB')
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
