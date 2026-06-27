# simpleNFB — Working Memory

*Use for in-progress work requiring user input. Clear when work is complete.*

---

*(empty — observer + file selection optimizations implemented 2026-06-25; see LONG_TERM_MEMORY.md)*

<!--

## File selection cascade audit — 2026-06-25

### Call graphs (current, buggy state)

**SXM — user clicks a file in `selectionList`:**
```
selectionList.value changes
  → handler_file_selection
      → load_new_image()
          → Spm(file)                        [disk I/O]
          → _update_channel_selection()
              → channelSelect.value = X      [widget write]
                  → handler_channel_selection   ← SPURIOUS
                      → update_image_data()  [#1 — full pipeline]
                      → _redraw()            [#1]
          → update_image_data()              [#2 — duplicate]
      → _redraw()                            [#2 — duplicate]
```
**Cost: 2× update_image_data, 2× _redraw per file click**

---

**SXM — user clicks Next/Previous button:**
```
nextDisplay / previousDisplay
  → image_index += 1
  → _update_info_text()
      → filenameText.value = ...             [widget write, no observer]
      → selectionList.value = ...            [widget write]
          → handler_file_selection           ← SPURIOUS ROUTE
              → (full cascade above: 2× update_image_data, 2× _redraw)
```
**Same 2× cost, triggered via widget value rather than direct call**

---

**SXM — user changes directory in `directorySelection`:**
```
directorySelection.value changes
  → handler_folder_selection
      → os.listdir(directory)
      → selectionList.options = sxm_files   [clears value → handler_file_selection with None → returns]
      → filenameText.value = sxm_files[0]   [no observer]
      → selectionList.value = sxm_files[0]  [widget write]
          → handler_file_selection
              → (full cascade: 2× update_image_data, 2× _redraw)
```

---

**DAT — user selects file(s) in `selectionList`:**
```
selectionList.value changes
  → handler_file_selection
      → spec_index = [...]
      → load_new_image()
          → Spm(file) for each              [disk I/O]
          → _update_channel_selection()
              → channelXSelect.value = X    [widget write]
                  → handler_channel_selection   ← SPURIOUS #1
                      → update_image_data()  [#1]
                      → _redraw()            [#1]
              → channelYSelect.value = [Y]  [widget write]
                  → handler_channel_selection   ← SPURIOUS #2
                      → update_image_data()  [#2]
                      → _redraw()            [#2]
          → update_image_data()              [#3 — deliberate but still duplicate]
      → _update_legend_parameters(update)
      → update_legend_settings(update)
      → _redraw()                            [#3 — duplicate]
```
**Cost: 3× update_image_data, 3-4× _redraw per file selection**

---

**DAT — user clicks Next/Previous:**
```
nextDisplay / previousDisplay
  → spec_index = [idx ± 1]
  → _update_info_text()
      → filenameText.value = ...             [no observer]
      → selectionList.value = [dat_files[idx]]   [widget write]
          → handler_file_selection           ← SPURIOUS ROUTE
              → (full cascade: 3× update_image_data, 3-4× _redraw)
```

---

**DAT — mtime sort (handler_folder_selection):**
```python
# Current — N filesystem stat() calls
files = [(f, os.path.getmtime(os.path.join(directory, f))) for f in self.dat_files]
self.dat_files = [f[0] for f in sorted(files, key=lambda x: x[1], reverse=True)]
```
For a folder with 500 dat files → 500 separate `stat()` calls.  
`os.scandir()` returns `DirEntry` objects whose `.stat()` is cached from the directory
read — one syscall instead of N.

---

### Fix design

**1. Add `_loading = False` to `BaseBrowser.__init__`**

The flag suppresses reactive observers when widget values are set programmatically.

**2. Guard `handler_channel_selection` in both browsers**
```python
def handler_channel_selection(self, update) -> None:
    if self._loading:
        return
    # ... existing body
```

**3. Guard `handler_file_selection` in both browsers**
```python
def handler_file_selection(self, update) -> None:
    if self._loading:
        return
    # ... existing body
```
(Only needed for the programmatic navigation paths; direct user clicks work normally.)

**4. Wrap `_update_channel_selection()` in `load_new_image()` (both browsers)**
```python
# SXM
def load_new_image(self) -> None:
    directory = ...
    self.img = Spm(...)
    self._scan_cache = {}
    self.filenameText.value = self.all_files[self.image_index]
    self._loading = True
    self._update_channel_selection()   # channel observers suppressed
    self._loading = False
    self.update_image_data()           # runs exactly once
    # caller is responsible for _redraw()

# DAT
def load_new_image(self, filename=None):
    ...
    self._loading = True
    self._update_channel_selection()
    self._loading = False
    self.update_image_data()
```

**5. Rewrite `nextDisplay` / `previousDisplay` to avoid cascade (both browsers)**
```python
# SXM
def nextDisplay(self, a) -> None:
    if self.image_index < len(self.all_files) - 1:
        self.image_index += 1
        self._loading = True
        self._update_info_text()   # sets filenameText + selectionList.value (suppressed)
        self._loading = False
        try:
            self.load_new_image()
            self._redraw()
        except Exception as err:
            self.updateErrorText('navigation error: ' + str(err))

def previousDisplay(self, a) -> None:
    if self.image_index > 0:
        self.image_index -= 1
        self._loading = True
        self._update_info_text()
        self._loading = False
        try:
            self.load_new_image()
            self._redraw()
        except Exception as err:
            self.updateErrorText('navigation error: ' + str(err))

# DAT — same pattern, but spec_index used instead of image_index
def nextDisplay(self, a) -> None:
    idx = self.spec_index[-1]
    if idx < len(self.all_files) - 1:
        self.spec_index = [idx + 1]
        self._loading = True
        self._update_info_text()
        self._loading = False
        try:
            self.load_new_image()
            self._redraw()
        except Exception as err:
            self.updateErrorText('navigation error: ' + str(err))
```

**6. Rewrite `handler_folder_selection` to avoid cascade (both browsers)**
```python
# SXM
def handler_folder_selection(self, a) -> None:
    index = self.selectionList.index if type(a) == type(self.refreshBtn) else 0
    directory = self.directories[self.directorySelection.index]
    self.sxm_files, self.dat_files = [], []
    for entry in os.listdir(directory):
        if '.sxm' in entry:
            self.sxm_files.append(entry)
        elif '.dat' in entry:
            self.dat_files.append(entry)
    self.all_files = self.sxm_files + self.dat_files
    self.sxm_files = list(np.flip(self.sxm_files))
    self._loading = True
    self.selectionList.options = self.sxm_files   # won't cascade
    if self.sxm_files:
        self.filenameText.value = self.sxm_files[index]
        self.selectionList.value = self.sxm_files[index]
    self._loading = False
    if self.sxm_files:
        self.image_index = index
        try:
            self.load_new_image()
            self._redraw()
        except Exception as err:
            self.updateErrorText('folder selection error: ' + str(err))

# DAT — same pattern + os.scandir for mtime sort
def handler_folder_selection(self, a) -> None:
    index = self.selectionList.index if type(a) == type(self.refreshBtn) else 0
    directory = self.directories[self.directorySelection.index]
    filter_vals = self.filterSelection.value

    self.sxm_files = []
    with os.scandir(directory) as it:
        dat_entries = []
        for e in it:
            if e.name.endswith('.sxm'):
                self.sxm_files.append(e.name)
            elif e.name.endswith('.dat'):
                if 'all' in filter_vals or any(f in e.name for f in filter_vals):
                    dat_entries.append(e)
    dat_entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    self.dat_files = [e.name for e in dat_entries]
    self.all_files = self.sxm_files + self.dat_files

    self._loading = True
    self.selectionList.options = self.dat_files
    if self.dat_files:
        first = (self.dat_files[index] if isinstance(index, int)
                 else self.dat_files[index[0]])
        self.filenameText.value = first
        self.selectionList.value = [first]
        self.plasmonReference.options = ['None'] + [f for f in self.dat_files
                                                    if 'stml' in f.lower()]
        self.plasmonReference.value = 'None'
    self._loading = False

    if self.dat_files:
        self.spec_index = [0]
        try:
            self.load_new_image()
            self._redraw()
        except Exception as err:
            self.updateErrorText('folder selection error: ' + str(err))
```

---

### Expected result after fixes

| Action | Before | After |
|---|---|---|
| SXM file click | 2× update_image_data, 2× _redraw | 1× each |
| SXM next/prev | 2× update_image_data, 2× _redraw | 1× each |
| SXM folder change | 2× update_image_data, 2× _redraw | 1× each |
| DAT file click | 3× update_image_data, 3-4× _redraw | 1× each |
| DAT next/prev | 3× update_image_data, 3-4× _redraw | 1× each |
| DAT folder change (N files) | 3× update_image_data + N stat() calls | 1× each + 1 scandir |

---

### Implementation order

1. Add `_loading = False` to `BaseBrowser.__init__`
2. Edit `SXM_browser.py` (one read + one edit pass):
   - `load_new_image`: wrap `_update_channel_selection`
   - `handler_channel_selection`: add `_loading` guard
   - `handler_file_selection`: add `_loading` guard
   - `nextDisplay` / `previousDisplay`: set `_loading`, call load directly
   - `handler_folder_selection`: set `_loading`, call load directly
3. Edit `DAT_browser.py` (one read + one edit pass):
   - Same pattern as SXM
   - `handler_folder_selection`: replace `os.listdir` + getmtime with `os.scandir`

**STATUS: COMPLETE — implemented 2026-06-25**

-->
