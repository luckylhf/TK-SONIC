# Teleoperation GUI license boundary

This directory is a separately identified **AGPL-3.0-or-later component** of
the SONIC distribution. It is not covered by the repository's general
Apache-2.0 source-code statement. The complete license text is at
`../../../../legal/AGPL-3.0-or-later.txt`.

The following files are derived from MakeHuman code, Copyright MakeHuman Team
2001-2020 and contributors:

- `library/getpath.py`, `image.py`, `language.py`, `log.py`, `matrix.py`,
  `mh.py`, `profiler.py`, `qtgui.py`, `universal.py`, `xdg_parser.py`
- `core/events3d.py`, `gui3d.py`, `guicommon.py`, `module3d.py`, `selection.py`

`library/qtgui.py` is derived from MakeHuman's `makehuman/lib/qtgui.py` and was
adapted to PyQt6 and SONIC before this v1.3 compliance pass. The historical
package does not identify the exact modification date. Licensing metadata was
clarified by luckylhf on 2026-08-18.

`main.py` and `cli.py` combine and launch the MakeHuman-derived library, so they
are distributed within the same AGPL component boundary. PyQt6 is an external
dependency licensed under GPL v3 or a commercial Riverbank license; it is not
vendored here. See the repository third-party notice for details.
