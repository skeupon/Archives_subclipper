# -*- mode: python -*-

block_cipher = None

a = Analysis(
    ['archive_subclipper.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('archives_subclipper_logo.png', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinterdnd2',
        'cv2',
        'PIL',
        'scenedetect',
        'scenedetect.detectors',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Archives Subclipper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False
)

app = BUNDLE(
    exe,
    name='Archives Subclipper.app',
    icon=None,
    bundle_identifier='com.erwan.archivessubclipper'
)
