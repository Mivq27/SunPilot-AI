import os
import sys
from pathlib import Path
from multiprocessing import freeze_support

freeze_support()

# Locate files both in development and in PyInstaller.
if getattr(sys, "frozen", False):
    app_root = Path(sys._MEIPASS)
else:
    app_root = Path(__file__).resolve().parent

os.chdir(app_root)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import app

if __name__ == "__main__":
    print("=====================================")
    print("         SUNPILOT AI")
    print("=====================================")
    print("Starting local application...")

    app.demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True,
    )
