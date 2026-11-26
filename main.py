"""目标检测 + 语义分割标注工具入口。"""

import os
import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

from main_window import MainWindow


def _fix_qt_plugin_path() -> None:
    # opencv-python（非 headless）导入时会把 QT_QPA_PLATFORM_PLUGIN_PATH
    # 指向自带的 Qt 插件，与 PyQt5 的 Qt 版本冲突，导致 QApplication 启动失败。
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "").replace("\\", "/")
    if "cv2/qt/plugins" not in plugin_path:
        return
    pyqt_platforms = os.path.join(
        os.path.dirname(QtCore.__file__), "Qt5", "plugins", "platforms"
    )
    if os.path.isdir(pyqt_platforms):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_platforms
    else:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def main() -> None:
    _fix_qt_plugin_path()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
