"""Kurulum gerektirmeyen, salt-okunur SQLite kanit goruntuleyicisi."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "models" / "aircraft_fingerprints_3000.sqlite3"
PAGE_SIZE = 250


class DatabaseWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.database_path: Path | None = None
        self.table_name: str | None = None
        self.offset = 0
        self.setWindowTitle("Shazam SQLite Veritabani — Salt Okunur Kanit Ekrani")
        self.resize(1400, 850)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        top = QHBoxLayout()
        self.path_label = QLabel("Veritabani secilmedi")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        choose = QPushButton("SQLite Dosyasi Sec")
        choose.clicked.connect(self.choose_database)
        top.addWidget(choose)
        top.addWidget(self.path_label, 1)
        outer.addLayout(top)

        self.summary = QLabel("Tablo ve kayit sayilari burada gorunecek.")
        self.summary.setObjectName("summary")
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)

        body = QHBoxLayout()
        self.tables = QListWidget()
        self.tables.setMinimumWidth(250)
        self.tables.currentTextChanged.connect(self.show_table)
        body.addWidget(self.tables)
        self.grid = QTableWidget()
        self.grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.grid.setAlternatingRowColors(True)
        body.addWidget(self.grid, 1)
        outer.addLayout(body, 1)

        nav = QHBoxLayout()
        self.previous = QPushButton("Onceki 250")
        self.next = QPushButton("Sonraki 250")
        self.page = QLabel("")
        self.previous.clicked.connect(self.previous_page)
        self.next.clicked.connect(self.next_page)
        nav.addWidget(self.previous)
        nav.addWidget(self.next)
        nav.addWidget(self.page)
        nav.addStretch()
        outer.addLayout(nav)

        self.setStyleSheet("""
            QWidget { background: #0d131a; color: #e8eef5; font-size: 14px; }
            QPushButton { background: #173247; border: 1px solid #23c9e8;
                          padding: 9px 16px; border-radius: 5px; font-weight: 600; }
            QPushButton:hover { background: #21465f; }
            QListWidget, QTableWidget { background: #121b24; border: 1px solid #344453; }
            QListWidget::item { padding: 9px; }
            QListWidget::item:selected { background: #176a7c; }
            QHeaderView::section { background: #173247; padding: 7px; border: 0; }
            QLabel#summary { background: #10283a; border: 1px solid #23c9e8;
                             padding: 12px; font-size: 16px; font-weight: 600; }
        """)

        if DEFAULT_DATABASE.is_file():
            self.open_database(DEFAULT_DATABASE)

    def connect_readonly(self) -> sqlite3.Connection:
        if self.database_path is None:
            raise RuntimeError("Veritabani secilmedi")
        return sqlite3.connect(f"file:{self.database_path.as_posix()}?mode=ro", uri=True)

    def choose_database(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "SQLite veritabani sec", str(ROOT), "SQLite (*.sqlite3 *.sqlite *.db)"
        )
        if selected:
            self.open_database(Path(selected))

    def open_database(self, path: Path) -> None:
        try:
            with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
                tables = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                    if not row[0].startswith("sqlite_")
                ]
                counts = []
                for table in tables:
                    safe = table.replace('"', '""')
                    count = connection.execute(f'SELECT COUNT(*) FROM "{safe}"').fetchone()[0]
                    counts.append(f"{table}: {count:,}")
        except sqlite3.Error as error:
            QMessageBox.critical(self, "Dosya acilamadi", str(error))
            return

        self.database_path = path.resolve()
        self.path_label.setText(str(self.database_path))
        self.summary.setText(
            f"{len(tables)} tablo bulundu  |  " + "  |  ".join(counts)
        )
        self.tables.clear()
        self.tables.addItems(tables)
        if tables:
            self.tables.setCurrentRow(0)

    def show_table(self, table: str) -> None:
        if not table:
            return
        self.table_name = table
        self.offset = 0
        self.load_page()

    def load_page(self) -> None:
        if not self.table_name:
            return
        safe = self.table_name.replace('"', '""')
        try:
            with self.connect_readonly() as connection:
                cursor = connection.execute(
                    f'SELECT * FROM "{safe}" LIMIT ? OFFSET ?', (PAGE_SIZE, self.offset)
                )
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description or []]
                total = connection.execute(f'SELECT COUNT(*) FROM "{safe}"').fetchone()[0]
        except sqlite3.Error as error:
            QMessageBox.critical(self, "Tablo okunamadi", str(error))
            return

        self.grid.clear()
        self.grid.setColumnCount(len(columns))
        self.grid.setHorizontalHeaderLabels(columns)
        self.grid.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                text = "NULL" if value is None else str(value)
                if len(text) > 500:
                    text = text[:500] + "…"
                self.grid.setItem(row_index, column_index, QTableWidgetItem(text))
        self.grid.resizeColumnsToContents()
        start = 0 if total == 0 else self.offset + 1
        end = min(self.offset + len(rows), total)
        self.page.setText(f"{self.table_name}: {start:,}–{end:,} / {total:,}")
        self.previous.setEnabled(self.offset > 0)
        self.next.setEnabled(self.offset + PAGE_SIZE < total)

    def previous_page(self) -> None:
        self.offset = max(0, self.offset - PAGE_SIZE)
        self.load_page()

    def next_page(self) -> None:
        self.offset += PAGE_SIZE
        self.load_page()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DatabaseWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
