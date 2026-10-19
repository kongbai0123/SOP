"""Measure real record notifications on a temporary DB; no model or camera runs.

Run this script with the same interpreter for both --project-root checkouts.
The counts include the continuous notification period and the final timer flush.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--notifications", type=int, default=40)
    parser.add_argument("--notify-ms", type=int, default=25)
    parser.add_argument("--refresh-ms", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.notifications, args.notify_ms, args.refresh_ms) < 1:
        parser.error("rows, notifications and intervals must be positive")
    sys.path.insert(0, str(args.project_root.resolve()))
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication
    from sop_app.recorder import connect
    from sop_app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    with tempfile.TemporaryDirectory(prefix="sop-records-bench-") as temporary, ExitStack() as stack:
        root = Path(temporary)
        db_path = root / "records.db"
        db = connect(db_path)
        stack.callback(db.close)
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.executemany("INSERT INTO cycles(id,sop_name,sop_version,started_at,result) VALUES(?,?,?,?,?)",
                       [(i, "synthetic SOP", 1, started, "OK") for i in range(1, args.rows + 1)])
        snapshots = root / "snapshots"
        snapshots.mkdir()
        for i, color in enumerate(("green", "blue"), 1):
            image = QImage(160, 90, QImage.Format.Format_RGB32)
            image.fill(QColor(color))
            image.save(str(snapshots / f"evidence-{i}.png"))
        db.commit()
        for target, value in (("RECORDS_DIR", root), ("PROJECT_ROOT", root),
                              ("SOPS_DIR", root / "sops"), ("APP_SETTINGS", root / "settings.json")):
            stack.enter_context(patch(f"sop_app.ui.main_window.{target}", value))
        for target, value in (("_load_settings", {}), ("_save_settings", None), ("_bind_station_logging", None)):
            stack.enter_context(patch.object(MainWindow, target, return_value=value))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.start"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.shutdown", return_value=True))
        window = MainWindow(defer_initial_sop=True)
        stack.callback(window.close)
        page = window.records_page
        window.resize(1500, 900)
        window.tabs.setCurrentWidget(page)
        window.show()
        app.processEvents()
        page.refresh()
        # Select an older round and its second evidence item. A rebuild used to
        # reset this detail selection to the first item after every notification.
        selected_cycle = int(page.cycles.item(page.cycles.rowCount() - 1, 0).text())
        page.cycles.selectRow(page.cycles.rowCount() - 1)
        db.executemany("INSERT INTO step_records(cycle_id,step_index,step_name,status,finished_at,snapshot) "
                       "VALUES(?,?,?,'done',?,?)",
                       [(selected_cycle, i, f"step {i}", started, f"evidence-{i}.png") for i in (1, 2)])
        db.commit()
        page.refresh()
        page.steps.selectRow(1)
        before_caption = page.preview_caption.text()
        before_image = page.preview.original.toImage()
        counts = {"records": 0, "today": 0}
        cpu_ms = {"records": 0., "today": 0.}
        wall_ms = {"records": 0., "today": 0.}
        latest_during_notifications = []

        def measured(kind, operation):
            def call(*positional, **keywords):
                cpu, wall = time.process_time_ns(), time.perf_counter_ns()
                try:
                    return operation(*positional, **keywords)
                finally:
                    counts[kind] += 1
                    cpu_ms[kind] += (time.process_time_ns() - cpu) / 1e6
                    wall_ms[kind] += (time.perf_counter_ns() - wall) / 1e6
                    if kind == "records":
                        latest_during_notifications.append(emitted[0])
            return call

        page._refresh_records = measured("records", page._refresh_records)
        window._refresh_today_stats = measured("today", window._refresh_today_stats)
        for owner, name in ((page, "_refresh_timer"), (window, "_record_stats_timer")):
            timer = getattr(owner, name, None)
            if timer is not None:
                timer.setInterval(args.refresh_ms)
        emitted = [0]
        notify = QTimer()
        notify.setInterval(args.notify_ms)

        def tick():
            emitted[0] += 1
            db.execute("INSERT INTO cycles(sop_name,sop_version,started_at,result) VALUES('synthetic SOP',1,?,'OK')",
                       (started,))
            db.commit()
            window.pipeline.records_changed.emit()
            if emitted[0] == args.notifications:
                notify.stop()
                QTimer.singleShot(args.refresh_ms + 150, app.quit)

        notify.timeout.connect(tick)
        began = time.perf_counter()
        notify.start()
        app.exec()
        image_kept = page.preview.original.toImage() == before_image
        last_visible = int(page.cycles.item(0, 0).text())
        result = {
            "scope": "temporary SQLite, synthetic writes, offscreen Qt; no model/camera/production data",
            "project_root": str(args.project_root.resolve()), "initial_rows": args.rows,
            "notifications": emitted[0], "notify_interval_ms": args.notify_ms,
            "coalescing_interval_ms": args.refresh_ms, "elapsed_seconds": round(time.perf_counter() - began, 4),
            "refresh_counts": counts,
            "refresh_gui_process_cpu_ms": {k: round(v, 3) for k, v in cpu_ms.items()},
            "refresh_gui_wall_ms": {k: round(v, 3) for k, v in wall_ms.items()},
            "refresh_after_notification_numbers": latest_during_notifications,
            "refreshed_before_notifications_ended": any(0 < n < args.notifications for n in latest_during_notifications),
            "selected_cycle_preserved": page._selected_cycle_id() == selected_cycle,
            "selected_evidence_preserved": page.steps.currentRow() == 1 and page.preview_caption.text() == before_caption,
            "preview_image_preserved": image_kept,
            "latest_row_visible": last_visible == args.rows + args.notifications,
            "today_final_count": sum(window.run_page._today.values()),
        }
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)


if __name__ == "__main__":
    main()
