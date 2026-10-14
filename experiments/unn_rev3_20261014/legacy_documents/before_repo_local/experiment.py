"""Главный файл эксперимента: чтение ячеек через штатный MemriBoard Connector.

Меняйте настройки в блоке ниже. Сейчас выбран симулятор и запуск выключен.
Измерения: одна проверка каждой ячейки либо повторные чтения без записи весов.
Параметры АЦП/ЦАП берутся из protocol.json, фактические настройки сохраняются
рядом с результатами. Импорт этого файла не подключает плату и не читает ячейки.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import uuid

from experiments.unn_rev3_20261014.config import parse_protocol
from experiments.unn_rev3_20261014.connector_session import connect
from experiments.unn_rev3_20261014.readback import file_sha256, read_record, utc_now


# ================= НАСТРОЙКИ ЭКСПЕРИМЕНТА =================
RUN_APPROVED = False       # Менять только после явного разрешения на этот запуск.
BACKEND = "simulator"      # Реальное железо в этом сценарии пока не включено.
BOARD_DRIVER = "rp5_fpga_python"  # Имя драйвера; НЕ переключает backend на real.
MEMRIBOARD_ROOT = Path("D:/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard")
ARRAY_SERIAL = None        # Полный путь к доверенному существующему массиву БЕЗ .cb.
CELLS = ((0, 2), (2, 6), (2, 4))  # (WL, BL); WL=1 исключена.
STAGE = "first_check"      # first_check: по одному чтению; repeatability: серия.
READS_PER_CELL = 1         # Для repeatability можно задать до 20.

PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_FILE = PROJECT_DIR / "experiments/unn_rev3_20261014/protocol.json"
RESULTS_DIR = PROJECT_DIR / "results" / "readback"


def main():
    # 1. Проверить разрешение и собрать фактические настройки.
    if RUN_APPROVED is not True:
        raise PermissionError("Запуск выключен: сначала нужно явное разрешение пользователя.")
    if BACKEND != "simulator":
        raise ValueError("Этот эксперимент поддерживает только simulator.")
    if STAGE not in ("first_check", "repeatability"):
        raise ValueError("STAGE должен быть first_check или repeatability.")
    if type(READS_PER_CELL) is not int or not 1 <= READS_PER_CELL <= 20:
        raise ValueError("READS_PER_CELL должен быть целым числом от 1 до 20.")
    if STAGE == "first_check" and READS_PER_CELL != 1:
        raise ValueError("Для серии чтений выберите STAGE = 'repeatability'.")
    if ARRAY_SERIAL is None:
        raise ValueError("Укажите ARRAY_SERIAL: полный путь к существующему массиву без .cb.")
    serial = Path(ARRAY_SERIAL)
    if not serial.is_absolute() or serial.suffix.lower() == ".cb":
        raise ValueError("ARRAY_SERIAL должен быть абсолютным путём без окончания .cb.")
    array_file = Path(str(serial) + ".cb")
    if not array_file.is_file():
        raise FileNotFoundError(array_file)

    # Блок выше имеет приоритет над шаблоном; сохраняем именно итоговый протокол.
    template = PROTOCOL_FILE.read_bytes()
    settings = json.loads(template.decode("utf-8-sig"))
    settings["status"] = "approved_simulator_run"
    settings["connection"].update(
        cb_type=BACKEND, board_type=BOARD_DRIVER, task_format="SI",
        memriboard_root=str(MEMRIBOARD_ROOT), crossbar_serial=str(serial),
    )
    settings["geometry"]["initial_cells"] = [{"wl": wl, "bl": bl} for wl, bl in CELLS]
    settings["execution"].update(
        approved=True, simulator_enabled=True, hardware_enabled=False,
        approval_scope=f"simulator:{STAGE}:{READS_PER_CELL} reads per configured cell",
    )
    settings["read_baseline"]["enabled"] = True
    count_key = ("first_check_reads_per_cell" if STAGE == "first_check"
                 else "repeatability_reads_per_cell_after_first_check")
    settings["read_baseline"][count_key] = READS_PER_CELL
    snapshot = (json.dumps(settings, indent=2, allow_nan=False) + "\n").encode("utf-8")
    protocol = parse_protocol(snapshot)

    # 2. Создать НОВУЮ папку результатов и сохранить настройки/версии исходников.
    session_id = str(uuid.uuid4())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RESULTS_DIR / f"{stamp}_{session_id[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_bytes(snapshot)
    manifest = {
        "session_id": session_id, "protocol_id": protocol.protocol_id,
        "source": BACKEND, "stage": STAGE, "reads_per_cell": READS_PER_CELL,
        "cell_order": settings["geometry"]["initial_cells"],
        "protocol_sha256": hashlib.sha256(snapshot).hexdigest(),
        "template_sha256": hashlib.sha256(template).hexdigest(),
        "experiment_sha256": file_sha256(__file__),
        "array_file": str(array_file), "started_utc": utc_now(),
        "status": "failed", "records_saved": 0,
    }

    session = None
    try:
        manifest["source_sha256"] = {
            name: file_sha256(MEMRIBOARD_ROOT / name) for name in (
                "manager/board.py", "manager/service/drivers.py",
                "manager/service/converters.py", "simulator/src.py", "simulator/memristor.py",
            )
        }
        manifest["experiment_support_sha256"] = {
            name: file_sha256(PROJECT_DIR / "experiments" / "unn_rev3_20261014" / name)
            for name in ("config.py", "connector_session.py", "readback.py", "measurements.py")
        }
        manifest["array_sha256_before"] = file_sha256(array_file)

        # 3. Подключить штатный Connector с выбранным backend.
        session = connect(protocol, logger=logging.getLogger(__name__), approved=RUN_APPROVED)

        # 4. Прочитать указанные ячейки. Каждую запись сразу сохранить на диск.
        with (output_dir / "raw_readback.jsonl").open("x", encoding="utf-8") as stream:
            for wl, bl in protocol.initial_cells:
                for _ in range(READS_PER_CELL):
                    task_id = manifest["records_saved"] + 1
                    row = read_record(
                        session, protocol, session_id=session_id, array_id=serial.name,
                        phase=STAGE, task_id=task_id, wl=wl, bl=bl,
                    )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    manifest["records_saved"] += 1
                    if row["status"] != "ok":
                        raise RuntimeError(f"Чтение {task_id}: {row['error']}")
                    print(f"WL={wl}, BL={bl}: ADC={row['raw_adc']}, R={row['resistance_ohm']:.2f} Ohm")
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # 5. Закрыть соединение и сохранить итог, в том числе при ошибке.
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                manifest.update(status="failed", close_error=str(exc))
        if "array_sha256_before" in manifest:
            try:
                manifest["array_sha256_after"] = file_sha256(array_file)
                if manifest["array_sha256_after"] != manifest["array_sha256_before"]:
                    manifest.update(status="failed", array_error="Array file changed during readback")
            except OSError as exc:
                manifest.update(status="failed", array_error=str(exc))
        manifest["finished_utc"] = utc_now()
        with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
    if manifest["status"] != "completed":
        raise RuntimeError(f"Эксперимент остановлен. Подробности: {output_dir / 'manifest.json'}")
    print(f"Сохранено {manifest['records_saved']} чтений: {output_dir}")


if __name__ == "__main__":
    main()
