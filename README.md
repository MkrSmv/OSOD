# osod — детектор объектов с открытым множеством классов без текстовой модальности

Код, конфигурации, журнал экспериментов, таблицы и рисунки к работе «Детектор объектов с открытым множеством классов на основе методов извлечения сегментов изображений». Метод: классово-агностический сегментатор SAM 2 порождает маски, отбор гранулярности ρ оставляет по одной маске на объект, замороженный визуальный энкодер (DINOv2 или DINOv3) переводит каждую маску в эмбеддинг, решение принимается поиском ближайшего эталона в галерее, которая пополняется без переобучения. Веса моделей не обучаются, текстовая модальность не используется нигде, кроме метода сравнения OWLv2 в режиме image-guided, у которого текстовая башня не вызывается.

В репозитории только то, что стоит за числами глав 2–4 и приложения работы: код, которым получены записи журнала, сами записи, таблицы и рисунки, построенные из них. Нумерация экспериментов совпадает с разделами главы 4: 4.2 — предел полноты сегментатора, 4.3 — способ формирования эмбеддинга, 4.4 — отбор гранулярности, 4.5 — отклонение при пополнении галереи, 4.6 — сравнительный анализ.

## Состав репозитория

| Каталог | Содержимое |
|---|---|
| `src/` | пакет `src`: `data` (датасеты), `segment` (SAM 2, кеш масок), `select` (отбор гранулярности ρ), `encode` (варианты кодирования), `gallery` (галерея, дедупликация, подвыборки 4.5), `search` (правило решения), `calib` (порог отклонения), `eval` (протоколы, метрики, отклонение 4.5, правила, опубликованные числа), `baselines` (OWLv2) |
| `scripts/` | 28 скриптов: подготовка данных, пилотный замер со стоп-критерием для PKU-Market-PCB, кеш масок, галереи, прогоны экспериментов 4.2–4.5 и OWLv2, дедупликация галереи и подвыборки 4.5, сверки точности, аналитика калибровочных сцен, `make_tables.py`, `make_figures.py` и `render_mermaid.py` |
| `tests/` | сверка конфигураций сетки с константами кода |
| `configs/` | 22 конфигурации прогонов сетки, порождаются из констант кода командой `python -m src.config` |
| `splits/` | разбиения датасетов, список калибровочных снимков для сверок точности, списки подвыборок галереи для 4.5 |
| `experiments/runs/` | журнал: 49 записей JSON, по одной на прогон или замер, с конфигурацией, версиями, коммитом кода, метриками и повторами бутстрэпа |
| `experiments/tables/` | восемь файлов с полными таблицами, порождаются из журнала, номер таблицы в тексте стоит в её заголовке; `print.md` собирает таблицы 4.1–4.6 и А.1–А.7 в том виде, в каком они напечатаны в работе |
| `experiments/figures/` | семь рисунков работы, схемы 2.1 и 3.1 — вместе с исходниками Mermaid |
| `experiments/*.json`, `experiments/analysis/` | журнал окружения `env.json`, проверки раздачи, сверки точности и замеры времени, калибровка порога, сводки дедупликации галерей, аналитика калибровочных сцен, по которой заданы параметры §2.3, диагностики §4.3 |
| `requirements.lock.txt` | полный список пакетов среды, в которой получены числа (его хеш записан в `env.json`) |

## Установка

Нужен Python 3.12 и видеокарта с CUDA. После клонирования поставьте зависимости из `requirements.lock.txt` и запускайте команды из корня репозитория.

Версии зависимостей, при которых получены все числа, записаны в `experiments/env.json`: PyTorch 2.7.1 с CUDA 12.8, torchvision 0.22.1, transformers 4.57.6, SAM 2 из репозитория facebookresearch/sam2 на коммите 2b90b9f, faiss-cpu 1.15.0, numpy 2.5.2, pycocotools 2.0.11, scikit-learn 1.9.1, opencv-python-headless, pillow, matplotlib, pyyaml, pytest. Полный список — `requirements.lock.txt`, снимок всей рабочей среды, включая пакеты, которые коду не нужны. torch и torchvision в нём собраны под CUDA 12.8 и ставятся с индекса PyTorch:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock.txt
```

## Данные и веса

Датасеты ожидаются в `data/` и только читаются: HR-InsDet — в `data/InsDet-FULL/` (каталоги `Objects`, `Scenes`, `Background` раздачи InsDet-FULL), PKU-Market-PCB — в `data/pku_market_pcb/` (`images`, `Annotations`, `PCB_USED`; повёрнутые копии `rotation/` не используются). Веса моделей скачиваются из Hugging Face Hub при первом запуске `scripts/check_env.py`; DINOv3 закрыт лицензией: её нужно принять на странице `facebook/dinov3-vitl16-pretrain-lvd1689m` и задать токен доступа в переменной `HF_TOKEN`. Ревизии снимков моделей, при которых получены числа, записаны в `experiments/env.json` (раздел `models`).

Каталоги `data/`, `cache/`, `gallery/` и `logs/` в репозиторий не входят. Сборка таблиц читает разметку из `data/` и паспорта кеша масок и отбора ρ из `cache/`, рисунки 2.2 и 2.3 строятся из снимка датасета и кеша масок.

## Воспроизведение

Порядок команд повторяет порядок экспериментов в работе; каждая следующая группа читает то, что записала предыдущая. Все прогоны прерываемые: повторный запуск продолжает с места остановки. Существующие записи журнала не перезаписываются: скрипт, чья запись уже лежит в `experiments/`, сообщает об этом и выходит, поэтому для пересчёта эксперимента его запись нужно убрать из журнала и затем сравнить новую с прежней. Сверки точности (`check_segmenter.py`, `check_encoder.py`, `check_owlv2.py`) при повторном запуске заменяют свои разделы `experiments/env.json`; `check_env.py` опубликованный `env.json` не меняет, пока не задан `--overwrite`, а отчёт машины пишет в `logs/env_check.json`.

```bash
# окружение, данные, пилотный замер
python scripts/check_env.py                                   # веса моделей, отчёт об окружении
python scripts/check_data.py --dataset hr_insdet && python scripts/check_data.py --dataset pcb
python scripts/prepare_data.py --dataset hr_insdet && python scripts/prepare_data.py --dataset pcb
python scripts/pilot.py segment                               # пилотный замер и стоп-критерий PCB; остальные этапы — в заголовке скрипта

# сегментатор: сверка точности, кеш масок, 4.2
python scripts/check_segmenter.py precision                   # SAM 2 под bf16 против fp32, повторяемость — до кеша
python scripts/segment.py --dataset hr_insdet --crop-n-layers 1
python scripts/segment.py --dataset hr_insdet --crop-n-layers 0
python scripts/segment.py --dataset hr_insdet --mode box      # маски эталонов по рамке
python scripts/segment.py --dataset hr_insdet --crop-n-layers 1 --split cal --fp32 && python scripts/check_segmenter.py recall
python scripts/segment.py --dataset pcb --crop-n-layers 1     # только тестовые платы
python scripts/segment.py --dataset pcb --crop-n-layers 0     # нужен run_4_2.py --dataset pcb
python scripts/segment.py --dataset pcb --mode box            # эталоны плат 01 и 04
python scripts/check_segmenter.py pcb                         # повторный счёт снимков PCB против кеша
python scripts/run_4_2.py --dataset hr_insdet && python scripts/run_4_2.py --dataset pcb

# сверки кодирования (каждый этап — свой раздел env.json)
python scripts/check_encoder.py u_r                           # U_r для P⊥ → gallery/dinov3/p_perp/U_r.npy
python scripts/check_encoder.py memory && python scripts/check_encoder.py onepass && python scripts/check_encoder.py blur
python scripts/check_encoder.py precision && python scripts/check_encoder.py crop_precision && python scripts/check_encoder.py blur_refs
python scripts/check_crop_pipeline.py && python scripts/check_crop_pipeline.py --paired   # конвейер вырезок, парные проходы

# 4.3: сетка кодирования
python -m src.config                                          # конфигурации сетки (совпадают с configs/)
python scripts/build_gallery.py --config configs/dinov2_c_0_10_hr_insdet.yaml
python scripts/check_hnsw.py --config configs/dinov2_c_0_10_hr_insdet.yaml encode   # первый замер HNSW; затем measure и diagnose
python scripts/run_4_3.py --config configs/dinov2_c_0_10_hr_insdet.yaml             # первый прогон
python scripts/run_4_3.py --config configs/dinov2_c_0_10_hr_insdet.yaml --protocol baseline   # контроль на входе
python scripts/run_4_3.py --config configs/dinov3_c_0_10_hr_insdet.yaml
python scripts/run_4_3.py --config configs/dinov2_p_hr_insdet.yaml
python scripts/run_4_3.py --config configs/dinov2_<variant>_hr_insdet.yaml --config configs/dinov3_<variant>_hr_insdet.yaml   # общий проход: c_0_15, c_mean_10, c_mean_15, c_blur_10, c_blur_15
python scripts/run_4_3.py --config configs/dinov3_p_hr_insdet.yaml --config configs/dinov3_p_perp_hr_insdet.yaml
python scripts/run_4_3.py --baseline-best dinov2 && python scripts/run_4_3.py --baseline-best dinov3   # контрольные прогоны лучших φ
python scripts/check_hnsw.py --config configs/<encoder>_<variant>_hr_insdet.yaml grid   # полнота HNSW на остальных 14 галереях
python scripts/run_4_3.py --config configs/dinov2_c_mean_10_pcb.yaml && python scripts/run_4_3.py --config configs/dinov2_c_mean_10_pcb.yaml --protocol baseline
python scripts/run_4_3.py --config configs/dinov2_<variant>_pcb.yaml   # по порядку: c_0_10, c_0_15, c_mean_15, c_blur_10, c_blur_15, p
python scripts/analyze_4_3.py && python scripts/check_pool_crop.py   # разбор после результата, в таблицы не идёт

# аналитика §2.3 и отбор ρ
python scripts/analyze_cal_masks.py && python scripts/analyze_rho_cal.py && python scripts/analyze_rho_cal.py --part3
python scripts/select_rho.py --dataset hr_insdet              # M* всех сцен → cache/rho/
python scripts/select_rho.py --dataset hr_insdet --verify-cal # сверка с аналитикой → experiments/rho_verify_cal.json

# калибровка порога на лучших φ (нужны cache/rho/ и записи check_hnsw.py grid)
python scripts/calibrate.py --config configs/dinov2_c_mean_10_hr_insdet.yaml run && python scripts/calibrate.py --config configs/dinov2_c_mean_10_hr_insdet.yaml verify
python scripts/calibrate.py --config configs/dinov3_c_blur_15_hr_insdet.yaml run && python scripts/calibrate.py --config configs/dinov3_c_blur_15_hr_insdet.yaml verify

# 4.4: отбор гранулярности
python scripts/check_rho_encode.py run && python scripts/check_rho_encode.py measure   # замер кодирования с ρ, предусловие 4.4
python scripts/dedup_gallery.py --config configs/dinov2_c_mean_10_hr_insdet.yaml       # состав dedup.json галереи
python scripts/dedup_gallery.py --config configs/dinov3_c_blur_15_hr_insdet.yaml
python scripts/run_4_4.py --config configs/dinov2_c_mean_10_hr_insdet.yaml
python scripts/run_4_4.py --config configs/dinov3_c_blur_15_hr_insdet.yaml

# 4.5: отклонение при пополнении галереи
python scripts/make_gallery_subsets.py                        # сверяет splits/gallery_subsets_hr_insdet.json с пересчётом по seed
python scripts/run_4_5.py --config configs/dinov2_c_mean_10_hr_insdet.yaml
python scripts/run_4_5.py --config configs/dinov3_c_blur_15_hr_insdet.yaml

# 4.6: метод сравнения OWLv2 (после 4.4: парные разности берут повторы бутстрэпа из cache/exp_4_4/)
python scripts/check_owlv2.py stock                           # сверка прямого вызова головок со штатным путём
python scripts/run_owlv2.py --dataset hr_insdet && python scripts/run_owlv2.py --dataset hr_insdet --evaluate
python scripts/run_owlv2.py --dataset pcb && python scripts/run_owlv2.py --dataset pcb --evaluate

# таблицы и рисунки — только из журнала, без GPU
python scripts/make_tables.py && python scripts/make_figures.py && python scripts/render_mermaid.py
```

Сверка `check_owlv2.py stock` записывает в `env.json` отпечаток (sha256) файла `src/baselines/owlv2.py`, а `run_owlv2.py` сверяет его с текущим файлом и при расхождении не начинает счёт: сверка, пройденная другим кодом, прогон не открывает.

Заголовок каждого скрипта описывает его входы, выходы и условия запуска. Метрики качества считаются только при точном переборе; HNSW измеряется отдельно по полноте и задержке относительно точного перебора. Разрешение входа, лучший вариант кодирования, параметры отбора ρ, порог дедупликации η и порог отклонения выбираются только по калибровочным сценам, а на тестовых метрики лишь измеряются. Пороги, правила выбора и правила чтения результатов заданы константами в `src/eval/rules.py` до соответствующих прогонов. Правило ρ подобрано по калибровочным сценам, когда результаты сетки кодирования на тестовых сценах уже были известны (§4.1 работы). Тест конфигураций запускается командой `python -m pytest -q tests`.

## Оборудование

Все числа получены на одной видеокарте NVIDIA GeForce RTX 3050 с 6 ГБ памяти: SAM 2 под bf16, DINOv2 в fp16, DINOv3 в bf16, OWLv2 в fp32.
