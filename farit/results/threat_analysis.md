# Threat Analysis

## 1. ExtractBench: A Benchmark for Schema-Guided Enterprise Document Extraction

### 1.1 Библиография

**Название:** ExtractBench: A Benchmark for Schema-Guided Enterprise Document Extraction
**Авторы:** Boyang Zhang, Adrian Lyjak, Eli Stewart, Zhaoqi Li, Simon Suo
**Аффилиации:** Llama (runllama.ai)
**Дата/Venue:** arXiv:2607.29677v2 [cs.AI], 5 Aug 2026
**Ссылки:** Dataset и evaluation code доступны на HuggingFace и GitHub

### 1.2 Суть

ExtractBench — это бенчмарк для схемо-управляемого извлечения данных из enterprise-документов. В отличие от фиксированных KIE-бенчмарков (SROIE, DocILE), ExtractBench позволяет подавать произвольную JSON-схему на вход и оценивает извлечение вместе с grounding (источником evidence), полнотой записей и стоимостью. Бенчмарк содержит 370 документов (4 869 страниц), 67 типов документов, 8 бизнес-доменов и 22 тега сложности.

### 1.3 Данные / Бенчмарк

- **Размер:** 370 документов, 4 869 страниц, 67 типов документов, 8 доменов
- **Домены:** финансы, энергетика, госзакупки, автосектор, цепочки поставок, здравоохранение, юриспруденция, недвижимость
- **Аннотации evidence:** Да. Ground truth содержит evidence list для каждого поля: ожидаемое значение, альтернативные чтения, номер страницы и word-level bounding box (где проверено человеком)
- **Полный набор evidence:** Частично. Evidence list может содержать несколько источников для одного значения (например, название препарата в заголовке, абзаце и таблице), но scoring использует OR-acceptance: достаточно найти хотя бы один источник
- **Типы документов по длине:**
  - Short (≤10 стр.): 252 документа (615 страниц)
  - Medium (11–50 стр.): 98 документов (2 438 страниц)
  - Long (>50 стр.): 20 документов (1 816 страниц)

### 1.4 Метрики — точные определения

**Unified Value F1:**
- JSON разворачивается в плоский набор ячеек (scalar fields + aligned record subfields)
- Для массивов записей применяется алгоритм Венгерского (Hungarian algorithm) для 1-к-1 сопоставления
- Нормализация: даты → ISO YYYY-MM-DD, whitespace схлопывается, сравнение exact и case-sensitive
- Leniencies: case_insensitive, optional_terminal_punctuation, phone_digits и др. (opt-in, per-field)
- OR-acceptance: prediction считается верным, если совпадает с ожидаемым значением или любым альтернативным чтением из evidence list

**Grounding Metrics:**
- **Word-level Grounding F1:** Поле засчитывается только если значение верно И predicted Bounding Box перекрывается с эталонным с IoU ≥ 0.5 на правильной странице
- **Page-level Grounding F1:** Требуется совпадение значения и правильного номера страницы
- Grounding precision denominator: только gradeable claims (citations на ячейках с box-bearing ground truth)
- Grounding recall denominator: ground-truth cells с verified box

**Completeness:**
- Recall и подписанный разрыв Δ = Precision − Recall
- Положительный Δ указывает на truncation (усечение) в длинных списках

**Cost:**
- Измеряется в центах за страницу (¢/page) на основе публичных API-тарифов на 1 июля 2026

### 1.5 Модели и главные числа

| Система | Overall Value F1 (%) | Short (L1) F1 (%) | Long (L3) F1 (%) | Word Grounding F1 (%) | Page Grounding F1 (%) | Cost (¢/page) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LlamaExtract Agentic Plus** | **95.6** | **96.6** | **94.4** | **46.4** | **84.9** | **8.1** |
| Codex GPT-5.5 | 93.6 | 95.7 | 78.9 | 0.0 | 0.0 | 27.8 |
| Reducto Deep Extract | 90.4 | 94.2 | 92.0 | 43.3 | 71.7 | 34.4 |
| LlamaExtract Agentic | 89.5 | 92.0 | 78.6 | 44.1 | 66.1 | 3.1 |
| Claude Code Opus 4.8 | 87.1 | 90.1 | 88.1 | 0.0 | 0.0 | 16.2 |
| LlamaExtract Cost-Eff. | 86.8 | 90.8 | 69.2 | 40.4 | 64.2 | 1.0 |
| Extend Max Context | 86.3 | 92.0 | 51.3 | 25.1 | 48.9 | 10.0 |
| Gemini 3.5 Flash | 79.8 | 87.9 | 27.9 | 0.0 | 0.0 | 1.0 |
| GPT-5.4 Nano | 74.9 | 77.4 | 35.8 | 0.0 | 0.0 | 0.21 |
| Datalab Accurate + Bal. | 64.5 | 62.8 | 40.5 | 2.0 | 48.5 | 3.5 |

**Ключевые наблюдения:**
- Commercial VLMs (Gemini, GPT) падают на длинных документах: Gemini 3.5 Flash с 87.9% до 27.9%
- Coding agents (Claude Code, Codex) дорогие (16–28 ¢/page) и не возвращают word-level boxes
- Word-level Grounding F1 остаётся низким даже у лучших: максимум 46.4%

### 1.6 Заявленные ограничения и то, что НЕ измеряется

1. **OR-acceptance вместо полноты evidence:** Система получает 100% балл, найдя хотя бы один источник из нескольких доступных. Не штрафуется за пропуск дополнительных evidence
2. **Нет явного лимита на бюджет чтения:** Стоимость измеряется постфактум, но система не ограничена в количестве токенов/страниц на этапе селекции
3. **Word-level grounding остаётся проблемой:** Максимум 46.4% F1
4. **Исключены физические нелинейные искажения сканов:** Заломы, изгибы страниц исключены, т.к. нельзя математически описать трансформацию Bounding Box
5. **Нет измерения cross-page reasoning:** Хотя есть теги T2.d (cross-reference/reconciliation), метрики не измеряют способность системы связывать evidence с разных страниц в единый вывод
6. **Нет abstention / отказа при недостаточной поддержке:** Система должна вернуть null для отсутствующих полей, но нет метрики калибровки уверенности

### 1.7 Пересечение с нашей темой

| Аспект | Статус | Цитата-доказательство |
|:---|:---|:---|
| (a) Полнота evidence-набора | **Частично** | OR-acceptance: "a prediction is correct when it matches the expected value or any recorded alternate reading" (Section 2.4, Appendix B.1). Система не штрафуется за нахождение только 1 из 5 доступных evidence |
| (b) Complementary premises / cross-page evidence | **Частично** | "T2.d (cross-reference / reconciliation) — сопоставление данных из разных секций" (Summary). Но метрики не требуют нахождения ВСЕХ complementary premises |
| (c) Alternative valid derivations | **Не тронуто** | Evidence list может содержать "genuinely ambiguous field has more than one defensible reading" (Appendix A.3), но scoring не измеряет способность системы находить альтернативные derivation paths |
| (d) Reading budget / стоимость чтения | **Частично** | "measured cost per page" в центах (Section 2.4), но "НЕ ограничивает систему в количестве токенов, которые она может прочитать" (Summary Section 7) |
| (e) Abstention / отказ при недостаточной поддержке | **Не тронуто** | "correctly use null for absent information" (Section 2.1), но нет метрики калибровки уверенности или abstention |

### 1.8 Релевантные цитируемые работы

- **SROIE [19], DocILE [39]:** Фиксированные KIE-бенчмарки, не поддерживают user-specified schemas (упомянуты как baseline сравнения)
- **VAREX [4]:** Multi-modal structured extraction (CVPR 2026 Workshop) — ближайший бенчмарк по схемо-управляемому извлечению, но без enterprise-масштаба
- **LongArray-Extract [9]:** Open-source array extraction benchmark — использован для сравнимости long-list scoring
- **Contextual AI ExtractBench [12]:** Несвязанный академический бенчмарк с тем же именем; не покрывает handwritten документы, visual grounding, rendering для synthetic lists

---

## 2. Benchmarking Large Language Models for Safety Data Extraction

### 2.1 Библиография

**Название:** Benchmarking Large Language Models for Safety Data Extraction
**Авторы:** Jonas Grill, Thomas Bayer (corresponding), Sören Berlinger
**Аффилиации:** SAP SE, Germany; Institute for Digital Transformation, Ravensburg-Weingarten University
**Дата/Venue:** arXiv:2606.11204v1 [cs.CL], 22 Apr 2026 (under review at Applied Intelligence)
**Email:** thomas.bayer@rwu.de

### 2.2 Суть

Систематический бенчмарк четырёх state-of-the-art LLM (Gemini 1.5 Pro, GPT-4o, Claude 3.7 Sonnet, Llama 3.1-70B) для извлечения структурированных данных из Safety Data Sheets (SDS). Сравниваются text-based и multimodal пайплайны с тремя стратегиями промптинга: zero-shot, few-shot, chain-of-thought. Оцениваются accuracy, latency и cost на >50 000 извлечённых полях.

### 2.3 Данные / Бенчмарк

- **Размер:** 10 SDS документов, ~50 000 labeled fields
- **Источник:** ChemicalSafety.com database
- **Домен:** Safety Data Sheets (химическая безопасность)
- **Аннотации evidence:** Нет. Только бинарный match indicator (true/false) на уровне поля
- **Полный набор evidence:** Нет. Нет аннотаций evidence-фрагментов, страниц или bounding boxes
- **Schema:** JSON schema с полями SDS (product identifiers, suppliers, composition, hazards, PPE и др.)

### 2.4 Метрики — точные определения

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + FP + FN + TN)
```
- Для каждого поля вычисляется бинарный match indicator (true/false)
- Агрегируется на уровне section, затем усредняется по документам

**Not-Found Rate (NF Rate):**
```
NF Rate = FN / (TP + FN)
```
- Доля required fields, присутствующих в SDS, но не извлечённых моделью

**False-Positive Rate (FP Rate):**
```
FP Rate = FP / (FP + TN)
```
- Доля fields, извлечённых моделью, но отсутствующих в ground truth

**BERTScore:**
- Семантическое сходство между extracted и reference text через contextual token embeddings (cosine similarity)

**Normalized Performance Score:**
```
Score = 0.7 · Accuracy_norm + 0.2 · Time_norm + 0.1 · Cost_norm
```
- Accuracy (0.7), latency (0.2), cost (0.1)

**Нет grounding metrics:** Нет page-level, word-level или box-level grounding. Нет evidence annotations.

### 2.5 Модели и главные числа

| Модель | Метод | Промпт | Accuracy | Примечания |
|:---|:---|:---|:---:|:---|
| Gemini 1.5 Pro | text-based | chain-of-thought | **0.84** | Лучший результат |
| GPT-4o | text-based | chain-of-thought | 0.81 | Самая низкая latency (~73s) |
| Claude 3.7 Sonnet | text-based | chain-of-thought | 0.79 | Средняя latency и cost |
| Llama 3.1-70B | text-based | chain-of-thought | 0.66 | Открытая модель, низкая точность |
| Gemini 1.5 Pro | text-based | zero-shot | 0.82 | |
| GPT-4o | text-based | zero-shot | 0.81 | |
| Claude 3.7 Sonnet | text-based | zero-shot | 0.78 | |
| Llama 3.1-70B | text-based | zero-shot | 0.70 | |
| Gemini 1.5 Pro | text-based | few-shot | 0.79 | |
| GPT-4o | text-based | few-shot | 0.80 | |
| Claude 3.7 Sonnet | text-based | few-shot | 0.79 | |
| Llama 3.1-70B | text-based | few-shot | 0.71 | |
| Gemini 1.5 Pro | multimodal | chain-of-thought | 0.73 | Хуже text-based |
| GPT-4o | multimodal | chain-of-thought | 0.76 | |
| Claude 3.7 Sonnet | multimodal | chain-of-thought | 0.77 | |

**Ключевые наблюдения:**
- Text-based consistently outperforms multimodal: +4–9 percentage points
- Multimodal latency значительно выше: GPT-4o ~300s vs ~73s text-based
- Prompting strategy влияет слабее, чем model choice
- Ни одна конфигурация не достигла 90% порога для safety-critical deployment

### 2.6 Заявленные ограничения и то, что НЕ измеряется

1. **Нет evidence annotations:** Только бинарный accuracy на уровне поля, без grounding, page/span/bbox
2. **Нет полноты evidence-набора:** Не измеряется, нашла ли модель все релевантные фрагменты
3. **Узкий scope:** Только 10 документов, "uncertain generalizability beyond the ten SDS"
4. **Sub-90% accuracy:** Ни одна модель не достигла порога для autonomous industrial use
5. **Нет reading budget / cost constraints:** Стоимость измеряется, но не является ограничением
6. **Нет abstention / calibration:** Нет метрик калибровки уверенности
7. **Нет cross-page reasoning:** Каждая section обрабатывается independently
8. **OCR artifacts в multimodal:** Дополнительные ошибки от OCR в image-based подходе

### 2.7 Пересечение с нашей темой

| Аспект | Статус | Цитата-доказательство |
|:---|:---|:---|
| (a) Полнота evidence-набора | **Не тронуто** | "For each field, a binary match indicator (true/false) is computed" (Section 3). Нет аннотаций evidence, нет метрик полноты |
| (b) Complementary premises / cross-page evidence | **Не тронуто** | "Each SDS section was processed independently to avoid cross-section leakage" (Section 3). Нет cross-page reasoning |
| (c) Alternative valid derivations | **Не тронуто** | Нет упоминания alternative derivations или reasoning paths в статье |
| (d) Reading budget / стоимость чтения | **Не тронуто** | "normalized cost function combines accuracy (0.7), processing time (0.2), and cost (0.1)" (Section 3), но cost не является ограничением на этапе retrieval |
| (e) Abstention / отказ при недостаточной поддержке | **Не тронуто** | "Future work should focus on... model calibration" (Abstract). Нет текущих метрик abstention |

### 2.8 Релевантные цитируемые работы

- **Khan et al. (2025) [3, 16]:** Machine learning-driven automated system для извлечения multiple fields из SDS — ближайшая предшествующая работа по SDS extraction
- **Fenton & Simske (2021, 2023) [14, 15]:** AI SDS document processing system для EHS compliance — early work, rule-based + neural hybrid
- **ChemTEB [21]:** Benchmark для SDS-specific language и domain-specialized embeddings — релевантно для нашей доменной специфики
- **Schilling-Wilhelmi et al. (2025) [20]:** Survey LLMs for chemical data extraction — обзорная статья, подтверждает растущий интерес к LLM для химических документов
- **Pekel et al. (2025) [18]:** GPT models для text recognition в SDS — применение generative models к SDS
- **SHACL/SKOS [22]:** Ontology-based representations для semantic validation SDS данных — complementary approach к нашему

---

---

## 3. LMDX: Language Model-based Document Information Extraction and Localization

### 3.1 Библиография

**Название:** LMDX: Language Model-based Document Information Extraction and Localization
**Авторы:** [авторы не указаны в доступном тексте]
**Аффилиации:** Google Research
**Дата/Venue:** [не указано в доступном тексте]
**Ссылки:** [не указано]

### 3.2 Суть

LMDX переформулирует задачу извлечения информации из визуально богатых документов (VRD) как последовательную текстовую генерацию. Модель получает текстовый промпт со встроенными токенами пространственных координат и целевой JSON-схемой, а на выходе генерирует структурированный JSON, где каждое значение сущности привязано к пространственному идентификатору текстового сегмента.

### 3.3 Данные / Бенчмарк

- **Задачи:** Form understanding (FUNSD, CORD), receipt understanding, document VQA
- **Аннотации evidence:** Да. Каждое значение привязано к координатам OCR-сегмента (page, span, bounding box)
- **Полный набор evidence:** Нет. LMDX извлекает одно значение на поле (или список значений для массивов), но не связывает несколько разрозненных мест в единую логическую цепочку доказательств
- **Многостраничность:** Страницы обрабатываются независимо (N независимых промптов), затем объединяются

### 3.4 Метрики — точные определения

**Extraction Accuracy:**
- Micro-F1 по извлечённым текстовым значениям
- Включая иерархические структуры (line_item)

**Localization Accuracy:**
```
Accuracy_Localization = N_E+L / N_E
```
- N_E+L — число верно извлечённых и корректно локализованных сущностей
- N_E — общее число верно извлечённых сущностей
- Локализация корректна, если predicted BBox накрывается эталонным с IoU > 80%

### 3.5 Модели и главные числа

| Модель | Задача | Micro-F1 | Localization |
|:---|:---|:---:|:---:|
| LMDX-Gemini Pro (zero-shot) | VRDU Reg. Form | 75.15% | 88-94% |
| LMDX-PaLM 2-S (zero-shot) | VRDU Reg. Form | 71.65% | 88-94% |
| LMDX-PaLM 2-S (finetuned, 10 docs) | VRDU Ad-Buy | 54.35% | 98-99.9% |
| GPT-4V (image) | VRDU Reg. Form | 65.34% | — |

**Ключевые наблюдения:**
- Zero-shot локализация: 88-94%
- Finetuned локализация: 98-99.9%
- Удаление токенов координат снижает F1 для line_item с 39.35% до 18.35%

### 3.6 Заявленные ограничения и то, что НЕ измеряется

1. **Нет cross-chunk / cross-page reasoning:** Чанки обрабатываются полностью независимо
2. **Нет complementary evidence:** Не связывает несколько разрозненных мест в единую цепочку
3. **Строковая гранулярность:** Локализация на уровне строк, не символов
4. **Только текстовые сущности:** Нет нетекстовых элементов
5. **Нет reading budget:** Каждая страница обрабатывается целиком
6. **Нет abstention:** Нет калибровки уверенности или отказа от ответа

### 3.7 Пересечение с нашей темой

| Аспект | Статус | Обоснование |
|:---|:---|:---|
| (a) Полнота evidence-набора | **Не тронуто** | Одно значение на поле, нет evidence set |
| (b) Complementary premises / cross-page | **Не тронуто** | Независимая обработка чанков |
| (c) Alternative valid derivations | **Не тронуто** | Нет упоминания |
| (d) Reading budget | **Не тронуто** | Каждая страница целиком |
| (e) Abstention | **Не тронуто** | Нет калибровки |

---

## 4. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control

### 4.1 Библиография

**Название:** Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control
**Авторы:** Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, Lihua Lei
**Аффилиации:** UC Berkeley, MIT, Stanford University, Stanford GSB
**Дата/Venue:** The Annals of Applied Statistics, 2025, Vol. 19, No. 2, 1641–1662
**Ссылки:** https://github.com/aangelopoulos/ltt

### 4.2 Суть

LTT — фреймворк для калибровки ML-моделей с finite-sample статистическими гарантиями. Использует multiple hypothesis testing для поиска параметров λ, которые контролируют риск R(T_λ) ≤ α с вероятностью ≥ 1-δ. Применяется к multilabel classification, selective classification, selective regression, OOD detection, instance segmentation.

### 4.3 Данные / Задачи

- **Задачи:** Object detection (COCO), multilabel classification, medical imaging, instance segmentation
- **Документы:** Нет. Задачи computer vision и tabular data
- **Аннотации evidence:** Нет

### 4.4 Методология

**Core Procedure:**
1. Для каждого λ_j: H_j : R(λ_j) > α
2. Вычислить p-value с помощью Hoeffding-Bentkus inequality
3. Применить FWER-controlling procedure (Bonferroni, fixed sequence testing, или sequential graphical testing)
4. Вернуть Λ̂ = {λ_j : p_j ≤ δ/|Λ|}

**Key Theorem:**
```
P(sup_{λ∈Λ̂} R(λ) ≤ α) ≥ 1 - δ
```

### 4.5 Заявленные ограничения и то, что НЕ измеряется

1. **Не работает с документами:** Computer vision и tabular data
2. **Не измеряет evidence retrieval:** Нет понятия evidence, grounding, completeness
3. **Не работает с reading budgets:** Нет ограничений на токены/страницы
4. **Не работает с cross-page reasoning:** Нет понятия страниц или документов
5. **Монотонность риска:** Хотя LTT работает с nonmonotone risks, приложения в статье — mostly monotone

### 4.6 Пересечение с нашей темой

| Аспект | Статус | Обоснование |
|:---|:---|:---|
| (a) Полнота evidence-набора | **Не тронуто** | Нет понятия evidence |
| (b) Complementary premises | **Не тронуто** | Нет документов |
| (c) Alternative valid derivations | **Не тронуто** | Нет reasoning paths |
| (d) Reading budget | **Не тронуто** | Нет ограничений на токены |
| (e) Abstention | **Частично** | Фреймворк для калибровки threshold, но не применён к document extraction |

**Как использовать:** LTT предоставляет статистическую основу для выбора порога abstention. Мы можем адаптировать его для калибровки confidence threshold в нашем пайплайне retrieval + extraction.

---

## 5. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

### 5.1 Библиография

**Название:** LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
**Авторы:** Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei
**Аффилиации:** Sun Yat-sen University, Microsoft Research Asia
**Дата/Venue:** MM '22, October 10–14, 2022, Lisboa, Portugal
**Ссылки:** https://aka.ms/layoutlmv3

### 5.2 Суть

Мультимодальная предобученная модель для Document AI, использующая unified text and image masking (MLM + MIM + WPA). Первый мультимодель в Document AI без CNN/Faster R-CNN backbone. Достигает SOTA на text-centric (FUNSD, CORD, DocVQA) и image-centric (RVL-CDIP, PubLayNet) задачах.

### 5.3 Данные / Бенчмарки

- **FUNSD:** Form understanding (text-centric)
- **CORD:** Receipt understanding (text-centric)
- **DocVQA:** Document visual question answering (text-centric)
- **RVL-CDIP:** Document image classification (image-centric)
- **PubLayNet:** Document layout analysis (image-centric)
- **Предобучение:** IIT-CDIP (11M документов)

### 5.4 Метрики — точные определения

**FUNSD / CORD:** F1 score
**DocVQA:** ANLS (Average Normalized Levenshtein Similarity)
**RVL-CDIP:** Accuracy
**PubLayNet:** mAP (mean Average Precision)

### 5.5 Модели и главные числа

| Модель | FUNSD F1 | CORD F1 | RVL-CDIP Acc | DocVQA ANLS |
|:---|:---:|:---:|:---:|:---:|
| LayoutLMv3_BASE | 90.29 | 95.25 | 95.44 | — |
| LayoutLMv3_LARGE | 92.08 | 97.46 | 95.93 | — |
| LayoutLMv2_BASE | 83.34 | — | 95.25 | — |
| BERT_BASE | 60.26 | 89.68 | — | — |

### 5.6 Заявленные ограничения и то, что НЕ измеряется

1. **Нет evidence retrieval:** Модель предсказывает значения, но не возвращает полный evidence set
2. **Нет reading budget:** Обрабатывает документ целиком
3. **Нет cross-page reasoning:** Ограничен контекстным окном (max 512 токенов)
4. **Нет abstention:** Нет калибровки уверенности
5. **Нет complementary premises:** Одно предсказание на поле
6. **Фиксированные задачи:** Не schema-guided, требует fine-tuning на каждую задачу

### 5.7 Пересечение с нашей темой

| Аспект | Статус | Обоснование |
|:---|:---|:---|
| (a) Полнота evidence-набора | **Не тронуто** | Одно предсказание, нет evidence set |
| (b) Complementary premises | **Не тронуто** | Нет cross-page reasoning |
| (c) Alternative valid derivations | **Не тронуто** | Нет reasoning |
| (d) Reading budget | **Не тронуто** | Обработка целиком |
| (e) Abstention | **Не тронуто** | Нет калибровки |

**Как использовать:** LayoutLMv3 — сильный baseline для document understanding и layout-aware представлений. Можно использовать как компонент для извлечения признаков из страниц, но не решает задачу complete evidence retrieval.

---

## 6. Итоговый вердикт K3

### Что остаётся открытым для нашей темы после всех пяти статей

1. **Полнота evidence-набора (Complete Evidence Set):** 
   - ExtractBench: OR-acceptance (достаточно 1 из N)
   - SDS LLM Benchmark: вообще не измеряет evidence
   - LMDX: одно значение на поле, нет evidence set
   - LayoutLMv3: одно предсказание, нет evidence set
   - Learn then Test: нет понятия evidence
   - **Наша задача:** Найти ВСЕ complementary premises и alternative valid derivations — принципиально другая метрика (Evidence Recall / Coverage@Evidence)

2. **Reading Budget / Cost of Reading:**
   - ExtractBench: измеряет стоимость постфактум (¢/page), но не ограничивает бюджет
   - SDS LLM Benchmark: использует полные документы
   - LMDX: каждая страница целиком
   - LayoutLMv3: обработка целиком
   - Learn then Test: нет ограничений на токены
   - **Наша задача:** Явное ограничение на количество прочитанных токенов/страниц и измерение trade-off между бюджетом и полнотой evidence

3. **Cross-page Evidence Fusion:**
   - ExtractBench: теги для cross-reference, но метрики не требуют связывания
   - SDS LLM Benchmark: sections обрабатываются independently
   - LMDX: чанки обрабатываются независимо
   - LayoutLMv3: ограничен контекстным окном (512 токенов)
   - Learn then Test: нет документов
   - **Наша задача:** Intentional cross-page reasoning для сбора complementary premises

4. **Alternative Valid Derivations:**
   - Ни одна из 5 статей не измеряет способность системы находить альтернативные пути вывода
   - **Наша задача:** substance_or_mixture может выводиться из разных комбинаций premises — ключевой аспект

5. **Abstention / Calibration:**
   - ExtractBench: требует null для отсутствующих полей, но не измеряет калибровку
   - SDS LLM Benchmark: упоминает calibration как future work
   - LMDX: нет калибровки
   - LayoutLMv3: нет калибровки
   - Learn then Test: фреймворк для калибровки threshold, но не применён к document extraction
   - **Наша задача:** Система должна отказаться от ответа при недостаточной поддержке evidence. Learn then Test может предоставить статистическую основу.

6. **Domain-specificity для SDS:**
   - ExtractBench: enterprise documents (финансы, недвижимость)
   - SDS LLM Benchmark: 10 документов без evidence annotations
   - LMDX: receipts, forms, invoices
   - LayoutLMv3: general Document AI
   - Learn then Test: computer vision, medical imaging
   - **Наша задача:** SDS (регуляторные документы химической безопасности) с 16 секциями GHS и специфическими требованиями к полноте evidence

7. **Word-level Grounding Quality:**
   - ExtractBench: максимум 46.4% Word Grounding F1
   - SDS LLM Benchmark: не измеряет grounding
   - LMDX: 88-94% (zero-shot), 98-99.9% (finetuned) — но строковая гранулярность
   - LayoutLMv3: не измеряет grounding для extraction
   - **Наша задача:** Точное grounding для аудита в регуляторных workflow

8. **Metrics for Evidence Completeness:**
   - Не существует стандартной метрики для оценки "нашёл ли retrieval model ВСЕ evidence-фрагменты"
   - **Наша задача:** Разработать Evidence Recall, Evidence Precision, Derivation Completeness

9. **Retrieval + Reading Budget + Reasoning Pipeline:**
   - ExtractBench: end-to-end extraction
   - SDS LLM Benchmark: end-to-end extraction
   - LMDX: end-to-end extraction + localization
   - LayoutLMv3: end-to-end understanding
   - Learn then Test: post-hoc calibration
   - **Наша задача:** Разделение на Stage 1 (retrieval under budget) и Stage 2 (cross-page evidence fusion & reasoning) с отдельными метриками

10. **Safety-critical Reliability:**
    - SDS LLM Benchmark: ни одна модель не достигает 90% порога
    - ExtractBench: не фокусируется на safety-critical доменах
    - **Наша задача:** Повышение надёжности через complete evidence retrieval и abstention для safety-critical deployment

---

## 7. Сводная таблица по всем 5 статьям

| Аспект | ExtractBench | SDS LLM Benchmark | LMDX | Learn then Test | LayoutLMv3 | Наша тема |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Полнота evidence-набора** | OR-acceptance (1 из N) | Не измеряет | Одно значение | Нет evidence | Одно значение | **ВСЕ premises + derivations** |
| **Reading budget** | Постфактум cost | Нет constraint | Целиком страница | Нет constraint | Целиком документ | **Явный лимит токенов** |
| **Cross-page reasoning** | Теги есть, метрики нет | Independent sections | Независимые чанки | Нет документов | Ограничено 512 токенами | **Intentional fusion** |
| **Alternative derivations** | Не тронуто | Не тронуто | Не тронуто | Не тронуто | Не тронуто | **Ключевой аспект** |
| **Abstention / Calibration** | Null, но не калибровка | Future work | Нет | Фреймворк (CV) | Нет | **Калиброванный отказ** |
| **Domain** | Enterprise (8 доменов) | SDS (10 доков) | Receipts/forms | CV/Medical | General Document AI | **SDS (29 доков, 252 evidence)** |
| **Grounding quality** | 46.4% Word F1 | Не измеряет | 88-99.9% (line-level) | Нет | Не измеряет | **Точное page/span/bbox** |
| **Evidence metrics** | OR-acceptance | Бинарный match | Localization Acc | Нет | Нет | **Evidence Recall/Precision** |
| **Pipeline** | End-to-end | End-to-end | End-to-end | Post-hoc calib | End-to-end | **Stage 1 + Stage 2** |
| **Safety-critical** | Нет фокуса | <90% accuracy | Нет фокуса | Нет фокуса | Нет фокуса | **Reliability + abstention** |

### Вывод

Ни одна из 5 статей не закрывает ни один из ключевых аспектов нашей темы. Наиболее близкие:
- **ExtractBench:** OR-acceptance вместо полноты, но есть grounding метрики
- **LMDX:** Localization + grounding, но нет complete evidence set
- **Learn then Test:** Можем адаптировать для калибровки abstention threshold
- **LayoutLMv3:** Сильный baseline для document understanding

**Research gap подтверждён:** Complete evidence retrieval under reading budgets для SDS остаётся открытой задачей.
