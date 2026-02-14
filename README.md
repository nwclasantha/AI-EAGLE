# AI-EAGLE

**AI-Powered Credential Detection & Secret Scanning Tool**

<img width="2752" height="1488" alt="unnamed" src="https://github.com/user-attachments/assets/412eabfd-5329-4078-8db1-b215e314fe42" />

AI-EAGLE is a high-performance secret scanner that combines 887 rule-based detectors with an adaptive ML ensemble to find leaked credentials, API keys, tokens, and other secrets across your code, repositories, cloud storage, and infrastructure. It features multi-signal confidence scoring, real-time verification, and enterprise reporting.

> Python 3.11+ | 1,949 source files | 915 detectors | 264 tests passing | 98%+ precision, 95%+ recall

---

## Table of Contents

<img width="1335" height="723" alt="image" src="https://github.com/user-attachments/assets/41e47001-fa17-457f-8ed4-ef4408d6faa9" />

- [Features](#features)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Docker Usage](#docker-usage)
- [Scan Sources](#scan-sources)
- [Scan Commands - Full Reference](#scan-commands---full-reference)
- [Output Formats](#output-formats)
- [ML Scoring Pipeline](#ml-scoring-pipeline)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## Features

- **887+ Secret Detectors** - AWS, GitHub, GitLab, Slack, Stripe, OpenAI, Anthropic, Azure, GCP, and hundreds more
- **98%+ Precision, 95%+ Recall** - Multi-layer false positive suppression with benchmarked accuracy (264 tests)
- **Adaptive ML Ensemble** - 5-classifier pipeline pre-trained on 46 labeled examples, learns from verification feedback in real-time
- **Multi-Signal Confidence Scoring** - Shannon entropy, chi-squared, serial correlation, charset diversity, structural fingerprinting, contextual analysis
- **Real-Time Verification** - Validates detected secrets against live APIs to confirm they are active
- **15 Scan Sources** - Git, GitHub, GitLab, filesystem, S3, GCS, Docker, Elasticsearch, Postman, Jenkins, HuggingFace, CircleCI, TravisCI, Syslog, stdin
- **7 Output Formats** - Plain text, JSON, Legacy JSON, SARIF v2.1.0, HTML enterprise report, Excel/CSV, GitHub Actions annotations
- **Multi-Layer Caching** - L1 LRU + L2 hash dedup + L3 prefix matching for 60-80% computation savings
- **Enterprise Compliance Mapping** - SOC 2, PCI DSS, HIPAA, GDPR, ISO 27001, NIST framework alignment in HTML reports
- **Docker-First** - Multi-stage Docker build, runs as non-root user
- **High-Performance Engine** - 4-stage threaded pipeline with Aho-Corasick keyword pre-filtering

---

## Architecture

<img width="3617" height="6537" alt="NotebookLM Mind Map" src="https://github.com/user-attachments/assets/e2a47dd6-fb51-4bc7-ab0c-b22fcb538f63" />

### High-Level System Overview

https://github.com/user-attachments/assets/b5d1f05f-6716-4dcf-a504-f1310214ab18

```
                          AI-EAGLE Architecture
 ============================================================================

 +-----------+    +-----------+    +-----------+    +----------+    +--------+
 |  Sources  | -> | Decoders  | -> |  Engine   | -> | Analysis | -> | Output |
 |           |    |           |    | Pipeline  |    | Pipeline |    |        |
 | git       |    | UTF-8     |    |           |    |          |    | Plain  |
 | github    |    | UTF-16    |    | Aho-      |    | Entropy  |    | JSON   |
 | gitlab    |    | Base64    |    | Corasick  |    | ML       |    | SARIF  |
 | filesys   |    | Escaped   |    | Keyword   |    | Scoring  |    | HTML   |
 | s3/gcs    |    | Unicode   |    | Matching  |    | Risk     |    | Excel  |
 | docker    |    |           |    |           |    | Context  |    | GH Act |
 | elastic   |    +-----------+    | 887+      |    |          |    |        |
 | postman   |                     | Detectors |    +----------+    +--------+
 | jenkins   |                     |           |
 | huggingf  |                     | Verify    |    +----------+
 | circleci  |                     |           |    |  Cache   |
 | travisci  |                     +-----------+    | L1: LRU  |
 | syslog    |                                      | L2: Hash |
 | stdin     |                                      | L3: Pfx  |
 +-----------+                                      +----------+
```

### Component Overview

<img width="1343" height="721" alt="image" src="https://github.com/user-attachments/assets/e277e3bb-28fb-4378-899a-2e5667c824f0" />

| Component | Location | Purpose |
|-----------|----------|---------|
| **CLI** | `ai_eagle/cli.py` | Click-based command interface with 15 subcommands |
| **Engine** | `ai_eagle/engine/engine.py` | 4-stage threaded pipeline orchestrator (1,692 lines) |
| **Aho-Corasick** | `ai_eagle/engine/ahocorasick_core.py` | Keyword pre-filter for fast detector dispatch |
| **Detectors** | `ai_eagle/detectors/` | 887+ rule-based secret detectors with verification |
| **Decoders** | `ai_eagle/decoders/` | UTF-8, UTF-16, Base64, escaped unicode decoders |
| **Sources** | `ai_eagle/sources/` | 15 data source connectors |
| **Analysis** | `ai_eagle/analysis/` | ML classifier, entropy, scoring, risk, context |
| **Output** | `ai_eagle/output/` | 7 output formatters including enterprise reports |
| **Cache** | `ai_eagle/analysis/cache.py` | 3-layer analysis caching system |
| **Context** | `ai_eagle/context/context.py` | Go-style cancellation context with structured logging |
| **Dispatcher** | `ai_eagle/engine/dispatcher.py` | Result routing (PrinterDispatcher, MultiPrinter) |
| **Handlers** | `ai_eagle/handlers/handlers.py` | File handlers with ZIP/TAR archive support |
| **Source Manager** | `ai_eagle/sources/manager.py` | Thread-safe chunk queue for source connectors |

### 4-Stage Threaded Engine Pipeline

The engine uses a producer-consumer pipeline with `queue.Queue` channels and `WaitGroup` barriers (ported from Go's `sync.WaitGroup`). Each stage runs as a pool of daemon threads:

<img width="1326" height="710" alt="image" src="https://github.com/user-attachments/assets/bac194ec-76c7-470c-a3b9-dfb44745ac0f" />

```
 ============================================================================
 Threading Model (4-Stage Pipeline)
 ============================================================================

                   queue.Queue          queue.Queue          queue.Queue
   ┌──────────┐   (unbounded)   ┌──────────────┐  (unbnd)  ┌──────────┐
   │ Scanner  │ ──────────────> │   Detector   │ ────────> │ Notifier │
   │ Workers  │  _detectable_   │   Workers    │  _results │ Workers  │
   │          │  _chunks_queue  │              │  _queue   │          │
   │ 1x conc. │                 │  8x conc.    │           │ 1x conc. │
   └──────────┘                 └──────┬───────┘           └──────────┘
                                       │
                                       │ (optional)
                                       v
                                ┌──────────────┐
                                │ Verification │
                                │ Overlap      │
                                │ Workers      │
                                │ 1x conc.     │
                                └──────────────┘

 Synchronization:
   _workers_wg         → Tracks scanner worker completion
   _detector_workers_wg → Tracks detector worker completion
   _overlap_wg         → Tracks verification overlap completion
   _notifier_wg        → Tracks notifier worker completion

 Shutdown:
   _SENTINEL (poison pill) objects propagate through queues
   to signal each stage to shut down when upstream is done.

 Cancellation:
   Context._cancel_event (threading.Event) checked at each stage
   via ctx.done() for graceful Ctrl+C shutdown.
```

**Stage 1: Scanner Workers** (`1 x concurrency` threads)

<img width="1337" height="717" alt="image" src="https://github.com/user-attachments/assets/bf0ecc4b-57c6-4609-9ccb-8fb919e6886d" />

- Pull raw `Chunk` objects from `SourceManager`'s thread-safe queue
- Run each chunk through the **decoder chain** (UTF-8 → UTF-16 → Base64 → Escaped Unicode)
- Each successful decode produces a `DecodableChunk` with `decoder_type` tag
- Query `AhoCorasickCore` to find which detectors' keywords match this chunk
- Build `_DetectableChunk` objects (one per matched detector) and push to `_detectable_chunks_queue`
- Tracks duplicate chunks via `_LRUCache` (capacity 512) to avoid re-scanning identical data

**Stage 2: Detector Workers** (`8 x concurrency` threads, configurable via `--detector-worker-multiplier`)
- Pull `_DetectableChunk` from queue
- Check `ctx.done()` for cancellation before and during detection
- Execute the matched detector's `.from_data(ctx, verify, data)` method
- Each detector: regex match → candidate extraction → optional HTTP verification
- Apply false positive filtering via `get_false_positive_check()`
- Apply Levenshtein deduplication (0.9 similarity threshold) to avoid near-duplicate results
- Build `ResultWithMetadata` with source context, line numbers, and links
- Run through ML scoring + entropy analysis + risk assessment
- Push results to `_results_queue` (or `_verification_overlap_queue` if overlap detected)
- Per-detector timeout: `_DETECTION_TIMEOUT` (default 60s)

**Stage 3: Verification Overlap Workers** (`1 x concurrency` threads, optional)

<img width="1325" height="726" alt="image" src="https://github.com/user-attachments/assets/b2012a07-47e2-4d06-8764-6938f72c851a" />

- Enabled with `--allow-verification-overlap`
- Receives chunks where multiple detectors matched the same secret
- Re-verifies results across detector boundaries using all matched detectors
- Prevents security gaps where Detector A finds a secret but Detector B would have verified it

**Stage 4: Notifier Workers** (`1 x concurrency` threads, configurable via `--notification-worker-multiplier`)
- Pull `ResultWithMetadata` from `_results_queue`
- Apply `--results` filter (verified/unverified/unknown/filtered_unverified)
- Apply `--filter-entropy` threshold
- Route through `ResultsDispatcher` → `PrinterDispatcher` → configured output formatter
- Thread-safe printing: all printers use `threading.Lock()` (mutex) internally
- `MultiPrinter` fans out to multiple formatters simultaneously (e.g., terminal + HTML report)

### Key Data Types

```
Chunk                          # Raw bytes + source metadata
  ├── data: bytes              # Raw content to scan
  ├── source_name: str         # e.g., "github.com/org/repo"
  ├── source_type: SourceType  # Enum: GIT, GITHUB, FILESYSTEM, S3, ...
  ├── source_metadata: MetaData # File path, line number, link, commit
  └── verify: bool             # Whether to verify found secrets

DecodableChunk                 # After decoder chain
  ├── chunk: Chunk             # Decoded chunk (data is now UTF-8 bytes)
  └── decoder_type: DecoderType # Which decoder succeeded (UTF8, UTF16, BASE64)

DetectorMatch                  # Aho-Corasick match result
  ├── detector: Detector       # The matched detector instance
  ├── key: DetectorKey         # (detector_type, version) identifier
  └── _matches: list[bytes]    # Matched keyword byte spans

Result                         # Single detection result
  ├── detector_type: DetectorType # Which detector found it
  ├── raw: bytes               # Raw secret bytes
  ├── raw_v2: bytes            # Alternate representation
  ├── redacted: str            # Masked version for display
  ├── verified: bool           # Confirmed active via API
  ├── verification_error: str  # Error if verification failed
  └── extra_data: dict         # Detector-specific metadata

ResultWithMetadata             # Enriched result for output
  ├── result: Result           # Core detection result
  ├── source_metadata: MetaData # Where it was found
  ├── source_type: SourceType  # Source classification
  ├── confidence_score: float  # 0.0-1.0 ML confidence
  ├── severity: int            # 0=INFO, 1=LOW, 2=MED, 3=HIGH, 4=CRIT
  ├── risk_score: float        # Composite risk assessment
  ├── entropy: float           # Shannon entropy of the secret
  ├── remediation_priority: int # P1-P5 priority
  └── recommended_action: str  # Human-readable recommendation
```

### Decoder Chain Architecture

The decoder chain is a pipeline where each decoder attempts to transform the raw bytes. Every successful decode produces an additional `DecodableChunk`, so a single input chunk may yield multiple decoded forms (e.g., both UTF-8 and Base64 decoded):

```
Raw Chunk (bytes)
    │
    ├── UTF8Decoder     → Always succeeds (identity/pass-through)
    │                      Tags as DecoderType.PLAIN
    │
    ├── UTF16Decoder    → Detects BOM or null-byte interleaving
    │                      Auto-detects LE vs BE via ASCII ratio heuristic
    │                      Tags as DecoderType.UTF16
    │
    ├── Base64Decoder   → Regex: [A-Za-z0-9+/]{20,}={0,2}
    │                      Minimum 20 chars to avoid false matches
    │                      Tags as DecoderType.BASE64
    │
    └── EscapedUnicode  → Detects \uXXXX and \xXX escape sequences
       Decoder             Converts to actual Unicode characters
                           Tags as DecoderType.ESCAPED_UNICODE
```

### Aho-Corasick Keyword Pre-Filter

Instead of running all 887+ detectors against every chunk (O(N) per chunk), AI-EAGLE builds an **Aho-Corasick automaton** from all detector keywords at startup. This enables O(1)-like matching:

```
Startup:
  887 detectors → each provides keywords() → build automaton
  Example: AWS detector registers ["AKIA", "ASIA"]
           GitHub PAT detector registers ["ghp_", "github_pat_"]
           Stripe detector registers ["sk_live_", "pk_live_"]

Per-Chunk:
  Input: decoded chunk bytes
  Automaton scans bytes in single pass → returns matched keyword IDs
  Keyword IDs → map back to detector(s) → only those detectors run

  Result: Instead of 887 detectors per chunk, typically 1-5 run.
          This is the primary performance optimization.

Implementation:
  Uses ahocorasick-rs (Rust) if installed (pip install ai-eagle[fast])
  Falls back to pure Python implementation otherwise
```

### Detector Lifecycle

Each detector follows this standardized interface (ported from Go's `Detector` interface):

```python
class Detector(ABC):
    def keywords() -> list[str]     # Keywords for Aho-Corasick registration
    def type() -> DetectorType      # Enum identifier (e.g., DetectorType.AWS)
    def description() -> str        # Human-readable description
    def from_data(ctx, verify, data) -> list[Result]  # Core detection logic

# Optional interfaces (Protocol-based):
class Versioner:   version() -> int        # Multi-version detectors
class MaxSecretSizeProvider:  max_secret_size() -> int
class EndpointCustomizer:     set_endpoint_url(url)
```

Detector execution flow:
```
from_data(ctx, verify, data):
  1. Regex match against data bytes
  2. Extract candidate secret strings
  3. If verify=True:
     a. Build HTTP request to provider API
     b. Send request with candidate as auth credential
     c. If 200/OK → verified=True
     d. If 401/403 → verified=False
     e. If error → verification_error set
  4. Return list[Result] with raw, redacted, verified status
```

### Context & Cancellation System

AI-EAGLE uses a Go-style `Context` for cancellation propagation across all threads:

```
Context (ai_eagle/context/context.py)
  ├── _cancel_event: threading.Event    # Shared across all threads
  ├── _log: logging.Logger              # Structured logger
  ├── _values: dict[str, Any]           # Request-scoped values
  └── _cancel_cause: Optional[Exception] # Why cancellation happened

Signal Handler (SIGINT/SIGTERM):
  → ctx.cancel() → sets _cancel_event
  → All pipeline stages check ctx.done() in their loops
  → Graceful drain: queues flush, reports finalize, then exit

Key methods:
  ctx.done() → bool       # Check if cancelled
  ctx.cancel()             # Trigger cancellation
  ctx.with_cancel() → ctx  # Child context with own cancel
```

---

## Data Flow

### Complete Pipeline Visualization

<img width="1323" height="710" alt="image" src="https://github.com/user-attachments/assets/34df1897-4145-4e5f-b05e-c3e839f64c4c" />

```
Source Scan Request (CLI command)
        │
        v
 ┌──────────────┐
 │ CLI          │  Click parses args → builds EngineConfig
 │ (cli.py)     │  Sets up signal handlers (SIGINT/SIGTERM → ctx.cancel())
 └──────┬───────┘  Creates Context, Printer, SourceManager
        │
        v
 ┌──────────────┐
 │ Source        │  Reads data from git repos, filesystems, cloud, APIs
 │ Connector    │  Produces raw Chunks (bytes + source metadata)
 │              │  Thread-safe chunk queue via SourceManager
 └──────┬───────┘  Each source runs in its own thread
        │
        v
 ┌──────────────┐
 │ Decoder      │  UTF-8 → UTF-16 → Base64 → Escaped Unicode
 │ Chain        │  Each decoder attempts to decode, passes through on failure
 │              │  One input chunk → multiple DecodableChunks
 └──────┬───────┘  Dedup cache (LRU-512) prevents re-decoding identical data
        │
        v
 ┌──────────────┐
 │ Aho-Corasick │  Scans decoded chunks against all detector keywords
 │ Pre-Filter   │  887 detectors registered with unique keyword sets
 │              │  Single-pass automaton scan → matched detector list
 └──────┬───────┘  Only matched detectors proceed (typically 1-5 of 887)
        │
        v
 ┌──────────────┐
 │ Detector     │  Regex extraction → candidate secret isolation
 │ Execution    │  Each matched detector runs .from_data() on chunk
 │              │  Per-detector timeout: 60s (configurable)
 └──────┬───────┘  Produces Result objects (raw_secret, detector_type, etc.)
        │
        v
 ┌──────────────┐
 │ Verification │  HTTP calls to provider APIs to check if secrets are active
 │ (optional)   │  200/OK → verified=True | 401/403 → verified=False
 │              │  Levenshtein dedup (0.9 threshold) removes near-duplicates
 └──────┬───────┘  Respects --no-verification flag
        │
        v
 ┌──────────────┐
 │ Feature      │  Extracts 33 numerical features from each candidate:
 │ Extraction   │  Length, entropy, char distribution, statistical tests,
 │              │  structural patterns, context signals, detection metadata
 └──────┬───────┘  Output: SecretFeatures vector (all floats in [0, 1])
        │
        v
 ┌──────────────┐
 │ ML Ensemble  │  5-classifier adaptive pipeline (pure Python, no deps):
 │ Classifier   │  1. Adaptive Naive Bayes (n-gram learning)
 │              │  2. Online Logistic Regression (SGD)
 │              │  3. Decision Stump Ensemble (threshold tuning)
 │              │  4. Anomaly Detector (Welford's algorithm)
 │              │  5. Structural Fingerprint Matcher
 └──────┬───────┘  All classifiers learn online from verification feedback
        │
        v
 ┌──────────────┐
 │ Confidence   │  Weighted combination:
 │ Scoring      │    Entropy (35%) + Structure (30%) + Context (20%) + FP Risk (15%)
 │              │  Verified secrets → 0.95+ confidence override
 │              │  NaN/Inf guard prevents undefined scores
 └──────┬───────┘  Produces SecretScore with severity classification
        │
        v
 ┌──────────────┐
 │ Risk         │  Categories: credential, infrastructure, PII, config
 │ Assessment   │  Scopes: organization, repository, environment, service
 │              │  Priority: P1 (immediate) through P5 (minimal)
 └──────┬───────┘  Assigns remediation priority and recommended action
        │
        v
 ┌──────────────┐
 │ Result       │  Applies --results filter (verified/unverified/unknown)
 │ Filtering    │  Applies --filter-entropy threshold
 │              │  Applies --filter-unverified dedup
 └──────┬───────┘  Applies --include/exclude-detectors
        │
        v
 ┌──────────────┐
 │ Output       │  PrinterDispatcher → routes to configured formatter:
 │ Formatter    │    Plain   → color terminal with safe print (Windows)
 │              │    JSON    → one JSON object per line (jq-compatible)
 │              │    SARIF   → v2.1.0 for GitHub Code Scanning
 │              │    HTML    → enterprise report with compliance mapping
 │              │    Excel   → CSV/XLSX with remediation tracking
 │              │    GH Act  → inline PR annotations
 └──────────────┘  MultiPrinter fans out to multiple formatters
```

### Cache Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    AnalysisCache (Singleton)                    │
 │                    Thread-safe via threading.Lock               │
 │                                                                 │
 │  ┌─────────────────────────────────────────────────────────┐   │
 │  │ L1: In-Memory LRU Cache (exact match)                   │   │
 │  │                                                          │   │
 │  │  _l1_entropy   (10,000 cap) → EntropyReport results    │   │
 │  │  _l1_ml        (10,000 cap) → ML classification scores │   │
 │  │  _l1_risk      ( 5,000 cap) → Risk assessment results  │   │
 │  │  _l1_context   ( 5,000 cap) → Context validation       │   │
 │  │  _l1_features  (10,000 cap) → Extracted feature vectors│   │
 │  │                                                          │   │
 │  │  Key: SHA-256(secret)[:16]   O(1) get/put with LRU     │   │
 │  └─────────────────────────────────────────────────────────┘   │
 │                          │ miss                                 │
 │                          v                                      │
 │  ┌─────────────────────────────────────────────────────────┐   │
 │  │ L2: Hash-Based Dedup Cache (near-identical secrets)      │   │
 │  │                                                          │   │
 │  │  _l2_dedup (5,000 cap)                                  │   │
 │  │  Key: "dedup:{detector}:{SHA-256(secret)[:16]}"         │   │
 │  │  Prevents re-analyzing same secret from different chunks│   │
 │  └─────────────────────────────────────────────────────────┘   │
 │                          │ miss                                 │
 │                          v                                      │
 │  ┌─────────────────────────────────────────────────────────┐   │
 │  │ L3: Prefix-Based Structural Cache (known formats)        │   │
 │  │                                                          │   │
 │  │  Pre-populated at startup with 22 known prefixes:       │   │
 │  │  "AKIA" → aws_access_key (0.95)                        │   │
 │  │  "ghp_" → github_pat (0.95)                            │   │
 │  │  "sk_live_" → stripe_secret (0.95)                     │   │
 │  │  "eyJ" → jwt_token (0.85)                              │   │
 │  │  "-----BEGIN" → private_key (0.95)                     │   │
 │  │  ... (22 total)                                         │   │
 │  │                                                          │   │
 │  │  Longest-prefix matching for instant confidence floor   │   │
 │  └─────────────────────────────────────────────────────────┘   │
 │                          │ miss                                 │
 │                          v                                      │
 │                    Full Computation                              │
 └─────────────────────────────────────────────────────────────────┘

 Cache hit rates: 60-80% on typical scans
 Estimated time savings tracked in CacheStats.total_saves
```

### Source Connector Architecture

Each source connector implements the `Source` abstract base class:

```python
class Source(ABC):
    def type() -> SourceType                        # Source identifier
    def source_id() -> SourceID                     # Unique instance ID
    def job_id() -> JobID                           # Current job ID
    def init(ctx, name, job_id, verify) -> None     # Initialize source
    def chunks(ctx, chunk_chan, ...) -> None         # Produce chunks
    def get_progress() -> Progress                  # Scan progress info

# SourceManager orchestrates sources:
#   - Thread-safe chunk queue (queue.Queue)
#   - Configurable buffer size (default: 64)
#   - Concurrent source execution
```

Source → Chunk flow:
```
Git Source:
  git clone → iterate commits → iterate diffs → produce Chunks with:
    data = diff content bytes
    source_metadata = {file, commit, link, line, email, timestamp}

Filesystem Source:
  ThreadPoolExecutor → walk directories → read files → produce Chunks
    Handles ZIP/TAR archives via handlers.py
    Respects include/exclude path filters

S3 Source:
  boto3 list_objects → download each object → produce Chunks
    source_metadata = {bucket, key, region}

Stdin Source:
  sys.stdin.buffer.read() → single Chunk
    Useful for piping: git diff | ai-eagle stdin
```

### Output Dispatcher Architecture

```
Engine._results_queue
        │
        v
 ResultsDispatcher (interface)
        │
        v
 PrinterDispatcher
        │
        ├── Direct mode: single printer
        │     └── PlainPrinter / JSONPrinter / SARIFPrinter / ...
        │
        └── Multi mode: MultiPrinter
              ├── Primary: PlainPrinter (terminal)
              ├── Report:  HTMLReportPrinter (accumulates → flush)
              └── Report:  ExcelReportPrinter (accumulates → flush)

Each printer implements:
  print(ctx, result_with_metadata)  → output one result
  flush() → finalize (for accumulating printers like HTML/Excel)

Thread safety:
  All printers use threading.Lock() internally
  _safe_print() wrapper on Windows handles UnicodeEncodeError
```

---

## Installation

### From Source (pip)

```bash
# Clone the repository
git clone https://github.com/your-org/ai-eagle.git
cd ai-eagle

# Install with pip (Python 3.11+ required)
pip install .

# Optional: Install with fast Aho-Corasick (recommended)
pip install ".[fast]"

# Optional: Install dev dependencies
pip install ".[dev]"
```

### Docker (Recommended)

```bash
# Build the Docker image
docker build -t ai-eagle .

# Verify it works
docker run --rm ai-eagle --help
```

---

## Quick Start

<img width="1886" height="187" alt="image" src="https://github.com/user-attachments/assets/cea9efff-8577-4e60-9e2c-a9e22cbe46c7" />

```bash
# Scan a Git repository
ai-eagle git --repo https://github.com/example/test-keys.git

# Scan local files
ai-eagle filesystem --directory /path/to/code

# Scan from stdin
echo "AKIA_FAKE_EXAMPLE_00" | ai-eagle stdin

# Output as JSON
ai-eagle git --repo https://github.com/example/repo.git --json

# Generate enterprise HTML report
ai-eagle filesystem --directory ./code --html-report report.html
```

---

## Docker Usage

> **Docker Hub:** [`nwclasantha/ai-eagle`](https://hub.docker.com/r/nwclasantha/ai-eagle)
>
> ```bash
> docker pull nwclasantha/ai-eagle:latest    # or :3.0.0
> ```

### How Volume Mounts Work (Scanning Local Folders)

Docker containers can't see your local files by default. Use `-v` to mount a folder:

```
-v <YOUR_LOCAL_PATH>:/scan
```

This maps your local folder into `/scan` inside the container. Then scan `/scan`:

```bash
# Windows (CMD)
docker run --rm -it -v %cd%:/scan nwclasantha/ai-eagle filesystem /scan

# Windows (PowerShell)
docker run --rm -it -v ${PWD}:/scan nwclasantha/ai-eagle filesystem /scan

# Linux / macOS
docker run --rm -it -v $(pwd):/scan nwclasantha/ai-eagle filesystem /scan

# Specific folder
docker run --rm -it -v C:/Users/you/Desktop/project:/scan nwclasantha/ai-eagle filesystem /scan
```

---

### All Docker Run Commands

```bash
# ── Git Repository ──
docker run --rm -it ai-eagle git https://github.com/org/repo.git
docker run --rm -it ai-eagle git https://github.com/org/repo.git --json --branch main
docker run --rm  -it ai-eagle git https://github.com/org/repo.git --json --results verified --fail
docker run --rm -it ai-eagle git https://github.com/org/repo.git --max-depth 100 --exclude-globs "*.min.js,vendor/*"
docker run --rm -it ai-eagle git https://github.com/org/repo.git --sarif
docker run --rm -itai-eagle git https://github.com/org/repo.git --html-report /reports/scan.html -v $(pwd)/reports:/reports

# ── GitHub Organization/Repos ──
docker run --rm -it -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github --org my-org
docker run --rm -it -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github --repo my-org/repo --json
docker run --rm -it -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github --org my-org --include-forks --include-wikis --issue-comments --pr-comments
docker run --rm -it -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github --org my-org --include-repos "backend-*" --exclude-repos "archived-*"
docker run --rm -it -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github --org my-org --endpoint https://github.mycompany.com/api/v3

# ── GitHub Experimental (Object Discovery) ──
docker run --rm -it -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github-experimental --repo https://github.com/org/repo --token ghp_YOUR_TOKEN --object-discovery

# ── GitLab Group/Repos ──
docker run --rm -it -e GITLAB_TOKEN=glpat-YOUR_TOKEN ai-eagle gitlab --group-id 12345
docker run --rm -it -e GITLAB_TOKEN=glpat-YOUR_TOKEN ai-eagle gitlab --repo https://gitlab.com/org/repo --token glpat-YOUR_TOKEN
docker run --rm -it -e GITLAB_TOKEN=glpat-YOUR_TOKEN ai-eagle gitlab --group-id 42 --endpoint https://gitlab.mycompany.com

# ── Filesystem ──
# Basic scan
docker run --rm -it -v "C:/Users/nwcla/Desktop/SOC-Defense-System:/scan" nwclasantha/ai-eagle filesystem /scan

# JSON output + CI fail exit code
docker run --rm -it -v "C:/Users/nwcla/Desktop/SOC-Defense-System:/scan" nwclasantha/ai-eagle filesystem /scan --json --fail

# Scan everything (no default excludes)
docker run --rm -it -v "C:/Users/nwcla/Desktop/SOC-Defense-System:/scan" nwclasantha/ai-eagle filesystem /scan --no-default-excludes

# HTML + Excel enterprise reports (saved to your Desktop)
docker run --rm -it -v "C:/Users/nwcla/Desktop/SOC-Defense-System:/scan" -v "C:/Users/nwcla/Desktop/reports:/reports" nwclasantha/ai-eagle filesystem /scan --html-report /reports/report.html --excel-report /reports/findings.xlsx


# ── Amazon S3 ──
docker run --rm -it -e AWS_ACCESS_KEY_ID=AKIA... -e AWS_SECRET_ACCESS_KEY=... ai-eagle s3 --bucket my-bucket
docker run --rm -it -e AWS_ACCESS_KEY_ID=AKIA... -e AWS_SECRET_ACCESS_KEY=... ai-eagle s3 --bucket prod-configs --bucket staging-configs --ignore-bucket logs-bucket
docker run --rm -it -e AWS_ACCESS_KEY_ID=AKIA... -e AWS_SECRET_ACCESS_KEY=... -e AWS_SESSION_TOKEN=... ai-eagle s3 --bucket my-bucket --max-object-size 50MB

# ── Google Cloud Storage ──
docker run --rm -it -v /path/to/sa.json:/creds/sa.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds/sa.json ai-eagle gcs --project-id my-project --include-buckets my-bucket
docker run --rm -it -e GOOGLE_API_KEY=AIza... ai-eagle gcs --project-id my-project --include-buckets my-bucket
docker run --rm -it ai-eagle gcs --project-id my-project --without-auth --include-buckets public-bucket

# ── Docker Image Layers ──
docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock ai-eagle docker --image my-app:latest
docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock ai-eagle docker --image my-app:latest --image my-api:v2
docker run --rm -it ai-eagle docker --image registry.example.com/app:latest --token YOUR_TOKEN

# ── Elasticsearch ──
docker run --rm -it --network host ai-eagle elasticsearch --nodes http://localhost:9200 --index-pattern "logs-*"
docker run --rm -it ai-eagle elasticsearch --nodes http://elasticsearch:9200 --username elastic --password changeme --index-pattern "app-*"
docker run --rm -it ai-eagle elasticsearch --cloud-id my-deployment:dXMtY2... --api-key base64key
docker run --rm -it ai-eagle elasticsearch --nodes http://elasticsearch:9200 --since-timestamp "2024-01-01T00:00:00Z"

# ── Postman ──
docker run --rm -it -e POSTMAN_TOKEN=PMAK-... ai-eagle postman --workspace-id abc123
docker run --rm -it -e POSTMAN_TOKEN=PMAK-... ai-eagle postman --collection-id col-123 --collection-id col-456

# ── Jenkins ──
docker run --rm -it ai-eagle jenkins --url http://jenkins:8080 --username admin --password admin123
docker run --rm -it ai-eagle jenkins --url https://jenkins.internal:8443 --username admin --password admin123 --insecure-skip-verify-tls

# ── HuggingFace ──
docker run --rm -it -e HUGGINGFACE_TOKEN=hf_... ai-eagle huggingface --model username/model-name
docker run --rm -it -e HUGGINGFACE_TOKEN=hf_... ai-eagle huggingface --org my-org
docker run --rm -it -e HUGGINGFACE_TOKEN=hf_... ai-eagle huggingface --dataset username/dataset-name
docker run --rm -it -e HUGGINGFACE_TOKEN=hf_... ai-eagle huggingface --user someone --skip-all-models --include-discussions --include-prs

# ── CircleCI ──
docker run --rm -it -e CIRCLECI_TOKEN=... ai-eagle circleci --token $CIRCLECI_TOKEN

# ── TravisCI ──
docker run --rm -it -e TRAVIS_TOKEN=... ai-eagle travisci --token $TRAVIS_TOKEN

# ── Syslog (Real-Time) ──
docker run --rm -it -p 5140:5140/udp ai-eagle syslog --address 0.0.0.0:5140 --protocol udp --format rfc5424
docker run --rm -it -p 6514:6514/tcp ai-eagle syslog --address 0.0.0.0:6514 --protocol tcp --format rfc5424

# ── Stdin (Pipe Data) ──
echo "sk_live_4eC39HqLyjWDarjtT1zdp7dc" | docker run --rm -i ai-eagle stdin
cat config.env | docker run --rm -i ai-eagle stdin --json
git diff HEAD~5 | docker run --rm -i ai-eagle stdin --json --fail

# ── Multi-Scan (Config File) ──
docker run --rm -v $(pwd)/scan-config.yaml:/config.yaml ai-eagle multi-scan --config /config.yaml

# ── Analyze API Key Permissions ──
docker run --rm -it ai-eagle analyze --key-type github --key key=ghp_YOUR_TOKEN
docker run --rm -it ai-eagle analyze --key-type stripe --key key=sk_live_YOUR_KEY
docker run --rm -it ai-eagle analyze --key-type twilio --key sid=AC123 --key key=auth_token
```

---

## Scan Sources

| Source | Command | Auth | Description |
|--------|---------|------|-------------|
| Git | `git` | None (public) / SSH key | Clone and scan any Git repository |
| GitHub | `github` | `GITHUB_TOKEN` env | Scan repos, orgs, users via GitHub API |
| GitHub Experimental | `github-experimental` | `GITHUB_TOKEN` env | Object discovery mode (alpha) |
| GitLab | `gitlab` | `--token` | Scan repos, groups via GitLab API |
| Filesystem | `filesystem` | None | Scan local files and directories |
| S3 | `s3` | AWS creds (env/profile) | Scan Amazon S3 buckets |
| GCS | `gcs` | Service account JSON | Scan Google Cloud Storage buckets |
| Docker | `docker` | Docker socket | Scan Docker image layers |
| Elasticsearch | `elasticsearch` | `--nodes` URL | Scan Elasticsearch indices |
| Postman | `postman` | `POSTMAN_TOKEN` | Scan Postman workspaces/collections |
| Jenkins | `jenkins` | `--username`/`--password` | Scan Jenkins build artifacts |
| HuggingFace | `huggingface` | `HUGGINGFACE_TOKEN` | Scan HuggingFace models/datasets |
| CircleCI | `circleci` | `--token` | Scan CircleCI build configs |
| TravisCI | `travisci` | `--token` | Scan TravisCI build logs |
| Syslog | `syslog` | `--address` (listen) | Real-time syslog stream scanning |
| Stdin | `stdin` | Pipe input | Scan piped data from any source |
| Multi-Scan | `multi-scan` | `--config` YAML | Multiple sources from a config file |
| Analyze | `analyze` | Varies by key type | Analyze API key permissions (40+ providers) |

---

## Scan Commands - Full Reference

Every scan command accepts all [Global Options](#cli-flags) in addition to its own source-specific options.

### `git` - Scan Git Repositories

Clones and scans the full commit history of any Git repository.

```bash
# Basic scan
ai-eagle git https://github.com/org/repo.git

# Scan a specific branch from a certain commit
ai-eagle git https://github.com/org/repo.git --branch main --since-commit abc123

# Limit commit depth and exclude certain files
ai-eagle git https://github.com/org/repo.git --max-depth 100 --exclude-globs "*.min.js,vendor/*"

# Scan a bare repo with custom clone path
ai-eagle git https://github.com/org/repo.git --bare --clone-path /tmp/scan

# JSON output, only verified secrets, CI exit code
ai-eagle git https://github.com/org/repo.git --json --results verified --fail
```

| Option | Type | Description |
|--------|------|-------------|
| `URI` | argument (required) | Git repository URL or local path |
| `-i, --include-paths` | string | Path to file with regexes for files to include |
| `-x, --exclude-paths` | string | Path to file with regexes for files to exclude |
| `--exclude-globs` | string | Comma-separated list of globs to exclude |
| `--since-commit` | string | Commit hash to start scan from |
| `--branch` | string | Branch to scan |
| `--max-depth` | int | Maximum depth of commits to scan |
| `--bare` | flag | Scan a bare repository |
| `--clone-path` | string | Custom clone path |
| `--no-cleanup` | flag | Do not delete cloned repos after scanning |

**Docker:**
```bash
docker run --rm ai-eagle git https://github.com/org/repo.git
docker run --rm ai-eagle git https://github.com/org/repo.git --json --branch main
```

---

### `github` - Scan GitHub Organizations & Repos

Scans GitHub repositories, orgs, wikis, issues, PRs, and gists via the GitHub API.

```bash
# Scan an entire organization
ai-eagle github --org my-org --token ghp_YOUR_TOKEN

# Scan specific repos
ai-eagle github --repo my-org/repo1 --repo my-org/repo2 --token ghp_YOUR_TOKEN

# Scan org including forks, wikis, issue comments, and PR comments
ai-eagle github --org my-org --token ghp_YOUR_TOKEN \
  --include-forks --include-wikis --issue-comments --pr-comments

# Scan with repo include/exclude globs
ai-eagle github --org my-org --token ghp_YOUR_TOKEN \
  --include-repos "backend-*" --exclude-repos "archived-*"

# Scan against GitHub Enterprise
ai-eagle github --org my-org --token ghp_YOUR_TOKEN \
  --endpoint https://github.mycompany.com/api/v3
```

| Option | Type | Description |
|--------|------|-------------|
| `--endpoint` | string | GitHub API endpoint (default: `https://api.github.com`) |
| `--repo` | string (repeatable) | GitHub repository to scan |
| `--org` | string (repeatable) | GitHub organization to scan |
| `--token` | string / `GITHUB_TOKEN` env | GitHub personal access token |
| `--include-forks` | flag | Include forked repositories |
| `--include-members` | flag | Include member repos in org scan |
| `--include-repos` | string (repeatable) | Repos to include in org scan (glob) |
| `--include-wikis` | flag | Include repository wikis |
| `--exclude-repos` | string (repeatable) | Repos to exclude (glob) |
| `-i, --include-paths` | string | Include paths file |
| `-x, --exclude-paths` | string | Exclude paths file |
| `--issue-comments` | flag | Include issue comments |
| `--pr-comments` | flag | Include PR comments |
| `--gist-comments` | flag | Include gist comments |
| `--comments-timeframe` | int | Days to review comments |
| `--clone-path` | string | Custom clone path |
| `--no-cleanup` | flag | Do not delete cloned repos |
| `--ignore-gists` | flag | Ignore all gists |

**Docker:**
```bash
docker run --rm -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github --org my-org
docker run --rm -e GITHUB_TOKEN=ghp_YOUR_TOKEN ai-eagle github --repo my-org/repo --json
```

---

### `github-experimental` - GitHub Object Discovery (Alpha)

Experimental mode using GitHub's object discovery for deeper scanning.

```bash
ai-eagle github-experimental --repo https://github.com/org/repo --token ghp_YOUR_TOKEN \
  --object-discovery --collision-threshold 3
```

| Option | Type | Description |
|--------|------|-------------|
| `--repo` | string (required) | GitHub repository URL |
| `--token` | string (required) / `GITHUB_TOKEN` env | GitHub token |
| `--object-discovery` | flag | Enable object discovery mode |
| `--collision-threshold` | int | Collision detection threshold (default: 1) |
| `--delete-cached-data` | flag | Delete cached data before scanning |

---

### `gitlab` - Scan GitLab Groups & Repos

Scans GitLab repositories and groups via the GitLab API.

```bash
# Scan a GitLab group
ai-eagle gitlab --group-id 12345 --token glpat-YOUR_TOKEN

# Scan specific repos
ai-eagle gitlab --repo https://gitlab.com/org/repo --token glpat-YOUR_TOKEN

# Scan self-hosted GitLab
ai-eagle gitlab --group-id 42 --token glpat-YOUR_TOKEN \
  --endpoint https://gitlab.mycompany.com

# Include/exclude specific repos in a group
ai-eagle gitlab --group-id 12345 --token glpat-YOUR_TOKEN \
  --include-repos "frontend-*" --exclude-repos "legacy-*"
```

| Option | Type | Description |
|--------|------|-------------|
| `--endpoint` | string | GitLab endpoint (default: `https://gitlab.com`) |
| `--repo` | string (repeatable) | GitLab repo URL |
| `--token` | string (required) / `GITLAB_TOKEN` env | GitLab personal access token |
| `--group-id` | string (repeatable) | GitLab group ID |
| `-i, --include-paths` | string | Include paths file |
| `-x, --exclude-paths` | string | Exclude paths file |
| `--include-repos` | string (repeatable) | Repos to include (glob) |
| `--exclude-repos` | string (repeatable) | Repos to exclude (glob) |
| `--clone-path` | string | Custom clone path |
| `--no-cleanup` | flag | Do not delete cloned repos |

**Docker:**
```bash
docker run --rm -e GITLAB_TOKEN=glpat-YOUR_TOKEN ai-eagle gitlab --group-id 12345
```

---

### `filesystem` - Scan Local Files & Directories

Scans local files and directories with smart default exclusions.

```bash
# Scan a directory
ai-eagle filesystem /path/to/code

# Scan multiple paths
ai-eagle filesystem /path/to/project1 /path/to/project2

# Scan with inline exclude patterns
ai-eagle filesystem /code --exclude-pattern '\.venv' --exclude-pattern 'node_modules'

# Scan everything (disable default exclusions for .venv, node_modules, etc.)
ai-eagle filesystem /code --no-default-excludes

# Generate HTML + Excel reports
ai-eagle filesystem /code --html-report report.html --excel-report findings.csv
```

| Option | Type | Description |
|--------|------|-------------|
| `PATHS` | argument (repeatable) | One or more directories/files to scan |
| `-i, --include-paths` | string | Include paths file |
| `-x, --exclude-paths` | string | Exclude paths file |
| `--exclude-pattern` | string (repeatable) | Inline regex pattern to exclude (e.g. `\.venv`) |
| `--no-default-excludes` | flag | Disable default exclusions (.venv, node_modules, .git, etc.) |

**Docker:**
```bash
docker run --rm -v /path/to/code:/scan ai-eagle filesystem /scan
docker run --rm -v /code:/scan ai-eagle filesystem /scan --json --fail
```

---

### `s3` - Scan Amazon S3 Buckets

Scans objects in S3 buckets using AWS credentials.

```bash
# Scan using environment credentials
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
ai-eagle s3 --bucket my-bucket

# Scan multiple buckets, ignore some
ai-eagle s3 --bucket prod-configs --bucket staging-configs --ignore-bucket logs-bucket

# Scan with IAM role assumption
ai-eagle s3 --role-arn arn:aws:iam::123456:role/Scanner --bucket my-bucket

# Scan in cloud environment (EC2/ECS/Lambda) using instance profile
ai-eagle s3 --cloud-environment --bucket my-bucket

# Limit object size
ai-eagle s3 --bucket my-bucket --max-object-size 50MB
```

| Option | Type | Description |
|--------|------|-------------|
| `--key` | string / `AWS_ACCESS_KEY_ID` env | AWS access key |
| `--secret` | string / `AWS_SECRET_ACCESS_KEY` env | AWS secret key |
| `--session-token` | string / `AWS_SESSION_TOKEN` env | AWS session token |
| `--bucket` | string (repeatable) | S3 bucket name |
| `--ignore-bucket` | string (repeatable) | S3 bucket to ignore |
| `--role-arn` | string (repeatable) | IAM role ARN to assume |
| `--cloud-environment` | flag | Use IAM credentials in cloud |
| `--max-object-size` | string | Maximum object size (default: `250MB`) |

**Docker:**
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=AKIA... \
  -e AWS_SECRET_ACCESS_KEY=... \
  ai-eagle s3 --bucket my-bucket --region us-east-1
```

---

### `gcs` - Scan Google Cloud Storage Buckets

Scans objects in GCS buckets using Google credentials.

```bash
# Scan with service account
ai-eagle gcs --project-id my-project --service-account /path/to/sa.json \
  --include-buckets my-bucket

# Scan using Application Default Credentials
ai-eagle gcs --project-id my-project --cloud-environment

# Scan without auth (public buckets)
ai-eagle gcs --project-id my-project --without-auth --include-buckets public-bucket

# Scan with API key, exclude certain objects
ai-eagle gcs --project-id my-project --api-key AIza... \
  --include-buckets my-bucket --exclude-objects "*.log"
```

| Option | Type | Description |
|--------|------|-------------|
| `--project-id` | string / `GOOGLE_CLOUD_PROJECT` env | GCS project ID |
| `--cloud-environment` | flag | Use Application Default Credentials |
| `--service-account` | string | Service account JSON file path |
| `--without-auth` | flag | Scan without authentication (public buckets) |
| `--api-key` | string / `GOOGLE_API_KEY` env | GCS API key |
| `--include-buckets` | string (repeatable) | Buckets to include |
| `--exclude-buckets` | string (repeatable) | Buckets to exclude |
| `--include-objects` | string (repeatable) | Objects to include |
| `--exclude-objects` | string (repeatable) | Objects to exclude |
| `--max-object-size` | string | Maximum object size (default: `10MB`) |

**Docker:**
```bash
docker run --rm \
  -v /path/to/sa.json:/creds/sa.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds/sa.json \
  ai-eagle gcs --project-id my-project --include-buckets my-bucket
```

---

### `docker` - Scan Docker Image Layers

Scans all layers of a Docker image for secrets.

```bash
# Scan a local image
ai-eagle docker --image my-app:latest

# Scan multiple images
ai-eagle docker --image my-app:latest --image my-api:v2

# Scan with authentication (private registry)
ai-eagle docker --image registry.example.com/app:latest --token YOUR_TOKEN

# Exclude paths inside the image
ai-eagle docker --image my-app:latest --exclude-paths "/usr/lib,/var/cache"
```

| Option | Type | Description |
|--------|------|-------------|
| `--image` | string (repeatable) | Docker image to scan |
| `--token` | string / `DOCKER_TOKEN` env | Docker bearer token |
| `--exclude-paths` | string | Comma-separated paths to exclude inside image |
| `--namespace` | string | Docker namespace |
| `--registry-token` | string | Docker registry access token |

**Docker:**
```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  ai-eagle docker --image my-app:latest
```

---

### `elasticsearch` - Scan Elasticsearch Indices

Scans documents in Elasticsearch clusters.

```bash
# Scan with basic auth
ai-eagle elasticsearch --nodes http://localhost:9200 \
  --username elastic --password changeme --index-pattern "app-*"

# Scan Elastic Cloud
ai-eagle elasticsearch --cloud-id my-deployment:dXMtY2... --api-key base64key

# Scan with service token
ai-eagle elasticsearch --nodes http://elasticsearch:9200 --service-token YOUR_TOKEN

# Scan documents since a specific time
ai-eagle elasticsearch --nodes http://localhost:9200 --since-timestamp "2024-01-01T00:00:00Z"

# Continuous best-effort scan
ai-eagle elasticsearch --nodes http://localhost:9200 --best-effort-scan
```

| Option | Type | Description |
|--------|------|-------------|
| `--nodes` | string (repeatable) / `ELASTICSEARCH_NODES` env | Elasticsearch node URLs |
| `--username` | string / `ELASTICSEARCH_USERNAME` env | Username |
| `--password` | string / `ELASTICSEARCH_PASSWORD` env | Password |
| `--service-token` | string / `ELASTICSEARCH_SERVICE_TOKEN` env | Service token |
| `--cloud-id` | string / `ELASTICSEARCH_CLOUD_ID` env | Elastic Cloud ID |
| `--api-key` | string / `ELASTICSEARCH_API_KEY` env | API key |
| `--index-pattern` | string | Index pattern filter (default: `*`) |
| `--query-json` | string | Document query filter (JSON) |
| `--since-timestamp` | string | Filter documents since timestamp |
| `--best-effort-scan` | flag | Continuous cluster scan |

**Docker:**
```bash
docker run --rm --network host ai-eagle elasticsearch \
  --nodes http://localhost:9200 --index-pattern "logs-*"
```

---

### `postman` - Scan Postman Workspaces

Scans Postman workspaces, collections, and environments.

```bash
# Scan a workspace
ai-eagle postman --workspace-id abc123 --token PMAK-...

# Scan specific collections
ai-eagle postman --collection-id col-123 --collection-id col-456 --token PMAK-...

# Include/exclude specific collections and environments
ai-eagle postman --workspace-id abc123 --token PMAK-... \
  --include-collection-id col-789 --exclude-environments "dev,staging"
```

| Option | Type | Description |
|--------|------|-------------|
| `--token` | string / `POSTMAN_TOKEN` env | Postman API token |
| `--workspace-id` | string (repeatable) | Postman workspace ID |
| `--collection-id` | string (repeatable) | Postman collection ID |
| `--environment` | string (repeatable) | Postman environment |
| `--include-collection-id` | string (repeatable) | Collection IDs to include |
| `--include-environments` | string (repeatable) | Environments to include |
| `--exclude-collection-id` | string (repeatable) | Collection IDs to exclude |
| `--exclude-environments` | string (repeatable) | Environments to exclude |

**Docker:**
```bash
docker run --rm -e POSTMAN_TOKEN=PMAK-... \
  ai-eagle postman --workspace-id abc123
```

---

### `jenkins` - Scan Jenkins Build Artifacts

Scans Jenkins build configurations and artifacts.

```bash
# Scan Jenkins
ai-eagle jenkins --url http://jenkins:8080 --username admin --password admin123

# Scan with header-based auth
ai-eagle jenkins --url http://jenkins:8080 --header-key Authorization --header-value "Bearer token"

# Skip TLS verification (self-signed certs)
ai-eagle jenkins --url https://jenkins.internal:8443 \
  --username admin --password admin123 --insecure-skip-verify-tls
```

| Option | Type | Description |
|--------|------|-------------|
| `--url` | string (required) / `JENKINS_URL` env | Jenkins URL |
| `--username` | string / `JENKINS_USERNAME` env | Jenkins username |
| `--password` | string / `JENKINS_PASSWORD` env | Jenkins password |
| `--header-key` | string | Header auth key |
| `--header-value` | string | Header auth value |
| `--insecure-skip-verify-tls` | flag / `JENKINS_INSECURE_SKIP_VERIFY_TLS` env | Skip TLS verification |

**Docker:**
```bash
docker run --rm ai-eagle jenkins \
  --url http://jenkins:8080 --username admin --password admin123
```

---

### `huggingface` - Scan HuggingFace Models, Datasets & Spaces

Scans HuggingFace models, datasets, spaces, discussions, and PRs.

```bash
# Scan a specific model
ai-eagle huggingface --model username/model-name --token hf_YOUR_TOKEN

# Scan an organization's models and datasets
ai-eagle huggingface --org my-org --token hf_YOUR_TOKEN

# Scan specific user's spaces and datasets
ai-eagle huggingface --user someone --token hf_YOUR_TOKEN \
  --skip-all-models --include-discussions --include-prs

# Scan a dataset
ai-eagle huggingface --dataset username/dataset-name --token hf_YOUR_TOKEN

# Ignore specific resources
ai-eagle huggingface --org my-org --token hf_YOUR_TOKEN \
  --ignore-models "my-org/test-*" --skip-all-spaces
```

| Option | Type | Description |
|--------|------|-------------|
| `--endpoint` | string | HuggingFace endpoint (default: `https://huggingface.co`) |
| `--model` | string (repeatable) | Model to scan (`username/model`) |
| `--space` | string (repeatable) | Space to scan (`username/space`) |
| `--dataset` | string (repeatable) | Dataset to scan (`username/dataset`) |
| `--org` | string (repeatable) | Organization to scan |
| `--user` | string (repeatable) | User to scan |
| `--token` | string / `HUGGINGFACE_TOKEN` env | HuggingFace token |
| `--include-models` | string (repeatable) | Models to include (with `--org`/`--user`) |
| `--include-spaces` | string (repeatable) | Spaces to include |
| `--include-datasets` | string (repeatable) | Datasets to include |
| `--ignore-models` | string (repeatable) | Models to ignore |
| `--ignore-spaces` | string (repeatable) | Spaces to ignore |
| `--ignore-datasets` | string (repeatable) | Datasets to ignore |
| `--skip-all-models` | flag | Skip all model scans |
| `--skip-all-spaces` | flag | Skip all space scans |
| `--skip-all-datasets` | flag | Skip all dataset scans |
| `--include-discussions` | flag | Include discussions |
| `--include-prs` | flag | Include pull requests |

**Docker:**
```bash
docker run --rm -e HUGGINGFACE_TOKEN=hf_... \
  ai-eagle huggingface --model username/model-name
```

---

### `circleci` - Scan CircleCI

Scans CircleCI build configurations.

```bash
ai-eagle circleci --token YOUR_CIRCLECI_TOKEN
```

| Option | Type | Description |
|--------|------|-------------|
| `--token` | string (required) / `CIRCLECI_TOKEN` env | CircleCI token |

**Docker:**
```bash
docker run --rm -e CIRCLECI_TOKEN=... ai-eagle circleci --token $CIRCLECI_TOKEN
```

---

### `travisci` - Scan TravisCI

Scans TravisCI build logs.

```bash
ai-eagle travisci --token YOUR_TRAVIS_TOKEN
```

| Option | Type | Description |
|--------|------|-------------|
| `--token` | string (required) / `TRAVISCI_TOKEN` env | TravisCI token |

**Docker:**
```bash
docker run --rm -e TRAVIS_TOKEN=... ai-eagle travisci --token $TRAVIS_TOKEN
```

---

### `syslog` - Scan Syslog Streams

Listens on a network address and scans syslog messages in real time.

```bash
# Listen on UDP
ai-eagle syslog --address 0.0.0.0:5140 --protocol udp --format rfc5424

# Listen on TCP with TLS
ai-eagle syslog --address 0.0.0.0:6514 --protocol tcp \
  --cert /path/to/cert.pem --key /path/to/key.pem --format rfc5424
```

| Option | Type | Description |
|--------|------|-------------|
| `--address` | string | Address and port to listen on (e.g. `0.0.0.0:514`) |
| `--protocol` | string | Protocol: `udp` or `tcp` |
| `--cert` | string | Path to TLS certificate |
| `--key` | string | Path to TLS key |
| `--format` | string (required) | Log format: `rfc3164` or `rfc5424` |

**Docker:**
```bash
docker run --rm -p 5140:5140/udp ai-eagle syslog \
  --address 0.0.0.0:5140 --protocol udp --format rfc5424
```

---

### `stdin` - Scan Piped Data

Scans data piped from any source via stdin.

```bash
# Pipe a file
cat config.env | ai-eagle stdin

# Pipe git diff
git diff HEAD~5 | ai-eagle stdin --json

# Pipe from curl
curl -s https://pastebin.com/raw/abc123 | ai-eagle stdin

# Pipe from Docker logs
docker logs my-container | ai-eagle stdin --json --fail

# Pipe clipboard (macOS/Linux)
pbpaste | ai-eagle stdin
xclip -selection clipboard -o | ai-eagle stdin
```

| Option | Type | Description |
|--------|------|-------------|
| (none) | | Reads from standard input. All global options apply. |

**Docker:**
```bash
echo "sk_live_4eC39HqLyjWDarjtT1zdp7dc" | docker run --rm -i ai-eagle stdin
cat secrets.txt | docker run --rm -i ai-eagle stdin --json
```

---

### `multi-scan` - Scan Multiple Sources from Config

Scans multiple sources defined in a YAML configuration file.

```bash
ai-eagle multi-scan --config scan-config.yaml
```

| Option | Type | Description |
|--------|------|-------------|
| `--config` | string (required) | Path to YAML configuration file |

---

### `analyze` - Analyze API Key Permissions

Analyzes API keys for fine-grained permissions and scope information. Supports 40+ providers.

```bash
# Analyze a GitHub token
ai-eagle analyze --key-type github --key key=ghp_YOUR_TOKEN

# Analyze a Stripe key
ai-eagle analyze --key-type stripe --key key=sk_live_YOUR_KEY

# Analyze Twilio (requires multiple parts)
ai-eagle analyze --key-type twilio --key sid=AC123 --key key=auth_token

# Analyze Shopify (requires key + URL)
ai-eagle analyze --key-type shopify --key key=shpat_... --key url=mystore.myshopify.com
```

Supported key types: `github`, `sendgrid`, `openai`, `postgres`, `mysql`, `slack`, `twilio`, `airbrake`, `huggingface`, `stripe`, `gitlab`, `mailchimp`, `postman`, `bitbucket`, `asana`, `mailgun`, `square`, `sourcegraph`, `shopify`, `opsgenie`, `privatekey`, `notion`, `dockerhub`, `anthropic`, `digitalocean`, `elevenlabs`, `planetscale`, `airtableoauth`, `airtablepat`, `groq`, `launchdarkly`, `figma`, `plaid`, `netlify`, `fastly`, `monday`, `datadog`, `ngrok`, `mux`, `posthog`, `dropbox`, `databricks`, `jira`

---

### Global Options (apply to all scan commands)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-j, --json` | flag | | Output in JSON format (jq-compatible) |
| `--json-legacy` | flag | | Use pre-v3.0 JSON format |
| `--sarif` | flag | | Output in SARIF v2.1.0 (GitHub Code Scanning) |
| `--github-actions` | flag | | Output GitHub Actions annotations |
| `--html-report` | string | | Generate enterprise HTML report to path |
| `--excel-report` | string | | Generate Excel/CSV report to path |
| `--concurrency` | int | CPU count | Number of concurrent workers |
| `--no-verification` | flag | | Don't verify detected secrets |
| `--results` | string | | Filter: `verified`, `unknown`, `unverified`, `filtered_unverified` |
| `--filter-unverified` | flag | | Only first unverified result per chunk per detector |
| `--filter-entropy` | float | 0.0 | Filter unverified by Shannon entropy (try 3.0) |
| `--ml-threshold` | float | 0.5 | ML confidence threshold (0.0-1.0) |
| `--include-detectors` | string | `all` | Comma-separated detector types to include |
| `--exclude-detectors` | string | | Comma-separated detector types to exclude |
| `--detector-timeout` | float | | Max seconds per detector per chunk |
| `--config` | string | | Path to YAML configuration file |
| `--fail` | flag | | Exit code 183 if results are found (CI mode) |
| `--fail-on-scan-errors` | flag | | Exit with non-zero code on scan errors |
| `--no-color` | flag | | Disable colorized output |
| `--log-level` | int | 0 | Logging verbosity 0 (info) to 5 (trace), -1 to disable |
| `--scan-entire-chunk` | flag | | Bypass keyword pre-filter (slow, thorough) |
| `--allow-verification-overlap` | flag | | Cross-detector verification |
| `--print-avg-detector-time` | flag | | Print per-detector timing stats |
| `--force-skip-binaries` | flag | | Force skipping binary files |
| `--force-skip-archives` | flag | | Force skipping archive files |
| `--detector-worker-multiplier` | int | 8 | Multiplier for detector workers |
| `--notification-worker-multiplier` | int | 1 | Multiplier for notification workers |

---

## Output Formats

### Plain Text (default)
Human-readable terminal output with color-coded severity.

### JSON (`--json`)
Machine-readable JSON, one object per line. Compatible with `jq`.

### Legacy JSON (`--json-legacy`)
Pre-v3.0 JSON format for backward compatibility.

### SARIF (`--sarif`)
[SARIF v2.1.0](https://sarifweb.azurewebsites.net/) for GitHub Code Scanning, Azure DevOps, and other SARIF-compatible tools.

### HTML Report (`--html-report path.html`)
Enterprise-grade HTML report with:
- Executive summary dashboard
- Severity distribution charts (pure CSS, no JS)
- Compliance framework mapping (SOC 2, PCI DSS, HIPAA, GDPR, ISO 27001, NIST)
- Detailed findings table
- Remediation priority matrix
- Risk heat map

### Excel/CSV Report (`--excel-report path.csv`)
Spreadsheet export for audit teams and management reporting.

### GitHub Actions (`--github-actions`)
Annotations that appear inline in GitHub Actions pull request diffs.

---

## ML Scoring Pipeline

AI-EAGLE uses a multi-layer scoring system that goes far beyond simple regex matching:

```
Detected Secret
      │
      ├─── Entropy Analysis (35% weight)
      │    ├── Shannon entropy (information density)
      │    ├── Chi-squared test (uniformity of char distribution)
      │    ├── Serial correlation (sequential dependency)
      │    └── Charset diversity (char class distribution profiling)
      │
      ├─── Structural Matching (30% weight)
      │    ├── Known credential format patterns (AKIA..., ghp_..., sk_live_...)
      │    ├── Format fingerprinting against 25+ credential templates
      │    └── L3 prefix cache for instant known-format lookups
      │
      ├─── Context Analysis (20% weight)
      │    ├── Positive signals: env vars, config assignments, auth headers
      │    └── Negative signals: example/placeholder, template vars, docs
      │
      └─── False Positive Risk (15% weight)
           ├── Common word detection
           ├── Template variable patterns
           └── Documentation context
```

### Adaptive ML Ensemble

The ML classifier extracts 33 numerical features from each candidate and runs them through 5 classifiers:

1. **Adaptive Naive Bayes** - Character n-gram learning
2. **Online Logistic Regression** - Stochastic gradient descent
3. **Decision Stump Ensemble** - Adaptive threshold tuning
4. **Anomaly Detector** - Running statistics via Welford's algorithm
5. **Structural Fingerprint Matcher** - Dynamic pattern registration

All classifiers use online learning and adapt from verification feedback during scans. The ensemble weights themselves rebalance based on per-classifier accuracy tracking.

### Severity Classification

| Severity | Confidence | Action |
|----------|-----------|--------|
| CRITICAL | Verified active | ROTATE IMMEDIATELY |
| HIGH | >= 0.85 | Investigate and rotate |
| MEDIUM | >= 0.65 | Review context, verify manually |
| LOW | >= 0.40 | May be a false positive |
| INFO | < 0.40 | Likely a false positive |

---

## Configuration

### CLI Flags

See [Global Options](#global-options-apply-to-all-scan-commands) for the full reference table. Key flags:

```
Output:
  -j, --json                   Output in JSON format (jq-compatible)
  --json-legacy                Use pre-v3.0 JSON format
  --sarif                      Output in SARIF v2.1.0 format (GitHub Code Scanning)
  --github-actions             Output GitHub Actions annotations
  --html-report PATH           Generate enterprise HTML report
  --excel-report PATH          Generate Excel/CSV report

Filtering:
  --results TYPE               Filter: verified, unknown, unverified, filtered_unverified
  --ml-threshold FLOAT         ML confidence threshold 0.0-1.0 (default: 0.5)
  --filter-entropy FLOAT       Filter unverified results by Shannon entropy (try 3.0)
  --filter-unverified          Only first unverified result per chunk per detector
  --include-detectors LIST     Comma-separated detector types to include
  --exclude-detectors LIST     Comma-separated detector types to exclude

Verification:
  --no-verification            Don't verify detected secrets against live APIs
  --allow-verification-overlap Cross-detector verification

Performance:
  --concurrency INT            Number of concurrent workers (default: CPU count)
  --detector-timeout FLOAT     Max seconds per detector per chunk
  --detector-worker-multiplier Multiplier for detector workers (default: 8x)
  --force-skip-binaries        Skip binary files
  --force-skip-archives        Skip archive files

CI/CD:
  --fail                       Exit code 183 if results are found
  --fail-on-scan-errors        Exit with non-zero on scan errors
  --no-color                   Disable colorized output
  --config PATH                Path to YAML configuration file
  --log-level INT              Logging verbosity 0 (info) to 5 (trace), -1 to disable
```

### Config File (YAML)

```yaml
detectors:
  - name: CustomAPIKey
    keywords: ["myapi"]
    regex:
      value: "myapi_[A-Za-z0-9]{32}"
    verify:
      - endpoint: "https://api.example.com/verify"
        unsafe: false
        headers:
          Authorization: "Bearer {secret}"

sources:
  - type: git
    uri: "https://github.com/org/repo.git"
```

---

## CLI Reference

```
ai-eagle [global options] <command> [command options]

Commands:
  git                  Scan a Git repository
  github               Scan GitHub organizations/repos
  github-experimental  Scan GitHub with object discovery (alpha)
  gitlab               Scan GitLab groups/repos
  filesystem           Scan local files and directories
  s3                   Scan Amazon S3 buckets
  gcs                  Scan Google Cloud Storage buckets
  docker               Scan Docker image layers
  elasticsearch        Scan Elasticsearch indices
  postman              Scan Postman workspaces/collections
  jenkins              Scan Jenkins build artifacts
  huggingface          Scan HuggingFace models/datasets/spaces
  circleci             Scan CircleCI build configs
  travisci             Scan TravisCI build logs
  syslog               Listen and scan syslog streams
  stdin                Scan data piped from stdin
  multi-scan           Scan multiple sources from config file
  analyze              Analyze API key permissions (40+ providers)
```

See [Scan Commands - Full Reference](#scan-commands---full-reference) for detailed options per command.

---

## Project Structure

```
ai_eagle/
├── cli.py                     # Click CLI entry point (15 subcommands)
├── version.py                 # Version info (3.0.0-dev)
├── __main__.py                # python -m ai_eagle support
│
├── engine/                    # Core scanning engine
│   ├── engine.py              # 4-stage threaded pipeline (1,692 lines)
│   ├── ahocorasick_core.py    # Aho-Corasick keyword pre-filter
│   ├── defaults.py            # Default detector registration
│   └── dispatcher.py          # Result routing (PrinterDispatcher, MultiPrinter)
│
├── detectors/                 # 887+ secret detectors
│   ├── base.py                # Detector base class & interfaces
│   ├── result.py              # Result, ResultWithMetadata, clean_results
│   ├── false_positives.py     # False positive detection logic
│   ├── abstract/              # Abstract detector (catches generic patterns)
│   ├── abuseipdb/             # AbuseIPDB API key detector
│   ├── ...                    # 900+ detector modules (one per service)
│   └── alchemy/               # Alchemy API key detector
│
├── analysis/                  # ML & scoring pipeline
│   ├── ml_classifier.py       # 5-classifier adaptive ensemble (online learning)
│   ├── entropy.py             # Shannon + chi-squared + serial correlation
│   ├── scoring.py             # Multi-signal confidence scoring
│   ├── risk.py                # Risk categorization & remediation priority
│   ├── context.py             # Contextual validation (proximity signals)
│   ├── cache.py               # 3-layer cache (LRU + hash + prefix)
│   └── statistics.py          # Statistical utilities
│
├── decoders/                  # Data format decoders
│   ├── base.py                # Decoder interface & DecodableChunk
│   ├── utf8.py                # UTF-8 pass-through decoder
│   ├── utf16.py               # UTF-16 LE/BE auto-detection
│   ├── base64_decoder.py      # Base64 detection & decoding
│   └── escaped_unicode.py     # Escaped unicode sequence handling
│
├── sources/                   # Data source connectors
│   ├── base.py                # Source interface & Chunk model
│   ├── manager.py             # SourceManager (thread-safe chunk queue)
│   ├── git/                   # Git repository scanner
│   ├── github/                # GitHub API connector
│   ├── gitlab/                # GitLab API connector
│   ├── filesystem/            # Local filesystem scanner (ThreadPoolExecutor)
│   ├── s3/                    # AWS S3 bucket scanner
│   ├── gcs/                   # Google Cloud Storage scanner
│   ├── docker/                # Docker image layer scanner
│   ├── elasticsearch/         # Elasticsearch index scanner
│   ├── postman/               # Postman workspace scanner
│   ├── jenkins/               # Jenkins build scanner
│   ├── huggingface/           # HuggingFace model scanner
│   ├── circleci/              # CircleCI config scanner
│   ├── travisci/              # TravisCI log scanner
│   ├── syslog/                # Syslog stream listener
│   └── stdin/                 # Stdin pipe reader
│
├── output/                    # Output formatters
│   ├── plain.py               # Color terminal output (with safe print for Windows)
│   ├── json_printer.py        # JSON & Legacy JSON formatters
│   ├── sarif.py               # SARIF v2.1.0 formatter
│   ├── html_report.py         # Enterprise HTML report (compliance mapping)
│   ├── excel_report.py        # Excel/CSV report generator
│   ├── github_actions.py      # GitHub Actions annotation formatter
│   └── progress.py            # Rich live progress display
│
├── common/                    # Shared utilities
│   ├── filter.py              # Path include/exclude filtering
│   ├── http_client.py         # HTTP client with retry logic
│   ├── patterns.py            # Shared regex patterns
│   ├── recover.py             # Panic recovery (Go-style)
│   ├── secrets.py             # Secret masking utilities
│   ├── utils.py               # Random IDs, helpers
│   └── vars.py                # Environment variable helpers
│
├── config/                    # Configuration file parsing
├── context/                   # Cancellation context (Go-style ctx)
├── handlers/                  # File handlers (ZIP/TAR archive support)
├── gitparse/                  # Git diff parser
├── giturl/                    # Git URL parser
├── log/                       # Structured logging
├── models/                    # Protobuf-style data models
├── sanitizer/                 # Output sanitization
└── custom_detectors/          # User-defined detector support

tests/
├── test_ahocorasick.py        # Aho-Corasick engine tests
├── test_analysis.py           # ML & scoring tests
├── test_cache.py              # Multi-layer cache tests
├── test_cli_setup.py          # CLI integration tests
├── test_config.py             # Config parsing tests
├── test_context.py            # Context/cancellation tests
├── test_decoders.py           # Decoder chain tests
├── test_detectors.py          # Detector tests
├── test_engine.py             # Engine pipeline tests
├── test_pass3_regressions.py  # Bug regression tests (pass 3)
├── test_accuracy_benchmark.py # Accuracy benchmark (precision/recall/F1)
├── test_runtime_hardening.py  # Runtime safety tests
└── test_sources_cancellation.py # Source cancellation tests

Dockerfile                     # Multi-stage Docker build (python:3.13-slim)
pyproject.toml                 # Package config & build system
requirements.txt               # Python dependencies
```

---

## Development

### Prerequisites

- Python 3.11+
- pip or uv

### Setup

```bash
pip install -e ".[dev,fast]"
```

### Run Tests

```bash
# Run all 264 tests
pytest tests/ -v

# Run accuracy benchmarks only
pytest tests/test_accuracy_benchmark.py -v
```

### Type Checking

```bash
mypy ai_eagle/
```

### Linting

```bash
ruff check ai_eagle/
```

### Build Docker Image

```bash
docker build -t ai-eagle .
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `click>=8.1` | CLI framework |
| `pydantic>=2.0` | Data validation & models |
| `requests>=2.31` | HTTP client for verification |
| `pyyaml>=6.0` | YAML config parsing |
| `colorama>=0.4` | Windows terminal colors |
| `cachetools>=5.3` | Additional caching utilities |
| `rich>=13.0` | Progress bars & rich terminal output |
| `ahocorasick-rs>=0.22` | Optional: Fast Aho-Corasick (Rust) |

---

## License

AGPL-3.0
