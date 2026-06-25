# CodeLaunguageClassifier

A machine learning pipeline that classifies source code snippets into their programming language. Built with Apple's Create ML and Core ML.

**23 languages supported** · ~1.1M total samples · Trained with pure Python (no ML dependencies) + Create ML

## Overview

This project collects real-world source code from 80+ popular open-source repositories, extracts function bodies, class bodies, and random line blocks, strips comments, and trains a text classifier that can identify the programming language of a code snippet.

The pipeline produces a **Core ML model** (`.mlmodel`) ready for integration into iOS, macOS, watchOS, or tvOS apps — no server round trip needed.

## Supported Languages

| Language | Samples | Language | Samples |
|----------|---------|----------|---------|
| **C** | 50,080 | **C++** | 50,098 |
| **C#** | 50,015 | **CSS** | ✓ |
| **Dart** | ✓ | **Go** | 50,006 |
| **HTML** | ✓ | **Java** | 50,003 |
| **JavaScript** | 50,007 | **JSON** | ✓ |
| **Kotlin** | 50,006 | **Lua** | ✓ |
| **Objective-C** | 50,008 | **PHP** | 50,009 |
| **Python** | 50,000 | **Ruby** | 50,021 |
| **Rust** | 50,010 | **Shell** | 30,998 |
| **SQL** | ✓ | **Swift** | 50,001 |
| **TypeScript** | 50,011 | **XML** | ✓ |
| **YAML** | ✓ | | |

> **Note:** Languages marked with ✓ (CSS, Dart, HTML, JSON, Lua, SQL, XML, YAML) were added beyond the original 15-language plan. Shell is capped at ~31k samples due to limited open-source shell script availability.

## Project Structure

```
CodeLaunguageClassifier/
├── CodeLaunguageClassifier.mlproj/   # Create ML project (Xcode)
│   ├── Project.json                   # Project configuration
│   ├── Data Sources/                  # Linked CSV data sources
│   ├── Model Containers/              # Training session tracking
│   └── Models/Source1.mlmodel        # Trained Core ML model (3.3 MB)
├── data/                              # Generated datasets (~617 MB)
│   ├── train.jsonl                    # Training set (~878k samples)
│   ├── val.jsonl                      # Validation set (~108k samples)
│   ├── test.jsonl                     # Test set (~116k samples)
│   ├── train/val/test_create_ml.csv   # Create ML compatible CSV
├── scripts/                           # Data processing pipeline
│   ├── download_repos.py              # Clone 80+ repos from GitHub for samples
│   ├── prepare_data.py                # Main pipeline: scan → extract → clean → JSONL
│   ├── verify_data.py                 # Validate JSONL integrity & distributions
│   ├── export_csv.py                  # JSONL → generic CSV
│   ├── export_create_ml_csv.py        # JSONL → Create ML CSV
│   ├── export_json.py                 # JSONL → JSON array
│   ├── export_test_csv.py             # test.jsonl → Create ML CSV
│   ├── downsample_train.py            # Balance classes via downsampling
│   └── extract_swiftui_samples.py     # Extra Swift samples from SwiftUI projects
├── repos/                             # Cloned source repositories (~3.3 GB)
│   ├── linguist/                      # GitHub Linguist samples
│   ├── Swift/  C/  C++/  Python/ …   # Open-source repos per language
├── findings.md                        # Research notes & technical decisions
├── task_plan.md                       # Task plan & completion tracking
└── progress.md                        # Session progress log
```

## Pipeline

```
Cloned repos (.swift, .py, .java, ...)
        │
        ▼
  1. Scan & filter by extension
  2. Extract samples:
     ├── Function bodies  (pattern-matched per language)
     ├── Class/struct bodies
     └── Random line groups & file truncation
  3. Strip comments (line & block)
  4. Cap at 5,000 chars per sample
  5. File-level 8:1:1 train/val/test split
        │
        ▼
    data/{train,val,test}.jsonl
        │
        ▼
    export_create_ml_csv.py
        │
        ▼
    Create ML (Xcode) → Core ML model
```

### Key Design Decisions

- **Zero external Python dependencies** — scripts use only the Python standard library (`json`, `csv`, `re`, `pathlib`, etc.)
- **Function extraction + file truncation** dual strategy maximizes sample quantity and covers both short and long code fragments
- **Comment stripping** removes single-line and block comments so the model learns syntax patterns, not natural language
- **File-level hashing** prevents duplicate samples across cloned repositories
- **No path/filename leakage** — samples contain only source code, never file metadata

## Getting Started

### Prerequisites

- **macOS** (for Create ML / Core ML training)
- **Xcode** (to open the `.mlproj` project)
- **Python 3.14+** (for data processing scripts)
- **Git** (for cloning source repositories)

### Downloading Source Code

The model is trained on real-world code from 80+ well-known open-source projects.

```bash
# Download all repositories (this will take a while — ~3.3 GB total)
python3 scripts/download_repos.py

# Download only specific languages
python3 scripts/download_repos.py --lang Swift
python3 scripts/download_repos.py --lang Python --lang Go

# Download with more parallel workers (default: 4)
python3 scripts/download_repos.py -j 8

# Skip GitHub Linguist (already have it or want to save bandwidth)
python3 scripts/download_repos.py --skip-linguist

# Full clone (with git history) instead of shallow depth=1
python3 scripts/download_repos.py --full
```

The script skips any repository that has already been cloned, so you can safely re-run it to resume interrupted downloads.

### Data Preparation

```bash
# Process all 23 languages (up to 50k samples per language)
python3 scripts/prepare_data.py

# Process only one language
python3 scripts/prepare_data.py --lang Swift

# Specify output directory
python3 scripts/prepare_data.py --output /path/to/data

# Limit samples per language
python3 scripts/prepare_data.py -n 20000
```

### Verification & Export

```bash
# Validate JSONL format, labels, and distributions
python3 scripts/verify_data.py

# Export to Create ML CSV (for training in Xcode)
python3 scripts/export_create_ml_csv.py

# Downsample training set for balanced class distribution
python3 scripts/downsample_train.py -n 20000
```

### Training the Model

1. Open `CodeLaunguageClassifier.mlproj` in Xcode
2. The data sources (`train_create_ml.csv`, `val_create_ml.csv`, `test_create_ml.csv`) are already linked
3. Select the **Text Classifier** template
4. Start training in Create ML
5. The trained `.mlmodel` is exported to `Models/Source1.mlmodel`

### Using the Model

Drag the `.mlmodel` file into your Xcode project and use Core ML's `NLModel` class:

```swift
import CoreML
import NaturalLanguage

guard let model = try? NLModel(mlModel: Source1().model) else { return }
let language = model.predictedLabel(for: "func hello() -> String { return \"world\" }")
// → "Swift"
```

## Script Reference

| Script | Purpose |
|--------|---------|
| `download_repos.py` | Clones all 80+ open-source repositories from GitHub for code samples |
| `prepare_data.py` | Main pipeline — scans repos, extracts samples, strips comments, outputs JSONL |
| `verify_data.py` | Checks JSONL format integrity, label distribution, path leakage, length stats |
| `export_csv.py` | Converts JSONL to generic CSV (`\n` encoded inside text) |
| `export_create_ml_csv.py` | Converts JSONL to Create ML CSV (newlines → spaces, escapes cleaned) |
| `export_json.py` | Converts JSONL to a JSON array |
| `export_test_csv.py` | Converts only `test.jsonl` to Create ML CSV |
| `downsample_train.py` | Downsamples training set to N samples per language for balanced classes |
| `extract_swiftui_samples.py` | Extracts extra Swift samples from SwiftUI projects |

## Source Repositories

Code samples are collected from **80+ well-known open-source projects** including:

- **Swift:** apple/swift, Alamofire, ReactiveX/RxSwift
- **Python:** python/cpython, django/django, psf/requests
- **Java:** spring-projects/spring-framework, elastic/elasticsearch
- **Go:** golang/go, grpc-go, kubernetes/kubernetes
- **Rust:** rust-lang/rust, tokio-rs/tokio
- **C++:** llvm/llvm-project, protocolbuffers/protobuf
- **C#:** dotnet/runtime, dotnet/aspnetcore
- **JavaScript/TypeScript:** expressjs/express, vuejs/vue, microsoft/TypeScript
- **C:** git/git, redis/redis
- **And more:** rails/ruby-on-rails, JetBrains/kotlin, ohmyzsh/ohmyzsh, laravel/laravel

See `findings.md` for the complete list and sourcing strategy.

## Trained Model

Two model versions were trained:

| Version | Classes | Samples | Iterations | Size |
|---------|---------|---------|------------|------|
| v1 (15-class) | 15 | 582,730 | — | — |
| v2 (23-class, full) | 23 | 878,264 | — | — |
| **v3 (23-class, balanced)** | **23** | **480,000** | **10** | **3.3 MB** |

The final `Source1.mlmodel` uses the balanced 23-class dataset (20k samples per language for 23 languages), trained for 10 iterations in Create ML.

## License

This project is for educational purposes. The source code samples collected from open-source repositories retain their original licenses. The pipeline scripts are MIT-licensed (see `LICENSE` if present).

## See Also

- [GitHub Linguist](https://github.com/github/linguist) — language detection for GitHub
- [Create ML](https://developer.apple.com/machine-learning/create-ml/) — Apple's ML framework
- [Core ML](https://developer.apple.com/documentation/coreml) — on-device ML inference
- [JSON Lines](https://jsonlines.org/) — the dataset format
