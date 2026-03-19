# Ownership Classifier CLI

## Overview
Command-line interface for classifying ownership records using the rule-based + model-assisted engine.

Supports:
- Single record classification
- Optional debug tracing

---

## Usage

### 1. Single Owner Classification

Basic:
```
python cli.py "OWNER NAME"
```
With optional fields:
```
python cli.py "OWNER NAME" --own2 "SECONDARY NAME" --state "CO"
```
Example:
```
python cli.py "MOUNTAIN STATES FIELD SERVICES DEPARTMENT" --state "AL"
```
Output:
```
Result:31
```

###2. Debug Mode (Trace Logging)

Enable full rule trace:
```
python cli.py "OWNER NAME" --state "AL" --debug
```
Output:
```
Result:31

Trace log:
{'stage': '00_loaded', ...}
{'stage': '26_government_keyword_check', ...}
...
```
Use this to:

- understand rule hits

- debug misclassifications

- inspect stage flow

