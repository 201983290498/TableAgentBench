# Dataset Documentation

English | [中文](README_zh.md)

This directory contains the test datasets and related sample files for the TableAgent project.

## 1. `samples.json` Field Description

The `samples.json` file defines the sample data for testing, with each sample containing a specific table QA task. The main fields are as follows:

*   **`file_path`**: Path to the folder where the table data is located.
    *   This path points to the directory containing the table files required for the task.
*   **`task`**: User's natural language query or task description.
*   **`design`**: The design solution for solving the task, including step-by-step execution logic.
    *   **`type`**: Task type (e.g., `unknown` or `Fork-Join`, etc.).
    *   **`checkout_list`**: List of specific steps to solve the task.
        *   **`idx`**: Step sequence number (starting from 0).
        *   **`info_item`**: Specific instruction or information to be obtained in this step.
        *   **`related_tables`**: List of related table filenames needed to complete this step.
        *   **`score_points`**: Expected key results or scoring points for this step (used to verify if the Agent's answer is correct).

## 2. Table Data Organization (`tables` directory)

The `dataset/tables` directory stores all original table files, organized by category or source.

*   **`tables/`**
    *   **`chinese_table/`**: Stores Chinese table datasets.
        *   Contains multiple subfolders named by **specific themes or years** (e.g., `2007 Land Transaction and Engineering Supervision Qualification`, `2008 Academic Ranking of World Universities`, etc.).
        *   Each subfolder contains one or more associated data files, usually in `.xlsx`, `.xls`, or `.csv` format.

### Example Structure

```text
dataset/
├── samples.json          # Sample task description file
└── tables/
    └── chinese_table/    # Chinese table collection
        ├── 2007 Land Transaction and Engineering Supervision Qualification/
        │   ├── 1706800061649.xlsx
        │   └── Engineering Supervision.xlsx
        ├── [Shop Data] Flagship Store Operation Report Dec-/
        │   ├── [Shop Data] Flagship Store Operation Report Dec-.xlsx
        │   ├── [Shop Data] Flagship Store Operation Report Dec-_CS.csv
        │   └── ...
        └── ...
```
