# NanoTorch 🔬

> A lightweight ML tensor framework built **from scratch** in pure Python + NumPy,
> developed strictly following **Test-Driven Development (TDD)**.

[![Tests](https://github.com/your-org/nanotorch/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/nanotorch/actions)
[![Coverage](https://img.shields.io/codecov/c/github/your-org/nanotorch)](https://codecov.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why?

NanoTorch is an educational project that mirrors PyTorch's API surface while
exposing every mathematical detail through clean, heavily-documented Python code.
Each feature is introduced on a dedicated day, accompanied by gradient-checked
tests that compare results against PyTorch.

## Roadmap

| Stage | Days | Topics |
|-------|------|--------|
| 1 | 1–15 | Scalar computation graph & autograd |
| 2 | 16–35 | Multi-dimensional tensors & broadcasting |
| 3 | 36–50 | Neural-network layers |
| 4 | 51–60 | Loss functions & optimizers |
| 5 | 61–70 | Training pipeline, benchmarks & release |

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

```
nanotorch/
├── src/
│   └── nanotorch/
│       ├── __init__.py
│       └── scalar/       # Days 2–15
│           └── __init__.py
├── tests/
│   ├── conftest.py
│   └── test_package.py   # smoke tests
├── pyproject.toml
└── .gitignore
```

*More to come — stay tuned for Day 2!*
