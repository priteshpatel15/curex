"""Smoke tests for the sandboxed executor. No network or LLM calls."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.executor import SafeCodeExecutor


def make_executor():
    return SafeCodeExecutor(pd.DataFrame({"region": ["a", "b", "a"], "sales": [1, 2, 3]}))


def test_runs_pandas_code_and_captures_stdout():
    result = make_executor().execute_code("print(df.groupby('region')['sales'].sum().to_dict())")
    assert result.success, result.error
    assert "{'a': 4, 'b': 2}" in result.output


def test_rejects_imports_and_dangerous_calls():
    ex = make_executor()
    for code in ["import os", "open('x')", "eval('1')", "os.system('ls')"]:
        result = ex.execute_code(code)
        assert not result.success
        assert "validation failed" in result.error


def test_saves_plot_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = make_executor().execute_code("plt.plot(df['sales'])")
    assert result.success, result.error
    assert result.plots_generated == ["plot_0.png"]
    assert os.path.exists(tmp_path / "plot_0.png")
