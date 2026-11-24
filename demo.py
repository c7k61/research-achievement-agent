"""Run the complete offline demonstration from any working directory."""

from pathlib import Path

from research_agent.__main__ import write_json
from research_agent.agent import DemoModel, run_agent
from research_agent.store import Store


def main():
    root = Path(__file__).resolve().parent
    store = Store(root / "data" / "demo.sqlite3")
    try:
        store.ingest(root / "examples" / "materials")
        result = run_agent(store, "请核验论文和专利的当前状态，能写已发表和已授权吗", DemoModel())
        output = root / "runs" / "demo.json"
        write_json(output, result)
        print("运行方式 offline-demo\n")
        print(result["answer"])
        print(f"\n执行轨迹已保存到 {output}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
