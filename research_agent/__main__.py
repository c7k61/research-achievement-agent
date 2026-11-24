from pathlib import Path
import argparse
import json
import os
import sys
import tempfile

from .agent import DemoModel, OllamaModel, run_agent
from .store import Store
from .diagnostics import check_environment


def write_json(path, result):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="科研成果核验 Agent")
    parser.add_argument("--db", default="data/index.sqlite3")
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest = subcommands.add_parser("ingest", help="导入目录，原子替换当前材料集")
    ingest.add_argument("directory")
    ask = subcommands.add_parser("ask", help="检索、读取并核验材料")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["demo", "ollama"], default="demo")
    ask.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3"))
    ask.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ask.add_argument("--output", default="runs/latest.json")
    subcommands.add_parser("list", help="列出材料与摘要指纹")
    doctor = subcommands.add_parser("doctor", help="检查文档依赖和本机 Ollama 服务")
    doctor.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3"))
    doctor.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    args = parser.parse_args()
    if args.command == "doctor":
        try:
            result = check_environment(args.base_url, args.model)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["model_available"] else 1
        except ValueError as error:
            print(f"检查失败　{error}", file=sys.stderr)
            return 1
    store = Store(args.db)
    try:
        if args.command == "ingest":
            print(json.dumps(store.ingest(args.directory), ensure_ascii=False, indent=2))
        elif args.command == "list":
            print(json.dumps(store.list_sources(), ensure_ascii=False, indent=2))
        else:
            model = DemoModel() if args.mode == "demo" else OllamaModel(args.model, args.base_url)
            result = run_agent(store, args.question, model)
            write_json(args.output, result)
            print(f"运行方式 {result['mode']}\n")
            print(result["answer"])
            print(f"\n工具执行记录已保存到 {args.output}")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"执行失败　{error}", file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
