"""Read-only model-service diagnostics without running inference."""

import importlib.util
import json
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

from .agent import OllamaModel


def check_environment(base_url, model_name):
    client = OllamaModel(model_name, base_url)
    result = {
        "python": sys.version.split()[0],
        "document_parsers": {name: importlib.util.find_spec(name) is not None
                             for name in ("pypdf", "docx")},
        "model": model_name,
        "service_reachable": False,
        "model_available": False,
        "installed_models": [],
    }
    request = Request(client.url.removesuffix("/chat") + "/tags")
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.load(response)
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise ValueError("模型列表响应缺少 models 数组")
        names = [item["name"] for item in models
                 if isinstance(item, dict) and isinstance(item.get("name"), str)]
        result["service_reachable"] = True
        result["installed_models"] = names
        result["model_available"] = model_name in names or model_name + ":latest" in names
        result["next_step"] = ("可运行 ask --mode ollama 进行真实模型联调"
                               if result["model_available"] else "请安装所需模型或指定已安装模型名称")
    except (OSError, URLError, ValueError) as error:
        result["next_step"] = "请启动 Ollama 并检查服务地址后重试"
        result["error"] = type(error).__name__
    return result
