"""
LLM 客户端模块（主题2 & 主题3 通用）。

功能：
- 使用 Python 标准库调用 OpenAI 兼容格式的 LLM API
- 支持真实调用与 Mock 自动降级两种模式
- 通过环境变量控制，无需修改代码即可切换

环境变量（均为可选）：
  LLM_API_KEY    – API 密钥，未设置时自动进入 Mock 模式
  LLM_BASE_URL   – API 基地址，默认 https://api.moonshot.cn/v1
  LLM_MODEL      – 模型名称，默认 moonshot-v1-8k
  LLM_TIMEOUT    – 请求超时秒数，默认 10
  LLM_MOCK       – mock 控制：auto（默认）| true | false
                   auto = 没有 API Key 时自动 mock

使用示例：
    client = LLMClient()
    reply = client.chat("你是银行信贷专家", "请分析以下企业...")
    if reply is None:
        # 降级到模板/规则逻辑
        ...
"""

import json
import os
import socket
import urllib.error
import urllib.request


class LLMClient:
    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY", "").strip()
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.moonshot.cn/v1").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "moonshot-v1-8k")
        self.timeout = int(os.environ.get("LLM_TIMEOUT", "10"))
        self.mock_mode = os.environ.get("LLM_MOCK", "auto").lower()

    def is_mock(self):
        """判断是否使用 Mock 模式。"""
        if self.mock_mode == "true":
            return True
        if self.mock_mode == "false":
            return False
        # auto：没有 API Key 时自动 mock
        return not self.api_key

    def chat(self, system_prompt, user_prompt, max_tokens=512):
        """
        调用 LLM 生成文本。

        参数：
            system_prompt – 系统提示词
            user_prompt   – 用户提示词
            max_tokens    – 最大生成 token 数

        返回：
            生成文本（str），若处于 Mock 模式或调用失败则返回 None
        """
        if self.is_mock():
            return None

        try:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            print(f"[LLM 调用失败，自动降级到模板模式] HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            print(f"[LLM 调用失败，自动降级到模板模式] 网络错误: {e.reason}")
        except socket.timeout:
            print("[LLM 调用失败，自动降级到模板模式] 请求超时")
        except Exception as e:
            print(f"[LLM 调用失败，自动降级到模板模式] 错误: {e}")

        return None
