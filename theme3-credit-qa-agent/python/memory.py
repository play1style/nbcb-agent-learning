"""
对话记忆模块。

保存用户问题和 Agent 回答，为上下文理解提供支持。
"""


class Memory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content):
        """记录用户问题。"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        """记录 Agent 回答。"""
        self.messages.append({"role": "assistant", "content": content})

    def get_user_history(self):
        """获取所有用户问题的拼接文本。"""
        return " ".join([m["content"] for m in self.messages if m["role"] == "user"])

    def get_messages(self):
        """返回所有对话记录。"""
        return self.messages
