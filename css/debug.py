from __future__ import annotations

import os


def save_memory_to_txt(buffer, filename="memory_buffer.txt"):
    """
    將 replay buffer 輸出成 .txt 檔案
    :param buffer: 例如 agent.memory.buffer (list / deque / array)
    :param filename: 儲存檔案名稱
    """
    import os

    # 預設存到當前資料夾
    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for i, item in enumerate(buffer):
            f.write(f"{i}: {str(item)}\n")

    print(f"[INFO] buffer 已儲存到 {filepath}")
    return filepath

