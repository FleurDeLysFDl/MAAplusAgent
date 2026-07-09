# 新游戏接入checklist

1. 复制这个目录（`games/_template/`），改名为新游戏名字，例如`games/新游戏/`
2. 把`profile.yaml.example`改名为`profile.yaml`并填写
3. 有现成MaaFramework资源包（类似[MBCCtools](https://github.com/quietlysnow/MBCCtools)）就放进`resource/`目录，并把对应的`interface.json`放进本目录；
   没有的话`resource_path`对应那一项留空字符串，纯自由探索也能跑，只是没有`run_known_task`可用。
   **注意**：只放游戏专属的`pipeline/`+`image/`，不要把OCR模型也拷进来——OCR模型是通用资源，
   同语言的游戏共用`assets/ocr_models/<language>/`一份就够了，没有就照`assets/ocr_models/zh_cn/`的样子建一份
4. 按需创建`sensitive_keywords.yaml`，补充本游戏专属敏感词
5. 启动MCP Server验证能连上设备：
   ```
   python core/maa_mcp_server.py games/新游戏/profile.yaml
   ```
6. 用`core/agent_runner.py`连这个profile开始探索

**判断标准**：接入过程中`core/`目录一行代码都不改，才算真正通用；如果需要改核心代码，
说明某个游戏相关的东西之前被错误地写死在核心层了。
