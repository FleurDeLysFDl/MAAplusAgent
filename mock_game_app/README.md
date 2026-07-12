# MockGame — 探索/导航流程测试用安卓app

给 `core/` 那套探索Agent（screenshot/OCR/节点匹配/BFS导航）当测试对象用的假游戏，
不是真的游戏。内容故意贴近 `skills/mobile-game-ui-nav/SKILL.md` 总结的常见二次元/
养成类手游UI模式，但用清晰、稳定、可预期的中文文字，不是真实游戏里那种花体banner/
滚动内容/OCR噪声——目的是让"探索流程本身对不对"和"真实游戏OCR太脏"这两件事能分开
验证，出问题时能确定是流程的锅还是游戏内容的锅。

## 覆盖的场景（对应SKILL.md里的模式）

- **主城图标墙**：`HubScreen`，一排功能入口（任务/编队/角色/商城/公告）+ 一个纯图标
  设置按钮（没有文字标签，测试`click_on_image`兜底路径）
- **登录弹窗诱导跳转**：进主城弹一个"登录奖励"弹窗，除了"确定"还有个"前往商城"按钮
  ——测试"弹窗优先关掉，别跟着跳转按钮走"这条经验；系统返回键也能关
- **顶部tab+左侧列表+右侧详情**：`AnnouncementsScreen`，公告页三个tab、点列表项换
  右侧详情；其中一条公告标题特意包含"充值"敏感词但是官方公告语境（误报场景）
- **角色列表→详情多级tab**：`CharactersScreen` → `CharacterDetailScreen`，详情页
  内有信息/技能/培养/烙印/档案五个二级tab，切tab是同节点内的状态变化
- **兑换列表**：`ShopScreen`，图标+数量+兑换按钮，点兑换只减库存（同节点状态变化，
  不是新界面）
- **任务进度**：`TasksScreen`，"(0/1)"进度文案+状态相关按钮（前往/领取/已完成），
  "前往"才会跳转到`TaskDetailScreen`，"领取"是同节点状态变化
- **编队多选**：`TeamScreen`，队伍-01~07列表选中态是同节点变化，不是新界面

## 构建（这次没有实际编译/运行，需要的时候按下面步骤）

这个仓库里只有源码，**没有`gradle-wrapper.jar`**（二进制文件不方便在这里生成）。
拿到一台装了Android Studio或本地Gradle的机器后：

```bash
cd mock_game_app
gradle wrapper --gradle-version 8.7   # 生成 gradlew / gradlew.bat / wrapper jar
./gradlew assembleDebug                # 或者直接用Android Studio打开这个目录构建
```

或者更省事：用Android Studio「Open」直接打开`mock_game_app/`目录，让它自动补全
wrapper再Run。

产物包名`com.maaplus.mockgame`，分辨率跟着模拟器走（Activity声明了强制横屏，跟
`games/无期迷途/profile.yaml`里的`resolution: [1280, 720]`一致，方便复用同一套
adb分辨率假设）。

## 接入core/框架

按 `games/_template/` 的套路新建 `games/mockgame/profile.yaml`：
- `package_name: com.maaplus.mockgame`
- `resolution: [1280, 720]`
- `ocr_model`: 复用`assets/ocr_models/zh_cn`即可（界面全是简体中文）
- 不需要`resource/base`这类MaaFramework pipeline资源包——这个测试app没有确定性
  任务(`run_known_task`)可复用，纯粹靠探索/导航验证

APK装上模拟器（`adb install`）之后，理论上直接就能用现有的`agent_runner.py`/
`navigate_demo.py`跑，不需要改`core/`下任何代码——这也是这个测试app存在的意义：
如果连这个内容干净、没有OCR噪声的假游戏都探索/导航不顺，那问题基本可以排除是
"游戏内容太脏"，锁定在流程本身。
