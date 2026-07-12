package com.maaplus.mockgame

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExitToApp
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.navigation.NavHostController

private val HubEntries = listOf(
    "任务" to "tasks",
    "编队" to "team",
    "角色" to "characters",
    "商城" to "shop",
    "公告" to "announcements",
)

@Composable
fun HubScreen(nav: NavHostController) {
    Box(modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("测试游戏 · 主城", style = MaterialTheme.typography.headlineSmall)
                // 纯图标按钮，没有任何文字标签——用来测试OCR认不出图标类按钮、
                // 必须走get_raw_image+click_on_image这条路的场景
                IconButton(onClick = { nav.navigate("settings") }) {
                    Icon(Icons.Filled.Settings, contentDescription = "设置")
                }
            }
            Spacer(modifier = Modifier.height(24.dp))
            LazyVerticalGrid(
                columns = GridCells.Fixed(3),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                items(HubEntries) { (label, route) ->
                    Card(
                        modifier = Modifier
                            .size(120.dp)
                            .clickable { nav.navigate(route) },
                    ) {
                        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                            Text(label, style = MaterialTheme.typography.titleMedium)
                        }
                    }
                }
            }
        }

        if (MockData.showLoginRewardPopup.value) {
            LoginRewardPopup(
                onDismiss = { MockData.showLoginRewardPopup.value = false },
                onGoShop = {
                    MockData.showLoginRewardPopup.value = false
                    nav.navigate("shop")
                },
            )
        }
    }
}

/**
 * 登录奖励弹窗：除了"确定"收尾按钮，还带一个"前往商城"跳转按钮——
 * 用来测试skills/mobile-game-ui-nav里"弹窗优先关掉，不要跟着跳转按钮走"这条经验。
 * 系统返回键（onDismissRequest）也能关掉，对应press_back优先的操作规则。
 */
@Composable
private fun LoginRewardPopup(onDismiss: () -> Unit, onGoShop: () -> Unit) {
    Dialog(onDismissRequest = onDismiss) {
        Card(modifier = Modifier.padding(16.dp)) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text("登录奖励", style = MaterialTheme.typography.titleLarge)
                Spacer(modifier = Modifier.height(12.dp))
                Text("连续登录第3天，获得强化石×10")
                Spacer(modifier = Modifier.height(20.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = onDismiss) { Text("确定") }
                    Button(onClick = onGoShop, colors = ButtonDefaults.buttonColors()) {
                        Text("前往商城")
                    }
                }
            }
        }
    }
}

@Composable
fun SettingsScreen(nav: NavHostController) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("设置", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        var soundOn by rememberSaveable { mutableStateOf(true) }
        var musicOn by rememberSaveable { mutableStateOf(true) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("音效", modifier = Modifier.width(80.dp))
            Switch(checked = soundOn, onCheckedChange = { soundOn = it })
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("音乐", modifier = Modifier.width(80.dp))
            Switch(checked = musicOn, onCheckedChange = { musicOn = it })
        }
        Spacer(modifier = Modifier.height(24.dp))
        // 另一个纯图标按钮（登出箭头，没有文字），同样用来测试图标类按钮识别
        IconButton(onClick = { /* 仅作为图标按钮测试点，不做真的登出 */ }) {
            Icon(Icons.Filled.ExitToApp, contentDescription = "退出登录")
        }
    }
}

@Composable
fun AnnouncementsScreen(nav: NavHostController) {
    var selectedTab by rememberSaveable { mutableStateOf(MockData.announcementTabs.first()) }
    var selected by remember { mutableStateOf(MockData.announcements.first()) }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("公告", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(8.dp))
        Row {
            MockData.announcementTabs.forEach { tab ->
                Text(
                    tab,
                    modifier = Modifier
                        .padding(8.dp)
                        .clickable { selectedTab = tab },
                    style = if (tab == selectedTab) MaterialTheme.typography.titleMedium else MaterialTheme.typography.bodyMedium,
                )
            }
        }
        Divider()
        Row(modifier = Modifier.fillMaxSize()) {
            val filtered = MockData.announcements.filter { it.tab == selectedTab }
            LazyColumn(modifier = Modifier.width(220.dp)) {
                items(filtered) { item ->
                    Text(
                        item.title,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { selected = item }
                            .padding(12.dp),
                    )
                    Divider()
                }
            }
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(1.dp)
                    .background(MaterialTheme.colorScheme.outlineVariant),
            )
            Column(modifier = Modifier.weight(1f).padding(16.dp)) {
                Text(selected.title, style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text(selected.content)
            }
        }
    }
}

@Composable
fun CharactersScreen(nav: NavHostController) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("角色养成", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        LazyRow {
            items(MockData.characters) { name ->
                Card(
                    modifier = Modifier
                        .padding(8.dp)
                        .size(140.dp)
                        .clickable { nav.navigate("character/$name") },
                ) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(name, style = MaterialTheme.typography.titleMedium)
                            Text("LV.10")
                        }
                    }
                }
            }
        }
    }
}

private val CharacterSubTabs = listOf("信息", "技能", "培养", "烙印", "档案")

@Composable
fun CharacterDetailScreen(nav: NavHostController, name: String) {
    var subTab by rememberSaveable { mutableStateOf(CharacterSubTabs.first()) }

    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("$name · 角色详情", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(12.dp))
        Row {
            CharacterSubTabs.forEach { tab ->
                Text(
                    tab,
                    modifier = Modifier
                        .padding(8.dp)
                        .clickable { subTab = tab },
                    style = if (tab == subTab) MaterialTheme.typography.titleMedium else MaterialTheme.typography.bodyMedium,
                )
            }
        }
        Divider()
        Spacer(modifier = Modifier.height(12.dp))
        when (subTab) {
            "信息" -> {
                Text("等级：LV.10")
                Text("攻击：6.0%")
                Text("生命：10.0%")
            }
            "技能" -> {
                Text("普攻：造成100%攻击力伤害")
                Text("技能：冷却时间12秒")
            }
            "培养" -> {
                Text("突破进度：2/6")
                Text("经验：120/500")
            }
            "烙印" -> {
                Text("倾覆高塔-I（已生效）")
                Text("暴击率提升12%")
            }
            "档案" -> {
                Text("$name 的角色档案简介占位文字。")
            }
        }
    }
}

@Composable
fun ShopScreen(nav: NavHostController) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("兑换中心", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        LazyColumn {
            items(MockData.shopItems) { item ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text("${item.name} × ${item.count}")
                    Button(
                        enabled = item.count > 0,
                        onClick = { if (item.count > 0) item.count -= 1 },
                    ) {
                        Text("兑换（消耗${item.cost}）")
                    }
                }
                Divider()
            }
        }
    }
}

@Composable
fun TasksScreen(nav: NavHostController) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("每日任务", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        LazyColumn {
            items(MockData.tasks) { task ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(modifier = Modifier.width(260.dp)) {
                        Text(task.title)
                        Text("（${task.progressText}）", style = MaterialTheme.typography.bodySmall)
                        LinearProgressIndicator(
                            progress = progressFraction(task.progressText),
                            modifier = Modifier.fillMaxWidth().height(6.dp),
                        )
                    }
                    when (task.status) {
                        TaskStatus.PENDING -> Button(onClick = { nav.navigate("taskDetail/${task.title}") }) {
                            Text("前往")
                        }
                        TaskStatus.CLAIMABLE -> Button(onClick = { task.status = TaskStatus.DONE }) {
                            Text("领取")
                        }
                        TaskStatus.DONE -> Text("已完成", style = MaterialTheme.typography.bodyMedium)
                    }
                }
                Divider()
            }
        }
    }
}

private fun progressFraction(progressText: String): Float {
    val parts = progressText.split("/")
    if (parts.size != 2) return 0f
    val done = parts[0].toFloatOrNull() ?: return 0f
    val total = parts[1].toFloatOrNull() ?: return 0f
    if (total <= 0f) return 0f
    return (done / total).coerceIn(0f, 1f)
}

@Composable
fun TaskDetailScreen(nav: NavHostController, title: String) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("任务详情", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(12.dp))
        Text(title)
        Spacer(modifier = Modifier.height(24.dp))
        Text("这是一个占位的任务详情页面，用来测试从任务列表跳转进入新界面的场景。")
    }
}

@Composable
fun TeamScreen(nav: NavHostController) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("编队", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        LazyColumn(modifier = Modifier.weight(1f)) {
            items(MockData.teamSlots) { slot ->
                val selected = MockData.selectedTeamSlot.value == slot
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { MockData.selectedTeamSlot.value = slot }
                        .background(
                            if (selected) MaterialTheme.colorScheme.primaryContainer
                            else MaterialTheme.colorScheme.surface,
                        )
                        .padding(12.dp),
                ) {
                    Text(slot)
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = { MockData.selectedTeamSlot.value = null }) { Text("取消") }
            Button(onClick = { /* 仅作为占位确认按钮 */ }) { Text("确定") }
        }
    }
}

