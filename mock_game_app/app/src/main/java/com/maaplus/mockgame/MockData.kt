package com.maaplus.mockgame

import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf

/**
 * 测试用假数据：内容故意贴近skills/mobile-game-ui-nav/SKILL.md里总结的常见二次元/
 * 养成类手游UI模式（主城图标墙、tab+列表+详情的公告页、角色详情多级tab、兑换列表、
 * 任务进度、编队多选、弹窗诱导跳转、纯图标按钮、敏感词误报），用清晰可控的文字替代
 * 真实游戏里那些花体banner/滚动内容，方便验证探索/OCR/节点匹配/BFS导航这套流程本身
 * 对不对，而不必被真实游戏的OCR噪声干扰。
 */

enum class TaskStatus { PENDING, CLAIMABLE, DONE }

data class TaskItem(
    val title: String,
    val progressText: String,
    var status: TaskStatus,
)

data class ShopItem(
    val name: String,
    var count: Int,
    val cost: Int,
)

data class Announcement(
    val tab: String,
    val title: String,
    val content: String,
)

object MockData {
    val characters = listOf("艾拉", "凯恩", "布莱恩", "索菲亚")

    val announcementTabs = listOf("系统公告", "活动公告", "资讯")

    val announcements = listOf(
        Announcement(
            "系统公告",
            "关于打击违规充值和不诚信退款的公告",
            "本公司始终坚持诚信运营，对第三方渠道违规充值、恶意退款等行为保持零容忍态度。" +
                "如发现相关线索，欢迎通过客服渠道如实举报，我们将依法依规处理。",
        ),
        Announcement(
            "系统公告",
            "07月09日停服维护结束公告",
            "本次维护已于07月09日4点59分结束，感谢各位指挥官的耐心等待。",
        ),
        Announcement(
            "活动公告",
            "疑凶追影",
            "限时活动疑凶追影现已开启，完成关卡即可获得丰厚奖励，活动截止时间见页面倒计时。",
        ),
        Announcement(
            "活动公告",
            "蚀日的烙痕",
            "全新烙印系统上线，参与活动即可解锁限定烙印词条。",
        ),
        Announcement(
            "资讯",
            "SNS关注奖励",
            "关注官方社交账号，凭截图领取兑换码，可在兑换中心使用。",
        ),
    )

    val shopItems = mutableStateListOf(
        ShopItem("经验药剂", 5, 100),
        ShopItem("强化石", 12, 200),
        ShopItem("契约凭证", 3, 500),
        ShopItem("狄斯币礼包", 8, 50),
    )

    val tasks = mutableStateListOf(
        TaskItem("完成1次日常派遣", "0/1", TaskStatus.PENDING),
        TaskItem("消耗100点体力", "40/100", TaskStatus.PENDING),
        TaskItem("登录游戏", "1/1", TaskStatus.CLAIMABLE),
        TaskItem("查看今日公告", "1/1", TaskStatus.DONE),
    )

    val teamSlots = (1..7).map { "队伍-%02d".format(it) }
    val selectedTeamSlot = mutableStateOf<String?>(null)

    val showLoginRewardPopup = mutableStateOf(true)
}
