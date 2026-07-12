package com.maaplus.mockgame

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MockGameNavHost()
                }
            }
        }
    }
}

@Composable
fun MockGameNavHost() {
    val navController: NavHostController = rememberNavController()
    NavHost(navController = navController, startDestination = "hub") {
        composable("hub") { HubScreen(navController) }
        composable("announcements") { AnnouncementsScreen(navController) }
        composable("characters") { CharactersScreen(navController) }
        composable("character/{name}") { backStackEntry ->
            val name = backStackEntry.arguments?.getString("name") ?: ""
            CharacterDetailScreen(navController, name)
        }
        composable("shop") { ShopScreen(navController) }
        composable("tasks") { TasksScreen(navController) }
        composable("taskDetail/{title}") { backStackEntry ->
            val title = backStackEntry.arguments?.getString("title") ?: ""
            TaskDetailScreen(navController, title)
        }
        composable("team") { TeamScreen(navController) }
        composable("settings") { SettingsScreen(navController) }
    }
}
