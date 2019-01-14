import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

class MyBot(sc2.BotAI):

    async def on_step(self, iteration):
        await self.distribute_workers()  # in sc2/bot_ai.py
    
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Terran, MyBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=True)