import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId


class MyBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers() 
        await self.build_workers()
        await self.build_pylons()  # pylons are protoss supply buildings
        await self.expand()  # expand to a new resource area.
        await self.build_assimilator()  # getting gas
        await self.offensive_force_buildings()
        await self.build_offensive_force() 
        await self.attack()  # basic attacking method

    async def build_workers(self):
        # nexus = command center
        for nexus in self.units(UnitTypeId.NEXUS).ready.noqueue:
            # we want at least 20 workers, otherwise let's allocate 70% of our supply to workers.
            # later we should use some sort of regression algo maybe for this?

            if self.can_afford(UnitTypeId.PROBE):
                await self.do(nexus.train(UnitTypeId.PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first)

    async def expand(self):
        if self.units(UnitTypeId.NEXUS).amount < 2 and self.can_afford(UnitTypeId.NEXUS):
            await self.expand_now()

    async def build_assimilator(self):
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(25.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vaspene))

    async def offensive_force_buildings(self):
        
        #  looking for a pylon to build near
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random  
            
            #  build a cybernetics core if gateway is already done

            if self.units(UnitTypeId.GATEWAY).ready.exists:
                if not self.units(UnitTypeId.CYBERNETICSCORE):
                    if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                        await self.build(UnitTypeId.CYBERNETICSCORE, near = pylon)

            #  building gateway otherwise

            else: 
                if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near = pylon)

    async def build_offensive_force(self):

        # build units for army - Stalkers
        for gateway in self.units(UnitTypeId.GATEWAY).ready.noqueue:
            if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0:
                await self.do(gateway.train(UnitTypeId.STALKER))

    async def attack(self):

        if self.units(UnitTypeId.STALKER).amount > 3: 
            if len(self.known_enemy_units) > 0:
                for stalker in self.units(UnitTypeId.STALKER).idle:
                    await self.do(stalker.attack(random.choice(self.known_enemy_units)))



run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, MyBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=False)