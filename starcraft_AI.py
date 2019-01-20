#  StarCraft2 - API
import sc2
import random
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId

#  deep learning
import cv2
import numpy as np


class MyBot(sc2.BotAI):

    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.scout()
        await self.distribute_workers() 
        await self.build_workers()
        await self.build_pylons()  # pylons are protoss supply buildings
        await self.expand()  # expand to a new resource area.
        await self.build_assimilator()  # getting gas
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()  # DEEP LEARNING S**T
        await self.attack()  # basic attacking method

    async def intel(self):

        draw_params = {
            UnitTypeId.NEXUS:           [15, (  0, 255,   0)],
            UnitTypeId.PYLON:           [ 3, ( 20, 235,   0)],
            UnitTypeId.PROBE:           [ 1, ( 55, 200,   0)],
            UnitTypeId.ASSIMILATOR:     [ 2, ( 55, 200,   0)],
            UnitTypeId.GATEWAY:         [ 3, (200, 100,   0)],
            UnitTypeId.CYBERNETICSCORE: [ 3, (150, 150,   0)],
            UnitTypeId.STARGATE:        [ 5, (255,   0,   0)],
            UnitTypeId.VOIDRAY:         [ 3, (255, 100,   0)],
            UnitTypeId.ROBOTICSFACILITY:[ 5, (215, 155,   0)],
        }

        base_names = ['nexus', 'commandcenter', 'hatchery']
        worker_names = ['probe', 'scv', 'drone']
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit_type in draw_params:
            for unit in self.units(unit_type).ready:
                unit_coor = unit.position
                cv2.circle(game_data, (int(unit_coor[0]), int(unit_coor[1])), draw_params[unit_type][0], draw_params[unit_type][1], -1)

        for enemy_building in self.known_enemy_structures:
            enemy_coor = enemy_building.position

            if enemy_building.name.lower() not in base_names:
                cv2.circle(game_data, (int(enemy_coor[0]), int(enemy_coor[1])), 5, (200, 50, 212), -1)

            ## O CHUJ CHODZI - DO WYJEBANIA ????!!!!
            elif enemy_building.name.lower() in base_names:
                cv2.circle(game_data, (int(enemy_coor[0]), int(enemy_coor[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                unit_coor = enemy_unit.position

                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(unit_coor[0]), int(unit_coor[1])), 1, (55, 0, 155), -1)
                else: 
                    cv2.circle(game_data, (int(unit_coor[0]), int(unit_coor[1])), 3, (50, 0, 215), -1)

        for observer in self.units(UnitTypeId.OBSERVER).ready:
            unit_coor = observer.position
            cv2.circle(game_data, (int(unit_coor[0]), int(unit_coor[1])), 1, (255, 255, 255), -1)

        # #   flipping image to match coors
        flipped = cv2.flip(game_data, 0)

        #   resize - only to visualization
        resized = cv2.resize(flipped, dsize = None, fx = 2, fy = 2)
        cv2.imshow('Intel', resized)
        cv2.waitKey(1)

    async def scout(self):

        if len(self.units(UnitTypeId.OBSERVER)) > 0:

            scout = self.units(UnitTypeId.OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                await self.do(scout.move(move_to))

        else:
            for robotfacility in self.units(UnitTypeId.ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                    await self.do(robotfacility.train(UnitTypeId.OBSERVER))


    async def build_workers(self):

        if (len(self.units(UnitTypeId.NEXUS)) * 16) > len(self.units(UnitTypeId.PROBE)) and len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(UnitTypeId.NEXUS).ready.noqueue:
                if self.can_afford(UnitTypeId.PROBE):
                    await self.do(nexus.train(UnitTypeId.PROBE))


    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first)


    async def expand(self):
        if self.units(UnitTypeId.NEXUS).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(UnitTypeId.NEXUS):
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
        
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random

            if self.units(UnitTypeId.GATEWAY).ready.exists and not self.units(UnitTypeId.CYBERNETICSCORE):
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

            elif len(self.units(UnitTypeId.GATEWAY)) < 1:
                if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon)

            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.ROBOTICSFACILITY)) < 1:
                    if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and not self.already_pending(UnitTypeId.ROBOTICSFACILITY):
                        await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)

            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE):
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon)


    async def build_offensive_force(self):

        for stargate in self.units(UnitTypeId.STARGATE).ready.noqueue:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                await self.do(stargate.train(UnitTypeId.VOIDRAY))


    async def attack(self):

        aggressive_units = {
                            # UnitTypeId.STALKER: [15, 5],
                            UnitTypeId.VOIDRAY: [8, 3]
                            }

        for UNIT in aggressive_units:
            for s in self.units(UNIT).idle:
                await self.do(s.attack(self.find_target(self.state)))

        # for UNIT in aggressive_units:
        #     if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
        #         for s in self.units(UNIT).idle:
        #             await self.do(s.attack(self.find_target(self.state)))

        #     elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
        #         if len(self.known_enemy_units) > 0:
        #             for s in self.units(UNIT).idle:
        #                 await self.do(s.attack(random.choice(self.known_enemy_units)))

        #     elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
        #         if len(self.known_enemy_units) > 0:
        #             for s in self.units(UNIT).idle:
        #                 await self.do(s.attack(random.choice(self.known_enemy_units)))


    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]


    def random_location_variance(self, enemy_start_location):

        x_coor = enemy_start_location[0]
        y_coor = enemy_start_location[1]

        x_coor += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y_coor += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x_coor < 0: 
            x_coor = 0
        if y_coor < 0: 
            y_coor = 0
        if x_coor > self.game_info.map_size[0]: 
            x_coor = self.game_info.map_size[0]
        if y_coor > self.game_info.map_size[1]: 
            y_coor = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x_coor, y_coor)))
        return go_to

run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, MyBot()),
    Computer(Race.Terran, Difficulty.Medium)
], realtime=False)