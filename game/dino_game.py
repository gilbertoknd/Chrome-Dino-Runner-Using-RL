import os
import random
import pygame
pygame.init()

#Global Consts
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

#Asset Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

RUNNING = [
    pygame.image.load(os.path.join(ASSETS_DIR, "Dino", "DinoRun1.png")),
    pygame.image.load(os.path.join(ASSETS_DIR, "Dino", "DinoRun2.png")),
]
JUMPING = pygame.image.load(os.path.join(ASSETS_DIR, "Dino", "DinoJump.png"))
DUCKING = [
    pygame.image.load(os.path.join(ASSETS_DIR, "Dino", "DinoDuck1.png")),
    pygame.image.load(os.path.join(ASSETS_DIR, "Dino", "DinoDuck2.png")),
]

SMALL_CACTUS = [
    pygame.image.load(os.path.join(ASSETS_DIR, "Cactus", "SmallCactus1.png")),
    pygame.image.load(os.path.join(ASSETS_DIR, "Cactus", "SmallCactus2.png")),
    pygame.image.load(os.path.join(ASSETS_DIR, "Cactus", "SmallCactus3.png")),
]
LARGE_CACTUS = [
    pygame.image.load(os.path.join(ASSETS_DIR, "Cactus", "LargeCactus1.png")),
    pygame.image.load(os.path.join(ASSETS_DIR, "Cactus", "LargeCactus2.png")),
    pygame.image.load(os.path.join(ASSETS_DIR, "Cactus", "LargeCactus3.png")),
]

BIRD = [
    pygame.image.load(os.path.join(ASSETS_DIR, "Bird", "Bird1.png")),
    pygame.image.load(os.path.join(ASSETS_DIR, "Bird", "Bird2.png")),
]

CLOUD = pygame.image.load(os.path.join(ASSETS_DIR, "Other", "Cloud.png"))
BG = pygame.image.load(os.path.join(ASSETS_DIR, "Other", "Track.png"))


class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, action):
        #Action: 0: Do Nothing, 1: Jump, 2: Duck
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        #Jump
        if action == 1 and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        #Duck
        elif action == 2 and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        #Run / Do Nothing
        elif not (self.dino_jump or action == 2):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < -self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self, game_speed):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self, game_speed, obstacles):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop(0) #Pop from front if goes off screen

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    BIRD_HEIGHTS = [250, 290, 320]

    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = random.choice(self.BIRD_HEIGHTS)
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1


class DinoGame:
    def __init__(self, headless=False):
        self.headless = headless
        
        self.SCREEN = SCREEN
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        self.game_speed = 20
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.points = 0
        self.obstacles = []
        self.player = Dinosaur()
        self.cloud = Cloud()
        self.font = pygame.font.Font("freesansbold.ttf", 20)
        return self.get_state()

    def get_state(self):
        #Feature extraction
        #Distance to next obstacle
        if len(self.obstacles) > 0:
            distance = self.obstacles[0].rect.x - self.player.dino_rect.x
            #TODO: Normalize distance
            obstacle_y = self.obstacles[0].rect.y
            
            #Type: 0: Small Cactus, 1: Large Cactus, 2: Bird.
            #Using width/height as class check
            if isinstance(self.obstacles[0], Bird):
                 obs_type = 2
            elif isinstance(self.obstacles[0], LargeCactus):
                 obs_type = 1
            else: 
                 obs_type = 0
                 
        else:
            distance = 1000 #Max distance
            obstacle_y = 0
            obs_type = -1

        return [
            distance,
            obstacle_y,
            obs_type,
            self.game_speed,
            self.player.dino_rect.y
        ]

    def step(self, action):
        #User input simulation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return self.get_state(), 0, True, {} 
        
        self.player.update(action)

        #Generating Obstacles
        if len(self.obstacles) == 0:
            if random.randint(0, 2) == 0:
                self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                self.obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                self.obstacles.append(Bird(BIRD))

        #Updating Obstacles
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed, self.obstacles)
            #Collision
            if self.player.dino_rect.colliderect(obstacle.rect):
                return self.get_state(), -10, True, {"score": self.points} #Reward -10 for dying

        #Updating Background
        self.x_pos_bg -= self.game_speed
        if self.x_pos_bg < -BG.get_width():
            self.x_pos_bg = 0

        #Updating points and speed
        self.points += 1
        if self.points % 100 == 0:
            self.game_speed += 1

        return self.get_state(), 0.1, False, {"score": self.points} #Reward for surviving

    def render(self):
        SCREEN.fill((255, 255, 255))
        
        image_width = BG.get_width()
        SCREEN.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        
        self.player.draw(SCREEN)
        
        for obstacle in self.obstacles:
            obstacle.draw(SCREEN)
            
        self.cloud.draw(SCREEN)
        self.cloud.update(self.game_speed)
        
        text = self.font.render("Points: " + str(self.points), True, (0,0,0))
        SCREEN.blit(text, (900, 40))

    def close(self):
        pygame.quit()
