import pygame
import os
import random
import neat
import pickle
import pygame_menu

GAME_WIDTH = 1100
GAME_HEIGHT = 700
clock = pygame.time.Clock()
pygame.font.init()
pygame.init()
os.environ['SDL_VIDEO_CENTERED'] = '1'
GAME_WINDOW = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
pygame.display.set_icon(pygame.image.load("res/icon.png"))
pygame.display.set_caption("AI CAR GAME")
STAT_FONT = pygame.font.SysFont("comicsans", 30)
GEN = 0
MODE = 0


class Background:
    BG_IMG = pygame.image.load('res/bg.png')
    bgImages = [BG_IMG, BG_IMG]

    def __init__(self, y1, x, y2, animationSpeed):
        self.y1 = y1
        self.x = x
        self.y2 = y2
        self.animationSpeed = animationSpeed

    def drawBackground(self, gameWindow):

        self.y1 += self.animationSpeed
        self.y2 += self.animationSpeed

        if self.BG_IMG.get_height() + self.y1 > 2 * GAME_HEIGHT:
            self.y1 = self.y2 - self.BG_IMG.get_height()
        if self.BG_IMG.get_height() + self.y2 > 2 * GAME_HEIGHT:
            self.y2 = self.y1 - self.BG_IMG.get_height()

        gameWindow.blit(self.bgImages[0], (0, self.y1))
        gameWindow.blit(self.bgImages[1], (0, self.y2))


class Wall:
    GAP = 150
    WALL_IMG = pygame.image.load('res/wall.png')

    def __init__(self, y, animationSpeed):
        self.y = y
        self.LEFT_WALL = self.WALL_IMG
        self.RIGHT_WALL = pygame.transform.flip(self.WALL_IMG, True, False)
        self.leftX = 0
        self.rightX = 0
        self.passed = False
        self.width = 0
        self.animationSpeed = animationSpeed
        self.set_width()

    def set_width(self):
        self.width = random.randrange(50, 950)
        self.leftX = self.width - self.WALL_IMG.get_width()
        self.rightX = self.leftX + self.WALL_IMG.get_width() + self.GAP

    def move(self):
        self.y += self.animationSpeed

    def drawWalls(self, gameWindow):
        gameWindow.blit(self.WALL_IMG, (self.leftX, self.y))
        gameWindow.blit(self.WALL_IMG, (self.rightX, self.y))

    def collide(self, playerCar):
        carMask = playerCar.getMask()
        leftWallMask = pygame.mask.from_surface(self.LEFT_WALL)
        rightWallMask = pygame.mask.from_surface(self.RIGHT_WALL)

        leftOffset = (self.leftX - playerCar.x, self.y - playerCar.y)
        rightOffset = (self.rightX - playerCar.x, self.y - playerCar.y)

        leftCollisionPoint = carMask.overlap(leftWallMask, leftOffset)
        rightCollisionPoint = carMask.overlap(rightWallMask, rightOffset)

        if leftCollisionPoint or rightCollisionPoint:
            return True
        else:
            return False


class Player:
    PLAYER_IMG = pygame.image.load('res/car.png')

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xVel = 20

    def drawCar(self, gameWindow):
        gameWindow.blit(self.PLAYER_IMG, (self.x, self.y))

    def moveRight(self):
        if self.x + self.PLAYER_IMG.get_width() < GAME_WIDTH:
            self.x += self.xVel

    def moveLeft(self):
        if self.x > 0:
            self.x -= self.xVel

    def getMask(self):
        return pygame.mask.from_surface(self.PLAYER_IMG)


def drawGame(gameWindow, bg, walls, score, cars):
    bg.drawBackground(gameWindow)
    for wall in walls:
        wall.drawWalls(gameWindow)
    for car in cars:
        car.drawCar(gameWindow)
    scoreText = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 0))
    quitText = STAT_FONT.render("ESC to stop", 1, (0, 0, 0))
    gameWindow.blit(scoreText, (GAME_WIDTH - 30 - scoreText.get_width(), 10))
    gameWindow.blit(quitText, (20, 10))
    pygame.display.update()


def drawGameAILearning(gameWindow, bg, cars, walls, score, gen):
    bg.drawBackground(gameWindow)
    for wall in walls:
        wall.drawWalls(gameWindow)
    for car in cars:
        car.drawCar(gameWindow)
        pygame.draw.line(gameWindow, (255, 255, 0), (int(car.x + car.PLAYER_IMG.get_width() // 2), car.y),
                         (walls[0].rightX, walls[0].y), 1)
        pygame.draw.line(gameWindow, (255, 255, 0), (int(car.x + car.PLAYER_IMG.get_width() // 2), car.y),
                         (walls[0].rightX - walls[0].GAP, walls[0].y), 1)
    scoreText = STAT_FONT.render("Score: " + str(score), 1, (0, 0, 0))
    quitText = STAT_FONT.render("ESC to stop", 1, (0, 0, 0))
    generation = STAT_FONT.render("Gen: " + str(gen), 1, (0, 0, 0))
    alive = STAT_FONT.render("Alive: " + str(len(cars)), 1, (0, 0, 0))
    gameWindow.blit(scoreText, (GAME_WIDTH - 30 - scoreText.get_width(), 10))
    gameWindow.blit(generation, (20, 10))
    gameWindow.blit(alive, (20, 35))
    gameWindow.blit(quitText, (20, 60))
    pygame.display.update()


def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    cars = []
    nets = []
    ge = []
    animationSpeed = 10
    bg = Background(0, 0, GAME_HEIGHT, 10)
    walls = [Wall(0, animationSpeed)]
    score = 0

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Player(GAME_WIDTH // 2, GAME_HEIGHT - 125))
        ge.append(genome)

    addWall = False
    running = True
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            ge[0].fitness = 2000
            cars.clear()
            GEN = 0

        wallInd = 0
        if len(cars) > 0:
            if len(walls) > 1 and cars[0].y > walls[0].y + walls[0].RIGHT_WALL.get_height():
                wallInd = 1
        else:
            break

        for x, car in enumerate(cars):
            ge[x].fitness += 0.1

            output = nets[x].activate((animationSpeed,
                                       (car.x + car.PLAYER_IMG.get_width() // 2)
                                       - (walls[wallInd].rightX - walls[wallInd].GAP),
                                       walls[wallInd].rightX - (car.x + car.PLAYER_IMG.get_width() // 2)))
            if output[0] > 0.8:
                car.moveRight()
            if output[1] > 0.8:
                car.moveLeft()

        rem = []
        for wall in walls:
            for x, car in enumerate(cars):
                if wall.collide(car):
                    ge[x].fitness -= 1
                    cars.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not wall.passed and wall.y > car.y:
                    wall.passed = True
                    addWall = True

            if wall.y + wall.RIGHT_WALL.get_height() > GAME_HEIGHT:
                rem.append(wall)
            wall.move()

        if addWall:
            score += 1
            if animationSpeed < 30:
                if score % 5 == 0:
                    animationSpeed += 1
                    bg.animationSpeed += 1
                    for car in cars:
                        if car.xVel < 50:
                            car.xVel += 2
            for g in ge:
                g.fitness += 5
            walls.append(Wall(0, animationSpeed))
            addWall = False

        for r in rem:
            walls.remove(r)

        if score == 150:
            for car in cars:
                car.xVel = 0
        drawGameAILearning(GAME_WINDOW, bg, cars, walls, score, GEN)


def runAILearning():
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    p.run(eval_genomes, 5)


def gameLoop():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    pickle_in = open("bestCar.pickle", "rb")
    bestCar = pickle.load(pickle_in)
    startGame(bestCar, config)


def startGame(genomes, config):
    cars = []
    animationSpeed = 10
    bg = Background(0, 0, GAME_HEIGHT, 10)
    walls = [Wall(0, animationSpeed)]
    score = 0
    net = neat.nn.FeedForwardNetwork.create(genomes, config)
    cars.append(Player(GAME_WIDTH // 2, GAME_HEIGHT - 125))

    if MODE == 1:
        cars.append(Player(GAME_WIDTH // 2, GAME_HEIGHT - 125))
        cars[1].PLAYER_IMG = pygame.image.load('res/car1.png')

    addWall = False
    running = True
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            break

        if len(cars) >= 2:
            if keys[pygame.K_LEFT]:
                cars[1].moveLeft()

            elif keys[pygame.K_RIGHT]:
                cars[1].moveRight()

        wallInd = 0
        if len(cars) > 0:
            if len(walls) > 1 and cars[0].y > walls[0].y + walls[0].RIGHT_WALL.get_height():
                wallInd = 1
        else:
            break

        output = net.activate((animationSpeed,
                               (cars[0].x + cars[0].PLAYER_IMG.get_width() // 2) - (
                                       walls[wallInd].rightX - walls[wallInd].GAP),
                               walls[wallInd].rightX - (cars[0].x + cars[0].PLAYER_IMG.get_width() // 2)))

        if output[0] > 0.8:
            cars[0].moveRight()
        if output[1] > 0.8:
            cars[0].moveLeft()

        rem = []
        for wall in walls:
            for x, car in enumerate(cars):
                if len(cars) > 2 and wall.collide(cars[1]):
                    running = False
                    break
                elif wall.collide(car):
                    cars.pop(x)

                if not wall.passed and wall.y > car.y:
                    wall.passed = True
                    addWall = True

            if wall.y + wall.RIGHT_WALL.get_height() > GAME_HEIGHT:
                rem.append(wall)
            wall.move()

            if (len(cars) >= 2) and wall.collide(cars[1]):
                cars.remove(cars[1])
                running = False
                break

        if addWall:
            score += 1
            if animationSpeed < 30:
                if score % 5 == 0:
                    animationSpeed += 1
                    bg.animationSpeed += 1
                    for car in cars:
                        if car.xVel < 50:
                            car.xVel += 2
            walls.append(Wall(0, animationSpeed))
            addWall = False

        for r in rem:
            walls.remove(r)

        if score == 500:
            for car in cars:
                car.xVel = 0

        if len(cars) < 2 and MODE == 1:
            break

        drawGame(GAME_WINDOW, bg, walls, score, cars)


def setMode(mode, value):
    global MODE
    MODE = value


def makeMenu(gameWindow, gameWidth, gameHeight):
    font = pygame_menu.font.FONT_NEVIS
    fontSize = 35
    fontColor = (0, 0, 0)
    shadowColor = (255, 255, 0)
    selectionColor = (0, 0, 255)

    menu = pygame_menu.Menu(gameHeight, gameWidth, 'Reinforcement learning Car Game', onclose=pygame_menu.events.EXIT,
                            theme=pygame_menu.themes.THEME_BLUE, mouse_visible=False, mouse_enabled=False)
    menu.add_label("Look how AI learns the game or just try yourself vs bot!", font_color=fontColor,
                   shadow=True, shadow_color=shadowColor, font_name=font,
                   font_size=fontSize)
    menu.add_vertical_margin(100)
    menu.add_selector('Mode: ', [('AI Game', 0), ('Self vs AI', 1)], onchange=setMode,
                      font_name=font, font_size=fontSize,
                      font_color=fontColor, shadow=True, shadow_color=shadowColor, selection_color=selectionColor)
    menu.add_button('Play!', gameLoop, font_name=font, font_size=fontSize,
                    font_color=fontColor, shadow=True, shadow_color=shadowColor, selection_color=selectionColor)
    menu.add_button('Look how AI learns the game!', runAILearning,  font_name=font, font_size=fontSize,
                    font_color=fontColor, shadow=True, shadow_color=shadowColor, selection_color=selectionColor)
    menu.add_button('Quit!', pygame_menu.events.EXIT,  font_name=font, font_size=fontSize,
                    font_color=fontColor, shadow=True, shadow_color=shadowColor, selection_color=selectionColor)
    menu.mainloop(gameWindow)


def main(gameWindow, gameWidth, gameHeight):
    makeMenu(gameWindow, gameWidth, gameHeight)


if __name__ == '__main__':
    main(GAME_WINDOW, GAME_WIDTH, GAME_HEIGHT)
