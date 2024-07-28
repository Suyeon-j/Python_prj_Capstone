import asyncio
import cv2
import pygame  # pip install -U pygame==2.6.0
import os
import random
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

async def load_model():
    model_name = "dima806/facial_emotions_image_detection"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    return feature_extractor, model

def initialize_pygame():
    pygame.init()
    screen_height = 600
    screen_width = 1100
    screen = pygame.display.set_mode((screen_width, screen_height))
    assets = {
        "RUNNING": [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
                   pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))],
        "JUMPING": pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png")),
        "DUCKING": [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
                   pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))],
        "SMALL_CACTUS": [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                        pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                        pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))],
        "LARGE_CACTUS": [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                        pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                        pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))],
        "BIRD": [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
                pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))],
        "CLOUD": pygame.image.load(os.path.join("Assets/Other", "Cloud.png")),
        "BG": pygame.image.load(os.path.join("Assets/Other", "Track.png")),
    }
    return screen, assets

async def main():
    global obstacles
    feature_extractor, model = await load_model()
    screen, assets = initialize_pygame()
    game_speed = 20
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    obstacles = []
    clock = pygame.time.Clock()
    font = pygame.font.Font('freesansbold.ttf', 20)
    screen_width = 1100
    screen_height = 600
    
    player = Dinosaur(assets)
    cloud = Cloud(assets['CLOUD'])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_counter = 0
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % 10 == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
            emotion = model.config.id2label[predicted_class_id]
            smile_detected = emotion == "happy"
            surprise_detected = emotion == "surprise"

        frame_counter += 1

        screen.fill((255, 255, 255))
        player.draw(screen)
        player.update(smile_detected, surprise_detected)

        if len(obstacles) == 0:
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(assets['SMALL_CACTUS']))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(assets['LARGE_CACTUS']))
            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(assets['BIRD']))

        for obstacle in obstacles:
            obstacle.draw(screen)
            obstacle.update(game_speed)
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(2000)
                cap.release()
                cv2.destroyAllWindows()
                await menu(screen, assets, game_speed, points, font, screen_width, screen_height)
                return

        background(screen, assets['BG'], game_speed, x_pos_bg, y_pos_bg)
        cloud.draw(screen)
        cloud.update(game_speed)
        points = score(screen, points, font)
        clock.tick(30)
        pygame.display.update()

    cap.release()
    cv2.destroyAllWindows()

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self, assets):
        self.duck_img = assets['DUCKING']
        self.run_img = assets['RUNNING']
        self.jump_img = assets['JUMPING']
        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False
        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, smile_detected, surprise_detected):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.step_index >= 10:
            self.step_index = 0

        if not self.dino_jump:
            if smile_detected:
                self.dino_duck = False
                self.dino_run = False
                self.dino_jump = True
            elif surprise_detected:
                self.dino_duck = True
                self.dino_run = False
                self.dino_jump = False
            else:
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

    def draw(self, screen):
        screen.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Cloud:
    def __init__(self, image):
        self.x = 1100
        self.y = random.randint(50, 100)
        self.image = image
        self.width = self.image.get_width()

    def update(self, game_speed):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = 1100
            self.y = random.randint(50, 100)

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = 1100

    def update(self, game_speed):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.remove(self)

    def draw(self, screen):
        screen.blit(self.image[self.type], self.rect)

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
    def __init__(self, image):
        self.type = random.randint(0, 1)
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, screen):
        if self.index >= 9:
            self.index = 0
        screen.blit(self.image[self.index // 5], self.rect)
        self.index += 1

def background(screen, bg_image, game_speed, x_pos_bg, y_pos_bg):
    image_width = bg_image.get_width()
    screen.blit(bg_image, (x_pos_bg, y_pos_bg))
    screen.blit(bg_image, (image_width + x_pos_bg, y_pos_bg))
    if x_pos_bg <= -image_width:
        x_pos_bg = 0
    x_pos_bg -= game_speed

def score(screen, points, font):
    points += 1
    text = font.render(f"Points: {points}", True, (0, 0, 0))
    screen.blit(text, (950, 50))
    return points

async def menu(screen, assets, game_speed, points, font, screen_width, screen_height):
    run = True
    while run:
        screen.fill((255, 255, 255))
        text = font.render("Press any Key to Restart", True, (0, 0, 0))
        score = font.render(f"Your Score: {points}", True, (0, 0, 0))
        screen.blit(text, (screen_width // 2 - 100, screen_height // 2))
        screen.blit(score, (screen_width // 2 - 100, screen_height // 2 + 50))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                await main()

if __name__ == "__main__":
    asyncio.run(main())
