from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import cv2
import pygame
import os
import random
import threading
import numpy as np
from imutils.video import VideoStream

# Load the pre-trained model
model_name = "dima806/facial_emotions_image_detection"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Initialize pygame
pygame.init()

# Pygame constants and assets
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")).convert_alpha(),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png")).convert_alpha()]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png")).convert_alpha()
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")).convert_alpha(),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png")).convert_alpha()]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")).convert_alpha(),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")).convert_alpha(),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png")).convert_alpha()]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")).convert_alpha(),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")).convert_alpha(),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")).convert_alpha()]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")).convert_alpha(),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png")).convert_alpha()]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png")).convert_alpha()
BG = pygame.image.load(os.path.join("Assets/Other", "Track.png")).convert_alpha()

# Circle collision detection function
def is_circle_collision(circle1_center, circle1_radius, circle2_center, circle2_radius):
    dist_x = circle1_center[0] - circle2_center[0]
    dist_y = circle1_center[1] - circle2_center[1]
    distance = (dist_x ** 2 + dist_y ** 2) ** 0.5
    return distance < (circle1_radius + circle2_radius)

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5
    RADIUS = 25  # Circle collision detection radius

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

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.step_index >= 10:
            self.step_index = 0
        if userInput == "jump" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "duck" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput == "duck"):
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

    def get_center_and_radius(self):
        center_x = self.dino_rect.x + self.image.get_width() // 2
        center_y = self.dino_rect.y + self.image.get_height() // 2
        return (center_x, center_y), self.RADIUS

class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
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

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def get_center_and_radius(self):
        center_x = self.rect.x + self.image[self.type].get_width() // 2
        center_y = self.rect.y + self.image[self.type].get_height() // 2
        radius = max(self.image[self.type].get_width(), self.image[self.type].get_height()) // 2
        return (center_x, center_y), radius

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
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 255
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1

def main():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    paused = False  # Pause state variable
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 20
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    emotion_text = "Neutral"

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = font.render("Points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    # Setup webcam
    cap = VideoStream(src=0).start()
    
    smile_detected = False
    surprise_detected = False
    fear_detected = False

    def process_frame():
        nonlocal smile_detected, surprise_detected, fear_detected, emotion_text
        while run:
            if not paused:  # Process frames only if not paused
                frame = cap.read()
                if frame is None:
                    break

                # Process frame for facial emotion detection
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax(-1).item()
                emotion = model.config.id2label[predicted_class_id]

                # Update the emotion text
                emotion_text = emotion.capitalize()

                # Detect smile and surprise
                smile_detected = emotion == "happy"
                surprise_detected = emotion == "surprise"
                fear_detected = emotion == "fear"

    # Start the frame processing thread
    threading.Thread(target=process_frame, daemon=True).start()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = not paused  # Toggle pause state

        if not paused:  # Update game state only if not paused
            action = None
            if smile_detected:
                action = "jump"
            elif (surprise_detected or fear_detected) and not player.dino_jump:
                action = "duck"

            SCREEN.fill((255, 255, 255))
            player.draw(SCREEN)
            player.update(action)

            if len(obstacles) == 0:
                if random.randint(0, 2) == 0:
                    obstacles.append(SmallCactus(SMALL_CACTUS))
                elif random.randint(0, 2) == 1:
                    obstacles.append(LargeCactus(LARGE_CACTUS))
                elif random.randint(0, 2) == 2:
                    obstacles.append(Bird(BIRD))

            for obstacle in obstacles:
                obstacle.draw(SCREEN)
                obstacle.update()
                player_center, player_radius = player.get_center_and_radius()
                obstacle_center, obstacle_radius = obstacle.get_center_and_radius()
                if is_circle_collision(player_center, player_radius, obstacle_center, obstacle_radius):
                    run = False
                    death_count += 1

            background()
            cloud.draw(SCREEN)
            cloud.update()
            score()

            # Capture and display the camera feed in the bottom-right corner
            frame = cap.read()
            if frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                camera_feed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_feed = cv2.resize(camera_feed, (150, 150))
                camera_feed = pygame.surfarray.make_surface(camera_feed)
                SCREEN.blit(camera_feed, (SCREEN_WIDTH - 160, SCREEN_HEIGHT - 160))

            # Display emotion text at the top-center of the screen
            emotion_font = pygame.font.Font('freesansbold.ttf', 20)
            emotion_surface = emotion_font.render(emotion_text, True, (0, 0, 0))
            emotion_rect = emotion_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
            SCREEN.blit(emotion_surface, emotion_rect)

            pygame.display.update()
        else:
            # Display "Paused" text when the game is paused
            pause_font = pygame.font.Font('freesansbold.ttf', 50)
            pause_surface = pause_font.render("Paused", True, (0, 0, 0))
            pause_rect = pause_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            SCREEN.blit(pause_surface, pause_rect)
            pygame.display.update()

        clock.tick(25)

    cap.stop()
    cv2.destroyAllWindows()
    menu(death_count)

def menu(death_count):
    global points
    run = True
    while run:
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)
        if death_count == 0:
            text = font.render("Press any Key to Start", True, (0, 0, 0))
        elif death_count > 0:
            text = font.render("Press any Key to Restart", True, (0, 0, 0))
            score = font.render("Your Score: " + str(points), True, (0, 0, 0))
            scoreRect = score.get_rect()
            scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(score, scoreRect)
        textRect = text.get_rect() 
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        SCREEN.blit(text, textRect)
        SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
            if event.type == pygame.KEYDOWN:
                main()

menu(death_count=0)