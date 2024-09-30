# streamlit run maze.py
import tkinter as tk
from tkinter import messagebox
import random
from PIL import Image, ImageTk
import os
import sys
import cv2
import torch
import streamlit as st
from transformers import AutoModelForImageClassification, AutoProcessor

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class GameState:
    IN_PROGRESS = 0
    SUCCESS = 1
    FAILED = 2

class MazeGame:
    def __init__(self, master):
        self.master = master
        self.master.title("미로 탈출 게임")
        self.master.geometry("1200x600")  # Extend width to accommodate camera feed
        self.master.bind("<Escape>", lambda event: self.pause_game())
        
        # Add camera feed frame on the right
        self.camera_frame = tk.Frame(self.master, width=400, height=600, bg="black")
        self.camera_frame.pack(side="right", fill="y")
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(expand=True)
        self.emotion_label = tk.Label(self.camera_frame, text="Emotion: neutral", font=("Helvetica", 14), bg="black", fg="white")
        self.emotion_label.pack(pady=10)

        self.start_frame = tk.Frame(self.master)
        self.start_frame.pack(expand=True, fill="both")

        self.game_frame = tk.Frame(self.master)
        self.canvas = tk.Canvas(self.game_frame, bg="white")
        self.canvas.pack(expand=True, fill="both")

        self.load_images()
        self.create_start_screen()

        self.move_delay = 100
        self.current_level = 1
        self.paused = False
        self.has_key = False

        # Initialize the facial emotion recognition model
        self.model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.processor = AutoProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        self.cap = cv2.VideoCapture(0)  # Open the webcam
        self.emotion = "neutral"  # Default emotion

    def load_images(self):
        self.wall_img = self.load_image("./maze/img/wall.jpg")
        self.floor_img = self.load_image("./maze/img/floor.jpg")
        self.key_img = self.load_image("./maze/img/k.jpg")
        self.lock_img = self.load_image("./maze/img/lock.jpg")
        self.exit_img = self.load_image("./maze/img/cat_can.jpeg")
        self.player_img = self.load_image("./maze/img/mimi_s.png")

    def load_image(self, filename):
        img = Image.open(resource_path(filename))
        img = img.resize((60, 60))
        return ImageTk.PhotoImage(img)

    def create_start_screen(self):
        title_img = Image.open(resource_path("./maze/img/mimi.png"))
        title_img = title_img.resize((200, 200))
        self.title_photo = ImageTk.PhotoImage(title_img)

        title_label = tk.Label(self.start_frame, image=self.title_photo)
        title_label.pack(pady=20)

        game_title = tk.Label(self.start_frame, text="미로 탈출 게임", font=("Helvetica", 24, "bold"))
        game_title.pack(pady=10)

        level_frame = tk.Frame(self.start_frame)
        level_frame.pack(pady=20)

        for level in range(1, 5):
            btn = tk.Button(level_frame, text=f"레벨 {level}", command=lambda l=level: self.start_game(l),
                            font=("Helvetica", 14), width=10, bg="#4CAF50", fg="white")
            btn.pack(side=tk.LEFT, padx=10)

    def start_game(self, level):
        self.current_level = level
        self.start_frame.pack_forget()
        self.game_frame.pack(expand=True, fill="both")
        self.camera_frame.pack(side="right", fill="y")
        self.reset()
        self.main_proc()

        
    def generate_maze(self):
        sizes = {1: 5, 2: 7, 3: 11, 4: 13}
        size = sizes.get(self.current_level, 5)  # 레벨에 맞는 사이즈를 가져오고, 기본값으로 5를 설정
        self.maze_width = self.maze_height = size
        self.maze = [[1] * self.maze_width for _ in range(self.maze_height)]
        
        if self.current_level <= 2:
            self.recursive_backtracking(1, 1)
        elif self.current_level == 3:
            self.prims_algorithm()
        else:
            self.hunt_and_kill_algorithm()

        self.set_key_lock_exit()
        
    def recursive_backtracking(self, x, y):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.maze_width and 0 <= ny < self.maze_height and self.maze[ny][nx] == 1:
                self.maze[y + dy // 2][x + dx // 2] = 0
                self.maze[ny][nx] = 0
                self.recursive_backtracking(nx, ny)

    def prims_algorithm(self):
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0
        frontier = [(start_x, start_y, dx, dy) for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]
                    if 0 <= start_x + dx < self.maze_width and 0 <= start_y + dy < self.maze_height]

        while frontier:
            x, y, dx, dy = random.choice(frontier)
            nx, ny = x + dx, y + dy
            if self.maze[ny][nx] == 1:
                self.maze[y + dy // 2][x + dx // 2] = 0
                self.maze[ny][nx] = 0
                for ndx, ndy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                    if 0 <= nx + ndx < self.maze_width and 0 <= ny + ndy < self.maze_height and self.maze[ny + ndy][nx + ndx] == 1:
                        frontier.append((nx, ny, ndx, ndy))
            frontier = [(fx, fy, fdx, fdy) for fx, fy, fdx, fdy in frontier if (fx, fy, fdx, fdy) != (x, y, dx, dy)]

    def hunt_and_kill_algorithm(self):
        x, y = 1, 1
        self.maze[y][x] = 0

        while True:
            while True:
                directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
                random.shuffle(directions)
                moved = False
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.maze_width and 0 <= ny < self.maze_height and self.maze[ny][nx] == 1:
                        self.maze[y + dy // 2][x + dx // 2] = 0
                        self.maze[ny][nx] = 0
                        x, y = nx, ny
                        moved = True
                        break
                if not moved:
                    break

            found = False
            for hy in range(1, self.maze_height, 2):
                for hx in range(1, self.maze_width, 2):
                    if self.maze[hy][hx] == 1:
                        for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                            nx, ny = hx + dx, hy + dy
                            if 0 <= nx < self.maze_width and 0 <= ny < self.maze_height and self.maze[ny][nx] == 0:
                                self.maze[hy + dy // 2][hx + dx // 2] = 0
                                self.maze[hy][hx] = 0
                                x, y = hx, hy
                                found = True
                                break
                        if found:
                            break
                if found:
                    break
            if not found:
                break
    
    def set_key_lock_exit(self):
        self.exit_x, self.exit_y = self.maze_width - 2, self.maze_height - 2
        self.maze[self.exit_y][self.exit_x+1] = 1
        self.maze[self.exit_y+1][self.exit_x] = 1
        self.maze[self.exit_y-1][self.exit_x] = 1
        self.lock_x, self.lock_y = self.exit_x - 1, self.exit_y
        while True:
            self.key_x, self.key_y = random.randint(1, self.maze_width - 2), random.randint(1, self.maze_height - 2)
            if self.maze[self.key_y][self.key_x] == 0 and (self.key_x, self.key_y) != (self.lock_x, self.lock_y):
                break
        path = self.find_path(1, 1, self.lock_x, self.lock_y)
        if not path:
            self.generate_maze()

    def find_path(self, start_x, start_y, end_x, end_y):
        queue = [(start_x, start_y, [])]
        visited = set()

        while queue:
            x, y, path = queue.pop(0)
            if (x, y) == (end_x, end_y):
                return path
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze_width and 0 <= ny < self.maze_height and self.maze[ny][nx] == 0:
                    queue.append((nx, ny, path + [(x, y)]))
        return None

    def capture_emotion(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                # happy -> 우 , surprise -> 상, angry -> 하, neutral -> 좌
                # Convert the frame to RGB and prepare it for emotion detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                inputs = self.processor(images=img, return_tensors="pt")

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_class = outputs.logits.argmax(-1).item()

                # Map the predicted class to emotion
                emotions = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
                self.emotion = emotions[predicted_class]
                self.emotion_label.config(text=f"Emotion: {self.emotion}")

                # Display the frame in the GUI
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            except Exception as e:
                print(f"Error during emotion detection: {e}")

    def move_player_based_on_emotion(self):
        if self.emotion == "happy":  # Move right
            self.move_player(1, 0)
        elif self.emotion == "surprise":  # Move up
            self.move_player(0, -1)
        elif self.emotion == "angry":  # Move down
            self.move_player(0, 1)
        elif self.emotion == "neutral":  # Move left
            self.move_player(-1, 0)

    def reset(self):
        self.generate_maze()
        self.player_x, self.player_y = 1, 1
        self.has_key = False
        self.update_canvas()

    def update_canvas(self):
        self.canvas.delete("all")
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y][x] == 1:
                    self.canvas.create_image(x * 60, y * 60, anchor="nw", image=self.wall_img)
                else:
                    self.canvas.create_image(x * 60, y * 60, anchor="nw", image=self.floor_img)
        if not self.has_key:
            self.canvas.create_image(self.key_x * 60, self.key_y * 60, anchor="nw", image=self.key_img)
        self.canvas.create_image(self.lock_x * 60, self.lock_y * 60, anchor="nw", image=self.lock_img)
        self.canvas.create_image(self.exit_x * 60, self.exit_y * 60, anchor="nw", image=self.exit_img)
        self.canvas.create_image(self.player_x * 60, self.player_y * 60, anchor="nw", image=self.player_img)

    def level_selection(self):
        self.game_frame.pack_forget()
        self.start_frame.pack(expand=True, fill="both")

    def main_proc(self):
        self.capture_emotion()
        self.move_player_based_on_emotion()
        self.master.after(self.move_delay, self.main_proc)

    def move_player(self, dx, dy):
        new_x, new_y = self.player_x + dx, self.player_y + dy
        if 0 <= new_x < self.maze_width and 0 <= new_y < self.maze_height:
            if self.maze[new_y][new_x] == 0:
                self.player_x, self.player_y = new_x, new_y
                self.check_game_status()
                self.update_canvas()

    def check_game_status(self):
        if (self.player_x, self.player_y) == (self.key_x, self.key_y) and not self.has_key:
            self.has_key = True
            self.update_canvas()
        elif (self.player_x, self.player_y) == (self.exit_x, self.exit_y):
            if self.has_key:
                self.show_message("축하합니다! 미로를 탈출했습니다!", GameState.SUCCESS)
            else:
                self.player_x, self.player_y = self.player_x - 1, self.player_y
                self.update_canvas()

    def show_message(self, message, state):
        if state == GameState.SUCCESS:
            messagebox.showinfo("게임 종료", message)
        self.reset()
    
    def pause_game(self):
        self.paused = True
        self.canvas.unbind("<KeyPress>")
        self.canvas.unbind("<KeyRelease>")

        self.pause_frame = tk.Frame(self.game_frame, bg="lightgrey", padx=10, pady=10)
        self.pause_frame.place(relx=0.5, rely=0.5, anchor="center")

        pause_label = tk.Label(self.pause_frame, text="게임 일시 정지", font=("Helvetica", 18))
        pause_label.pack(pady=10)

        resume_btn = tk.Button(self.pause_frame, text="일시 정지 해제", command=self.resume_game, font=("Helvetica", 14))
        resume_btn.pack(pady=5)

        restart_btn = tk.Button(self.pause_frame, text="재시작", command=self.restart_game, font=("Helvetica", 14))
        restart_btn.pack(pady=5)

        level_btn = tk.Button(self.pause_frame, text="레벨 선택", command=self.level_selection, font=("Helvetica", 14))
        level_btn.pack(pady=5)
        
    def resume_game(self):
        self.paused = False
        self.canvas.bind("<KeyPress>", self.key_down)
        self.canvas.bind("<KeyRelease>", self.key_up)
        self.pause_frame.destroy()
        self.canvas.focus_set()
        self.main_proc()
        
    def restart_game(self):
        self.reset()
        self.resume_game()

def startMaze():
    root = tk.Tk()
    game = MazeGame(root)
    root.mainloop()

if __name__ == "__main__":
    startMaze()