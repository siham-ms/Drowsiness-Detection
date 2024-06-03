from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.image import Image as KivyImage
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import torch
import numpy as np
import cv2

class LogoScreen(Screen):
    pass

class DetectionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'yolov5\runs\train\exp2\weights\last.pt')
        self.camera = Camera(resolution=(640, 480), play=True)
        self.add_widget(self.camera)
        self.frame_resized = np.zeros((480, 640, 3), dtype=np.uint8)
        Clock.schedule_interval(self.detect, 1/30)
        self.detecting = True

    def detect(self, dt):
        if not self.detecting:
            return

        frame = self.camera.texture
        if frame is not None:
            buf = frame.pixels
            shape = (frame.size[1], frame.size[0], 4)
            frame = np.frombuffer(buf, dtype='uint8').reshape(shape)

            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            self.frame_resized = cv2.resize(frame, (640, 480))
            results = self.model(self.frame_resized)
            img = np.squeeze(results.render())

            buf1 = cv2.flip(img, 0).tostring()
            texture1 = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
            texture1.blit_buffer(buf1, colorfmt='rgb', bufferfmt='ubyte')
            self.camera.texture = texture1

    def stop_detection(self):
        self.detecting = False

class DrowsinessDetectionApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.logo_screen = LogoScreen(name="logo")
        self.sm.add_widget(self.logo_screen)
        self.detection_screen = DetectionScreen(name="detection")
        self.sm.add_widget(self.detection_screen)

        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (1, 1)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        self.logo_image = KivyImage(source="logo.png")
        self.window.add_widget(self.logo_image)

        self.greeting = Label(
            text="Begin your vehicle ride safely",
            font_size=20,
            color='#00FFCE'
        )
        self.window.add_widget(self.greeting)

        self.button = Button(
            text="Start",
            size_hint=(0.5, 0.5),
            bold=True,
            background_color='#00FFCE',
        )
        self.button.bind(on_press=self.switch_to_detection)
        self.window.add_widget(self.button)

        self.logo_screen.add_widget(self.window)

        return self.sm

    def switch_to_detection(self, instance):
        self.sm.current = "detection"

    def on_stop(self):
        self.detection_screen.stop_detection()

# Run the app
if __name__ == "__main__":
    DrowsinessDetectionApp().run()
