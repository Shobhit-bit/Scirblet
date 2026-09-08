import tkinter as tk
import cv2
import numpy as np
import sympy
from PIL import Image, ImageDraw
from keras.models import load_model

MODEL_PATH = 'equation_reader_v2.keras'
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','+', '-', '*', '%', '[', ']']
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 300
STROKE_WIDTH = 10

def preprocess_array(gray_img, symbols_are_dark_in_source=True):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh_type = cv2.THRESH_BINARY if symbols_are_dark_in_source else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
    return binary

def thin_strokes(binary_image):
    inv = cv2.bitwise_not(binary_image)
    thinned_inv = cv2.ximgproc.thinning(inv)
    return cv2.bitwise_not(thinned_inv)

def _get_raw_boxes(binary_image, min_area=40, min_dim=5):
    inv = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area or w < min_dim or h < min_dim:
            continue
        boxes.append((x, y, w, h))
    return boxes

def segment_symbols(binary_image, min_area=40, min_dim=5, x_gap_thresh=15):
    boxes = sorted(_get_raw_boxes(binary_image, min_area, min_dim), key=lambda b: b[0])
    merged, used = [], [False] * len(boxes)
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if used[i]:
            continue
        cx1, cy1, cx2, cy2 = x1, y1, x1 + w1, y1 + h1
        used[i] = True
        stroke_count = 1
        changed = True
        while changed:
            changed = False
            for j, (x2, y2, w2, h2) in enumerate(boxes):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
                h_close = not (bx1 > cx2 + x_gap_thresh or bx2 < cx1 - x_gap_thresh)
                if h_close:
                    used[j] = True
                    cx1, cy1 = min(cx1, bx1), min(cy1, by1)
                    cx2, cy2 = max(cx2, bx2), max(cy2, by2)
                    stroke_count += 1
                    changed = True
        merged.append({"box": (cx1, cy1, cx2 - cx1, cy2 - cy1), "strokes": stroke_count})

    return sorted(merged, key=lambda m: m["box"][0])


def is_equals_sign(entry, w_over_h_thresh=0.9):
    x, y, w, h = entry["box"]
    return entry["strokes"] >= 2 and (w / h) > w_over_h_thresh

def normalize_symbol(binary_image, box, target_size=28, padding_ratio=0.2):
    x, y, w, h = box
    crop = binary_image[y:y + h, x:x + w]
    side = max(w, h)
    pad = int(side * padding_ratio)
    side += pad * 2
    square = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized.astype('float32') / 255.0

def classify_symbols(model, binary_image, entries, categories):
    equation_chars = []
    for entry in entries:
        if is_equals_sign(entry):
            continue
        symbol = normalize_symbol(binary_image, entry["box"])
        symbol_input = np.expand_dims(symbol, axis=(0, -1))
        prediction = model.predict(symbol_input, verbose=0)
        class_index = int(np.argmax(prediction))
        equation_chars.append(categories[class_index])
    return ''.join(equation_chars)


def solve_equation(equation_str, percent_as_modulo=True):
    expr_str = equation_str.replace('[', '(').replace(']', ')')
    if not percent_as_modulo:
        expr_str = expr_str.replace('%', '/100')
    try:
        expr = sympy.sympify(expr_str)
        return sympy.N(expr)
    except (sympy.SympifyError, TypeError, ValueError) as e:
        return f"Could not evaluate '{equation_str}' (parsed as '{expr_str}'): {e}"

def scribble_array_to_answer(gray_img, model, categories, apply_thinning=False, percent_as_modulo=True):
    binary = preprocess_array(gray_img, symbols_are_dark_in_source=True)
    if apply_thinning:
        binary = thin_strokes(binary)
    entries = segment_symbols(binary)
    if not entries:
        raise ValueError("No symbols detected on the pad.")
    equation_str = classify_symbols(model, binary, entries, categories)
    if not equation_str:
        raise ValueError("Only an '=' sign (or nothing) was detected.")
    result = solve_equation(equation_str, percent_as_modulo=percent_as_modulo)
    return equation_str, result

class ScribblePad:
    def __init__(self, root, model, categories):
        self.root = root
        self.model = model
        self.categories = categories
        root.title("Equation Scribble Pad")
        root.resizable(False, False)
        tk.Label(
            root, text="Write your equation below, then hit Solve.",
            font=('Arial', 12)
        ).pack(pady=(10, 0))
        self.canvas = tk.Canvas(
            root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
            bg='white', cursor='pencil', highlightthickness=1,
            highlightbackground='#888'
        )
        self.canvas.pack(padx=10, pady=10)
        self.image = Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.canvas.bind('<ButtonPress-1>', self.start_stroke)
        self.canvas.bind('<B1-Motion>', self.draw_stroke)
        self.canvas.bind('<ButtonRelease-1>', self.end_stroke)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=(0, 12))
        tk.Button(
            button_frame, text='Solve', command=self.solve, width=14,
            bg='#4CAF50', fg='white', font=('Arial', 11, 'bold')
        ).pack(side='left', padx=6)
        tk.Button(
            button_frame, text='Clear', command=self.clear, width=14,
            font=('Arial', 11)
        ).pack(side='left', padx=6)
    def start_stroke(self, event):
        self.last_x, self.last_y = event.x, event.y
    def draw_stroke(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=STROKE_WIDTH, fill='black',
                capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=0, width=STROKE_WIDTH
            )
        self.last_x, self.last_y = x, y
    def end_stroke(self, event):
        self.last_x, self.last_y = None, None
    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), color=255)
        self.draw = ImageDraw.Draw(self.image)
    def solve(self):
        gray = np.array(self.image)
        try:
            equation_str, result = scribble_array_to_answer(
                gray, self.model, self.categories
            )
            self.show_result(equation_str, result, error=False)
        except ValueError as e:
            self.show_result(None, str(e), error=True)
    def show_result(self, equation_str, result, error):
        popup = tk.Toplevel(self.root)
        popup.title("Result")
        popup.geometry("340x170")
        popup.resizable(False, False)
        if error:
            tk.Label(
                popup, text="Couldn't read a valid equation.",
                fg='#c0392b', font=('Arial', 13, 'bold'), wraplength=300
            ).pack(pady=(16, 4))
            tk.Label(popup, text=result, wraplength=300).pack(pady=4)
        else:
            tk.Label(
                popup, text=f"Equation: {equation_str}", font=('Arial', 14)
            ).pack(pady=(16, 6))
            tk.Label(
                popup, text=f"Answer: {result}", font=('Arial', 17, 'bold'), fg='#2e7d32'
            ).pack(pady=4)

        tk.Button(popup, text='OK', command=popup.destroy, width=10).pack(pady=14)
if __name__ == '__main__':
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)

    root = tk.Tk()
    app = ScribblePad(root, model, CATEGORIES)
    root.mainloop()
