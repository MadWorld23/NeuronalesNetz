import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # für Mac/VS Code oft stabiler
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Interaktiven Modus aktivieren
plt.ion()

# =========================================
# 1) Einstellungen
# =========================================
np.random.seed(42)
learning_rate = 0.1
epochs = 5000

# =========================================
# 2) CSV laden
# =========================================
file_path = os.path.join(os.path.dirname(__file__), "rgb.csv")
# print("Pfad:", file_path)
# print("Existiert:", os.path.exists(file_path))

# all_data = pd.read_csv(file_path, sep=";")
all_data = pd.read_csv(file_path)
# print("\nErste Zeilen der CSV:")
# print(all_data.head())

# =========================================
# 3) Daten vorbereiten
# =========================================
all_inputs = all_data.iloc[:, 0:3].values / 255.0
all_outputs = all_data.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    all_inputs, all_outputs, test_size=1/3, random_state=42
)

X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

n_train = X_train.shape[1]

# =========================================
# 4) Netzwerk initialisieren
# =========================================
w_hidden = np.random.randn(3, 3) * 0.1
b_hidden = np.zeros((3, 1))

w_output = np.random.randn(1, 3) * 0.1
b_output = np.zeros((1, 1))

# =========================================
# 5) Funktionen
# =========================================
def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(float)

def logistic(x):
    x = np.clip(x, -500, 500)  # stabiler gegen Overflow
    return 1 / (1 + np.exp(-x))

def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

def compute_loss(Y, A2):
    eps = 1e-10
    return -np.mean(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps))

def predict(X):
    _, _, _, A2 = forward_prop(X)
    return (A2 >= 0.5).astype(int)

def predict_raw(r, g, b):
    x = np.array([[r], [g], [b]]) / 255.0
    _, _, _, A2 = forward_prop(x)
    return int(A2[0, 0] >= 0.5)

# In diesem Datensatz gilt:
# 1 = dunkle Schrift
# 0 = helle Schrift
def predict_text_color(r, g, b):
    prediction = predict_raw(r, g, b)
    if prediction == 1:
        return "black", "DUNKLE SCHRIFT"
    else:
        return "white", "HELLE SCHRIFT"

def prepare_single_window(title="RGB-Vorhersage"):
    plt.figure(title, figsize=(8, 5))
    plt.clf()

def show_color_prediction(r, g, b):
    prepare_single_window("RGB-Vorhersage")

    text_color, label = predict_text_color(r, g, b)
    bg = np.array([[[r / 255, g / 255, b / 255]]])

    plt.imshow(bg)
    plt.axis("off")

    plt.text(
        0.5,
        0.5,
        label,
        color=text_color,
        fontsize=20,
        ha="center",
        va="center",
        transform=plt.gca().transAxes
    )

    plt.title(f"RGB = ({r}, {g}, {b})")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def show_loss():
    prepare_single_window("Loss")

    plt.plot(loss_history)
    plt.title("Loss während des Trainings")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def show_matrix():
    prepare_single_window("Verwechslungsmatrix")

    cm = confusion_matrix(Y_test.flatten(), Y_pred_test.flatten())
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["helle Schrift", "dunkle Schrift"]
    )
    disp.plot(ax=plt.gca())
    plt.title("Verwechslungsmatrix")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def show_predictions(num_samples=12):
    prepare_single_window("Testvorhersagen")

    max_samples = X_test.shape[1]
    num_samples = min(num_samples, max_samples)

    indices = np.random.choice(X_test.shape[1], num_samples, replace=False)

    rows = int(np.ceil(num_samples / 4))
    cols = min(4, num_samples)

    fig = plt.gcf()
    fig.set_size_inches(12, 3 * rows)
    fig.clf()

    for i, idx in enumerate(indices):
        r = int(X_test[0, idx] * 255)
        g = int(X_test[1, idx] * 255)
        b = int(X_test[2, idx] * 255)

        pred = int(Y_pred_test[0, idx])
        true = int(Y_test[0, idx])

        bg_color = np.array([[[r / 255, g / 255, b / 255]]])

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(bg_color)
        ax.axis("off")

        text_color = "black" if pred == 1 else "white"
        pred_text = "dunkle Schrift" if pred == 1 else "helle Schrift"
        status = "OK" if pred == true else "FALSCH"

        ax.text(
            0.5,
            0.5,
            f"{pred_text}\n{status}",
            color=text_color,
            fontsize=11,
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox=dict(facecolor="gray", alpha=0.3)
        )

        ax.set_title(f"RGB=({r}, {g}, {b})", fontsize=10)

    fig.suptitle("Vorhersagen des Netzes auf Testdaten")
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def show_help():
    print("\nVerfügbare Befehle:")
    print("  help          -> zeigt diese Hilfe")
    print("  loss          -> zeigt die Loss-Kurve")
    print("  matrix        -> zeigt die Verwechslungsmatrix")
    print("  test          -> zeigt mehrere zufällige Testfarben")
    print("  test 8        -> zeigt z. B. 8 Testfarben")
    print("  acc           -> zeigt die Testgenauigkeit")
    print("  exit          -> beendet das Programm")
    print("  R G B         -> z. B. 20 20 20")
    print()

# =========================================
# 6) Training
# =========================================
loss_history = []

for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_prop(X_train)

    loss = compute_loss(Y_train, A2)
    loss_history.append(loss)

    dZ2 = A2 - Y_train
    dW2 = (dZ2 @ A1.T) / n_train
    dB2 = np.sum(dZ2, axis=1, keepdims=True) / n_train

    dA1 = w_output.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (dZ1 @ X_train.T) / n_train
    dB1 = np.sum(dZ1, axis=1, keepdims=True) / n_train

    w_output[:] = w_output - learning_rate * dW2
    b_output[:] = b_output - learning_rate * dB2
    w_hidden[:] = w_hidden - learning_rate * dW1
    b_hidden[:] = b_hidden - learning_rate * dB1

    # if epoch % 500 == 0:
    #    print(f"Epoch {epoch}: Loss = {loss:.4f}")

# =========================================
# 7) Testdaten auswerten
# =========================================
Y_pred_test = predict(X_test)
accuracy = np.mean(Y_pred_test == Y_test)
# print(f"\nTest Accuracy: {accuracy:.4f}")

# print("\nBeispiel-Vorhersagen:")
# for r, g, b in [(20, 20, 20), (240, 240, 240), (120, 180, 50)]:
#    _, label = predict_text_color(r, g, b)
#    print(f"({r}, {g}, {b}) -> {label}")

# =========================================
# 8) Interaktive Schleife
# =========================================
print("\nGib RGB-Werte oder Befehle ein.")
show_help()

while True:
    user_input = input("Eingabe: ").strip().lower()

    if user_input == "exit":
        print("Programm beendet.")
        plt.close("all")
        break

    elif user_input == "help":
        show_help()

    elif user_input == "loss":
        show_loss()

    elif user_input == "matrix":
        show_matrix()

    elif user_input == "acc":
        print(f"Test Accuracy: {accuracy:.4f}")

    elif user_input.startswith("test"):
        parts = user_input.split()

        if len(parts) == 1:
            show_predictions()
        elif len(parts) == 2 and parts[1].isdigit():
            show_predictions(int(parts[1]))
        else:
            print("Benutzung: test oder test 8")

    else:
        try:
            r, g, b = map(int, user_input.split())

            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                print("Bitte nur Werte zwischen 0 und 255 eingeben.")
                continue

            show_color_prediction(r, g, b)

        except ValueError:
            print("Unbekannte Eingabe.")
            print("Beispiele:")
            print("  20 20 20")
            print("  loss")
            print("  matrix")
            print("  test")
            print("  help")
            print("  exit")