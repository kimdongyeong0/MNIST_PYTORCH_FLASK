from flask import Flask, request, Response, jsonify, render_template
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as f
import numpy as np

app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "mps")


# Define the neural network structure, must match the trained model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512).to(device)
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc3 = nn.Linear(256, 128).to(device)
        self.fc4 = nn.Linear(128, 64).to(device)
        self.fc5 = nn.Linear(64, 32).to(device)
        self.fc6 = nn.Linear(32, 10).to(device)

    def forward(self, x):
        x = x.float()
        h1 = f.relu6(self.fc1(x.view(-1, 784)))
        h2 = f.relu6(self.fc2(h1))
        h3 = f.relu6(self.fc3(h2))
        h4 = f.relu6(self.fc4(h3))
        h5 = f.relu6(self.fc5(h4))
        h6 = self.fc6(h5)

        return f.log_softmax(h6, dim=1)


# Load the trained model
model = Net()
model_path = "NN.Model/model_weights.pth"  # Update this path to where you've stored the model file
model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


@app.route("/")
def index():
    return render_template("index.html")


# Prediction endpoint
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code


@app.route("/predict", methods=["GET", "POST"])
def predict():
    print("/predict request")
    input_data = request.json["inputdata"]
    print(input_data)
    image_matrix = process_tensor(input_data)
    image_matrix = image_matrix
    # 예측 수행
    with torch.no_grad():
        prediction = model(image_matrix)
        predicted_digit = torch.argmax(prediction).item()
    # 예측된 숫자를 문자열로 반환
    return str(predicted_digit)


def process_tensor(number_data):
    rows = number_data.strip().split("\n")
    tensor_data = []

    for row in rows:
        values = [float(val) for val in row.strip().split(",")]
        tensor_data.append(values)

    tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


@app.route("/update", methods=["post"])
def update_model():
    input_data = request.json["inputdata"]
    label = request.json["label"]
    print(label)
    image_matrix = process_tensor(input_data)
    label = torch.tensor([int(label)], dtype=torch.long).to(device)

    output = model(image_matrix)
    loss = criterion(output, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), "mnist_model.pth")

    return str("model updated successfully")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
