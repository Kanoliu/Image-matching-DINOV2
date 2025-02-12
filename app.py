# %%
import io
import base64
import pickle
import numpy as np
import torch
import random  # Import the random module
from flask import Flask, render_template, request
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# -------------------------
# 1. Load Dataset Features
# -------------------------
# features.pkl is assumed to be a dictionary mapping image paths (relative to the static folder)
# to their corresponding feature vectors (stored as numpy arrays).
with open('features.pkl', 'rb') as f:
    features_dict = pickle.load(f)
    features_dict = { key.replace("\\", "/"): value for key, value in features_dict.items() }

# Create lists of image paths and features
dataset_paths = list(features_dict.keys())
dataset_features = np.array(list(features_dict.values()))
print(f"Loaded features for {len(dataset_paths)} images.")

# ------------------------------
# 2. Set Up the Model & Processor
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.to(device)
model.eval()  # Set model to evaluation mode

# ----------------------------
# 3. Define Cosine Similarity
# ----------------------------
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    # Adding a small constant to avoid division by zero.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ----------------------------
# 4. Create the Flask App
# ----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                # Open and convert the uploaded image to RGB.
                test_img = Image.open(file.stream).convert("RGB")
            except Exception as e:
                return f"Error processing image: {e}"

            # ------------------------------
            # 3. Extract Feature from Uploaded Image
            # ------------------------------
            inputs = processor(images=test_img, return_tensors="pt", padding=True)
            # Move tensors to the same device as the model.
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            # Assume the model returns a 'pooler_output'
            test_feature = outputs.pooler_output.cpu().numpy().squeeze()

            # -------------------------------------------
            # 4. Compute Similarities with Dataset Features
            # -------------------------------------------
            similarities = np.array([
                cosine_similarity(test_feature, feat) for feat in dataset_features
            ])

            # ------------------------------
            # 5. Retrieve Top-5 Matches and Generate Random Price
            # ------------------------------
            top_k = 5
            top_indices = similarities.argsort()[::-1][:top_k]
            # Create a list of tuples: (image_path, random_price)
            results = [(dataset_paths[i], random.randint(45, 100)) for i in top_indices]

            # --------------------------------------------------
            # Convert the uploaded test image to base64 for display.
            # --------------------------------------------------
            buffered = io.BytesIO()
            test_img.save(buffered, format="JPEG")
            test_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Render the results page, passing the list of matches and the test image.
            return render_template("results.html", results=results, test_img=test_img_str)
    # For GET requests, simply show the upload form.
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


