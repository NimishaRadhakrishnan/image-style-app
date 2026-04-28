🎨 AI Image Analysis & Stylization Web App:

An AI-powered web app that takes an image, understands what’s in it, and transforms it into different artistic styles using deep learning and image processing.


🚀 Overview:

This project runs on Flask and brings together computer vision, transformer models, and image processing to do three main things:

* 🧠 Understand what the image shows (caption + analysis)
* 🎨 Turn it into different art styles
* ⚡ Show results quickly through a simple web interface

✨ Features:

🧠 Intelligent Image Analysis:

* Generates captions using BLIP
* Identifies:

  * Main subject (person, animal, object)
  * Mood and lighting
  * Scene details
* Recommends a style that fits the image

🎨 Artistic Stylization:

You can apply styles like:

* Cartoon
* Anime
* Pencil sketch
* Digital painting
* Watercolour
* Oil painting
* Van Gogh-inspired
* Cyberpunk

⚡ Fast Processing:

* Enhances and resizes images before processing
* Uses NumPy and PIL for efficient handling
* Returns results in Base64 for quick display in the browser


🛠️ Tech Stack:

| Category         | Tools Used            |
| ---------------- | --------------------- |
| Backend          | Flask                 |
| AI Models        | CLIP, BLIP            |
| Deep Learning    | PyTorch               |
| Image Processing | PIL, NumPy            |
| Frontend         | HTML, CSS (via Flask) |


📂 Project Structure:

├── app.py
├── static/
├── uploads/
├── outputs/
├── requirements.txt
├── README.md


⚙️ Installation:
1️⃣ Clone the repository

bash:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

2️⃣ Create a virtual environment:

bash:
python3 -m venv venv
source venv/bin/activate   # Mac/Linux

3️⃣ Install dependencies

bash:
pip install -r requirements.txt


▶️ Run the Application:

bash:
python app.py


Then open:
http://localhost:5001

🔄 Workflow:

1. Upload an image
2. The app cleans and resizes it
3. AI generates a caption and analysis
4. A suitable style is suggested
5. The style is applied
6. You see the final result


📸 Example Use Cases:

* Creating content for social media
* Building AI photo editing tools
* Design experiments and creative projects
* Learning computer vision and deep learning

 ⚠️ Notes:

* Large models (around 1–2GB) download on first run
* Make sure your system has enough RAM
* Don’t upload `venv/` or model files to GitHub

🔮 Future Improvements:

* Add GAN-based style transfer
* Deploy with Docker or cloud services
* Introduce user accounts
* Speed up model loading

🤝 Contributing:

Contributions are welcome. Fork the repo, make changes, and open a pull request.

📜 License:

This project is open-source under the MIT License.

⭐ Support:

If you found this useful, consider starring the repo on GitHub.

💡 Author:

Nimisha R
Aspiring Data Scientist | AI Enthusiast
