# 🖊 Text2Sketch: Generate 2D Floor Plans from Text

*Text2Sketch* is an AI-powered system that takes natural language descriptions and generates 2D floor plans using a GAN-based architecture. This project merges NLP and Computer Vision to automate architectural sketching.


---

## 📌 Project Overview

- *Goal:* Convert text inputs into structured 2D floor plan images
- *Approach:* Text encoder + GAN architecture
- *Team:* Rushil Shah, Vivek, Kunal, Vanshika

---

## ⚙ Tools & Technologies

| Category          | Tools/Technologies       |
|-------------------|--------------------------|
| Programming       | Python                   |
| Deep Learning     | PyTorch, DCGAN           |
| Data Handling     | NumPy, JSON              |
| Visualization     | Matplotlib               |
| Version Control   | Git, GitHub              |
| Environment       | Windows 10, VS Code      |

---

## 🧩 Project Structure
Text2Sketch/
├── data_preprocessing.py # Preprocessing & augmentation (Vivek)
├── model_architecture.py # GAN with text encoder (Rushil)
├── train_model.py # Training loop & custom loss (Kunal)
├── visualize_output.py # Visualization module (Vanshika)
├── requirements.txt
├── README.md
└── outputs/

Copy

---

## 🛠 Installation & Usage

### Requirements
```bash
git clone https://github.com/yourusername/text2sketch.git
cd text2sketch
pip install -r requirements.txt
Workflow
Prepare Dataset:

Place images in data/images/

Add JSON descriptions in data/labels/

Preprocess Data:

bash
Copy
python data_preprocessing.py
Train Model:

bash
Copy
python train_model.py
Generate Outputs:

bash
Copy
python visualize_output.py
📊 Results
Description	Expected Output	Actual Output
"1BHK flat with toilet and hall"	Basic layout	✅ Accurate with minor scaling
"2 bedrooms + 1 kitchen"	3-room layout	✅ Precise configuration
👥 Contributions
Member	Contribution
Rushil	Text encoder integration & GAN design
Vivek	JSON processing & data augmentation
Kunal	Training pipeline & loss functions
Vanshika	Visualization modules
Innovation: First system bridging NLP and floor plan generation with GANs

⏳ Development Timeline
Weekly milestones via GitHub Projects

Continuous integration using GitHub Actions

Code reviews through PRs

📜 License
MIT License © 2023 Text2Sketch Team

🙏 Acknowledgements
Faculty advisors

PyTorch documentation

Open-source community resources
