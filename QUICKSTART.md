# 🚀 Quick Start Guide

## Option 1: Local Setup

### Prerequisites
- Python 3.8+
- Git

### Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/AhmedAl-Mahdi/Aircraft-Classifier.git
   cd Aircraft-Classifier
   ```

2. **Run setup script**
   ```bash
   python setup.py
   ```

3. **Train the model (optional)**
   ```bash
   jupyter notebook aircraft_classifier.ipynb
   ```
   Follow the notebook to train and save the model.

4. **Launch Gradio interface**
   ```bash
   python app.py
   ```

5. **Open browser**
   Navigate to `http://localhost:7860`

## Option 2: Docker Deployment

### Prerequisites
- Docker installed

### Steps
1. **Clone and build**
   ```bash
   git clone https://github.com/AhmedAl-Mahdi/Aircraft-Classifier.git
   cd Aircraft-Classifier
   docker build -t aircraft-classifier .
   ```

2. **Run container**
   ```bash
   docker run -p 7860:7860 aircraft-classifier
   ```

3. **Access application**
   Open `http://localhost:7860` in your browser

## Option 3: Cloud Deployment

### Hugging Face Spaces
1. Fork this repository
2. Create a new Space on Hugging Face
3. Link your forked repository
4. The app will deploy automatically

### Google Colab
1. Open the notebook in Colab
2. Add `!python app.py` in a new cell
3. Use ngrok for public access

## Troubleshooting

### Common Issues

**Model not found error:**
- Train the model using the Jupyter notebook
- Or run with random weights for demo

**Memory issues:**
- Use CPU mode: Set `device = 'cpu'` in app.py
- Reduce batch size in training

**Port already in use:**
- Change port in app.py: `server_port=7861`

**Dependencies issues:**
- Update pip: `pip install --upgrade pip`
- Install requirements: `pip install -r requirements.txt`

## Support

- 📚 Check the main README.md for detailed documentation
- 🐛 Report issues on GitHub
- 💡 Feature requests welcome!