# In VS Code terminal
cd mu-predictor

# Install in development mode
pip install -e .

# Test it
python -c "from mu_predictor import MuPredictor; print(MuPredictor(4,1,1,1).mu)"

# Run example
python examples/base_case.py
