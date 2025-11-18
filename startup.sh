#!/bin/bash

# NL2SQL Testbench Setup Script
echo "=€ Setting up NL2SQL Testbench..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "   Python 3.12 not found, using python3 instead"
    PYTHON_CMD="python3"
else
    echo " Found Python 3.12"
    PYTHON_CMD="python3.12"
fi

# Create and activate virtual environment
echo "=æ Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo " Virtual environment created"
else
    echo " Virtual environment already exists"
fi

# Activate virtual environment
echo "=' Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "=Ú Installing Python dependencies..."
pip install -r requirements.txt

# Clone Spider repository if it doesn't exist
if [ ! -d "spider" ]; then
    echo "=Ê Cloning Spider dataset repository..."
    git clone https://github.com/taoyds/spider.git
    echo " Spider dataset cloned"
else
    echo " Spider dataset already exists"
fi

# Check if Spider dataset has the required files
if [ -f "spider/evaluation.py" ] && [ -f "spider/process_sql.py" ]; then
    echo " Spider evaluation scripts found"
else
    echo "   Warning: Spider evaluation scripts not found in expected locations"
    echo "   Make sure spider/evaluation.py and spider/process_sql.py exist"
fi

# Create necessary directories
echo "=Á Creating directories..."
mkdir -p results
mkdir -p plots
mkdir -p configs

# Check for secrets file
if [ ! -f "secrets.txt" ]; then
    echo "= Creating secrets.txt template..."
    echo "# Add your Novita AI API key here" > secrets.txt
    echo "NOVITA_API_KEY=your_api_key_here" >> secrets.txt
    echo "   Please add your Novita AI API key to secrets.txt"
else
    echo " secrets.txt already exists"
fi

# Make scripts executable
chmod +x run_testbench.py
chmod +x examples/quick_start.py
chmod +x examples/comprehensive_evaluation.py

echo ""
echo "<‰ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your Novita AI API key to secrets.txt:"
echo "   echo 'NOVITA_API_KEY=your_actual_api_key' > secrets.txt"
echo ""
echo "2. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "3. Run a quick test:"
echo "   python run_testbench.py quick"
echo ""
echo "4. Or try the examples:"
echo "   python examples/quick_start.py"
echo ""
echo "5. For comprehensive evaluation:"
echo "   python examples/comprehensive_evaluation.py"
echo ""

# Test basic functionality
echo ">ê Testing basic functionality..."
source .venv/bin/activate
python -c "
try:
    import pandas, numpy, matplotlib, seaborn, aiohttp, requests, tqdm
    print(' All required packages imported successfully')
except ImportError as e:
    print(f'L Import error: {e}')

try:
    from src.config import ConfigPresets
    from src.novita_client import ModelRegistry
    print(' Testbench modules imported successfully')
    print(f'=Ë Available presets: {ConfigPresets.list_presets()}')
    print(f'> Available model families: {ModelRegistry.get_model_families()}')
except ImportError as e:
    print(f'L Testbench import error: {e}')
"

echo ""
echo "<Á Startup script completed!"