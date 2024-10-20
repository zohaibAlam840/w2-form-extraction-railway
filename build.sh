
# Install system dependencies, including Poppler
echo "Installing system dependencies..."
apt-get update && apt-get install -y poppler-utils

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
