echo "Virtual env activated"
.\.venv\Scripts\activate

# echo "Running (01) data ingestion"
# python -m scripts.01-ingest_data

# echo "Running (02) feature building"
# python -m scripts.02-build_features

echo "Running (03) agent training"
python -m scripts.03-train_agent

echo "Running (04) agent testing"
python -m scripts.04-test_agent
