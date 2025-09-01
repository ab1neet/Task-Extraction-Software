# Task Extraction and Financial Recommendation Software

The Task Extraction software is designed to automatically extract tasks, deadlines, and recipients from natural language text inputs using Named Entity Recognition (NER).

### Files:
- `doc_dbte2.pdf`: Documentation for the Task Extraction software
- `train_data.py`: Contains the training data for the NER model
- `dbapp.py`: The main application file for task extraction
- `dbmodel.py`: Contains the model definition and training code

### Dependencies:
- Python 3.7+
- transformers
- torch
- datasets
- sklearn
- numpy
- seqeval
- nltk

### Usage:
1. Ensure all dependencies are installed.
2. Run `dbmodel.py` to train the NER model (if not already trained).
3. Execute `dbapp.py` to start the task extraction system.
4. Enter task-related text when prompted.
5. Review the extracted tasks, deadlines, and recipients displayed in the console.
