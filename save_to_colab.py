# 1. First run (mount drive)
from google.colab import drive
drive.mount('/content/drive')

# 2. After training
def save_model():
    import shutil
    shutil.copytree('./results', '/content/drive/MyDrive/email_model/results', dirs_exist_ok=True)
    print("Model saved!")

# 3. Next session  
def load_model():
    import shutil
    shutil.copytree('/content/drive/MyDrive/email_model/results', './results', dirs_exist_ok=True)
    print("Model loaded!")

# Use them
save_model()  # After training
load_model()  # At start of new session