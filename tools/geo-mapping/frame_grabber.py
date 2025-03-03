import time
import os
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

# Set an environment variable that has helped some users running snap Firefox.
os.environ["MOZ_DISABLE_RDD_SANDBOX"] = "1"

# Configure Firefox options (running in headless mode; set to False if you want to see the browser)
options = Options()
options.headless = True

# Optionally, if you know the Firefox binary location you want to use:
# options.binary_location = "/snap/bin/firefox"   # or try "/usr/bin/firefox" if available

# Initialize Firefox webdriver; let Selenium create a temporary profile automatically.
driver = webdriver.Firefox(options=options)

# Use your notebook URL
notebook_url = "http://localhost:8888/notebooks/yl4300/project/Multi-Camera-Vision-Pipeline-YQ/tools/geo-mapping/geo-mapper-visualizer.ipynb"
driver.get(notebook_url)

# Wait for the notebook and map to load
time.sleep(10)

# Create an output folder for screenshots
output_folder = 'geomap_img'
os.makedirs(output_folder, exist_ok=True)

# Capture full-page screenshots
num_frames = 100  # Number of frames to capture
for i in range(num_frames):
    time.sleep(0.5)  # Adjust waiting time as needed
    screenshot_filename = os.path.join(output_folder, f"frame_{i:02d}.png")
    driver.save_screenshot(screenshot_filename)
    print(f"Saved {screenshot_filename}")

driver.quit()
